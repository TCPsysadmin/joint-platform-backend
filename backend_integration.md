# Backend Integration Guide

API reference for building the **video-agent** frontend. This document contains
everything a frontend developer (or AI coding agent) needs to integrate with the
backend: authentication, every endpoint, request/response shapes, and the
Server-Sent Events (SSE) streaming protocol.

> Source of truth: [orchestrator/main.py](orchestrator/main.py),
> [orchestrator/auth.py](orchestrator/auth.py),
> [orchestrator/session_manager.py](orchestrator/session_manager.py).

---

## 1. Overview

The backend is a FastAPI service. A user chats with an AI agent that recommends
short-form video clips. The core flow is:

1. Create (or reuse) a **session**.
2. Send a **chat message** → the server streams progress over SSE and may pause,
   asking the user to approve a clip recommendation.
3. If paused, the user **confirms** (approve/reject) → the server resumes,
   publishes the clip (on approve), and streams to completion.

A single chat turn can either finish immediately (`done`) or pause for
confirmation (`awaiting_confirmation`). There is **no polling** — consume the SSE
stream until it ends.

---

## 2. Base URL & environment

| Environment | Base URL |
|-------------|----------|
| Local dev   | `http://127.0.0.1:8000` |
| Production  | Your deployed host (e.g. Render) |

- **CORS:** controlled server-side by `CORS_ORIGINS`. In production this must
  include your frontend origin. `allow_credentials` is enabled.
- **SSE proxies:** the server sets `X-Accel-Buffering` for nginx; no client
  action needed.

---

## 3. Authentication

Auth is handled by **Supabase Auth (JWT)**. Every endpoint except `/health` and
`/readyz` requires a Bearer token.

```
Authorization: Bearer <supabase_access_token>
```

> **Important:** The client is identified **solely** by this token. The backend
> verifies the JWT with Supabase, then looks up which tenant (`client_id`) the
> user belongs to via their profile. **Do not** send a client ID header — the
> frontend never needs to know or pass `client_id`.

### How the frontend obtains a token

Authenticate the user directly against Supabase Auth (using the Supabase JS
client or a raw call), then use the returned `access_token` as the Bearer value:

```
POST <SUPABASE_URL>/auth/v1/token?grant_type=password
Headers:
  apikey: <SUPABASE_ANON_KEY>
  Content-Type: application/json
Body:
  { "email": "user@example.com", "password": "password" }
→ Response: { "access_token": "...", "refresh_token": "...", ... }
```

Use `access_token` as `Authorization: Bearer <access_token>` on all API calls.
Refresh it via Supabase when it expires.

> User accounts are provisioned by an admin (see `POST /admin/users`). There is
> no public sign-up endpoint on this backend.

### Auth error responses

| Status | When |
|--------|------|
| `401`  | Missing/empty/invalid/expired token, or user has no linked client profile. Body: `{ "detail": "..." }` |
| `403`  | Accessing a session you don't own, or admin endpoint without the admin key. |

---

## 4. Endpoints

### Quick reference

| Method | Path | Auth | Streaming | Purpose |
|--------|------|------|-----------|---------|
| POST | `/chat` | User JWT | SSE | Send a message; stream agent progress |
| POST | `/confirm` | User JWT | SSE | Approve/reject a pending recommendation |
| POST | `/sessions` | User JWT | JSON | Pre-create a session |
| GET | `/sessions` | User JWT | JSON | List the user's active sessions |
| GET | `/sessions/{session_id}/messages` | User JWT | JSON | Full message history for a session |
| POST | `/admin/clients` | Service key | JSON | (Admin) provision a tenant |
| POST | `/admin/users` | Service key | JSON | (Admin) create + link a user |
| GET | `/health` | none | JSON | Liveness |
| GET | `/readyz` | none | JSON | Readiness (DB reachable) |

---

### 4.1 `POST /chat` — send a message (SSE)

Sends a user message into a session and streams the agent's progress. Auto-creates
the session if it doesn't exist yet (first message provisions it; the message is
used as the default session title).

**Headers**
```
Authorization: Bearer <token>
Content-Type: application/json
Accept: text/event-stream
```

**Request body**
```json
{
  "session_id": "string (any stable id; a UUID is recommended)",
  "message": "Find me a punchy hook clip about consistency"
}
```

**Response:** `200 OK`, `Content-Type: text/event-stream`. See
[§5 SSE protocol](#5-sse-streaming-protocol) for event details.

The stream emits one or more `node` events (progress), then **exactly one**
terminal event:
- `awaiting_confirmation` — a recommendation is ready; call `POST /confirm` next.
- `done` — the turn completed with no confirmation needed (e.g. chit-chat).
- `error` — the turn failed; body has `detail`.

---

### 4.2 `POST /confirm` — approve or reject (SSE)

Resumes a session that is paused awaiting confirmation. Call this **only** after a
`/chat` stream ended with `awaiting_confirmation`.

**Headers:** same as `/chat`.

**Request body**
```json
{
  "session_id": "must match the paused session",
  "action": "approved"   // or "rejected"
}
```

- `action: "approved"` → the agent publishes the clip and streams to `done`.
- `action: "rejected"` → the clip is not posted; the agent acknowledges and
  streams to `done`.

**Response:** `200 OK`, SSE stream of `node` events followed by a `done` event.

**Errors**
| Status | When |
|--------|------|
| `422`  | `action` is not `"approved"` or `"rejected"`. |
| `401` / `403` | Auth / ownership failures. |

---

### 4.3 `POST /sessions` — pre-create a session

Optional. Lets the frontend register a session and obtain its ID before sending
any message (e.g. to show an empty conversation immediately).

**Headers**
```
Authorization: Bearer <token>
Content-Type: application/json
```

**Request body** (all fields optional)
```json
{
  "session_id": "optional; omit to get a server-generated UUID",
  "title": "optional human-friendly title"
}
```

**Response:** `201 Created` — the created session row:
```json
{
  "session_id": "…",
  "user_id": "…",
  "client_id": "…",
  "title": "New conversation",
  "message_count": 0,
  "last_message_at": "2026-05-31T12:00:00+00:00",
  "status": "active",
  "created_at": "2026-05-31T12:00:00+00:00"
}
```

> Note: creating a session with an ID that already exists raises a server error.
> Either pre-create once, or skip this and let `/chat` auto-create.

---

### 4.4 `GET /sessions` — list sessions

Returns the authenticated user's **active** sessions, newest first (max 50).

**Headers**
```
Authorization: Bearer <token>
```

**Response:** `200 OK`
```json
{
  "sessions": [
    {
      "session_id": "…",
      "title": "Find me a hook clip about consistency",
      "message_count": 4,
      "last_message_at": "2026-05-31T12:34:56+00:00",
      "status": "active",
      "created_at": "2026-05-31T12:00:00+00:00"
    }
  ]
}
```

---

### 4.5 `GET /sessions/{session_id}/messages` — message history

Returns the full user/assistant message history for a session, reconstructed from
the agent's persisted state. Only the session owner may read it.

**Headers**
```
Authorization: Bearer <token>
```

**Response:** `200 OK`
```json
{
  "session_id": "…",
  "client_id": "…",
  "turn_status": "idle",
  "message_count": 4,
  "messages": [
    { "role": "user", "content": "Find me a hook clip…" },
    { "role": "assistant", "content": "Here's a clip recommendation…" }
  ]
}
```

- Only `user` and `assistant` roles are returned (internal/system messages are
  filtered out).
- Assistant recommendations are included here, so this endpoint is the canonical
  way to render a conversation on reload.
- `turn_status` is `"idle"`, `"running"`, or `"completed"`. If a user navigates
  away during an active SSE stream, the backend keeps the graph turn running; on
  return, use this value to decide whether to keep a progress state visible and
  refresh messages again.

**Errors**
| Status | When |
|--------|------|
| `404`  | Session not found. |
| `403`  | Session belongs to another user. |

---

### 4.6 `POST /admin/clients` — provision a tenant (admin only)

Not for the end-user frontend. Requires the **service role key** as the Bearer
token. Creates the `clients_registry` row and seeds default brand doctrine.

**Headers**
```
Authorization: Bearer <SUPABASE_SERVICE_KEY>
Content-Type: application/json
```

**Request body**
```json
{
  "slug": "acme",
  "display_name": "Acme Co",
  "source_kind": "gdrive",
  "b2_bucket": null,
  "b2_prefix": null,
  "plan_tier": "standard"
}
```

**Response:** `201 Created`
```json
{
  "client_id": "…",
  "slug": "acme",
  "display_name": "Acme Co",
  "source_kind": "gdrive",
  "b2_bucket": null,
  "b2_prefix": null,
  "plan_tier": "standard"
}
```

Use the returned `client_id` when creating users for that tenant.

---

### 4.7 `POST /admin/users` — create & link a user (admin only)

Not for the end-user frontend. Requires the **service role key** as the Bearer
token. Creates a Supabase Auth user and links them to a tenant.

**Headers**
```
Authorization: Bearer <SUPABASE_SERVICE_KEY>
Content-Type: application/json
```

**Request body**
```json
{
  "email": "user@example.com",
  "password": "…",
  "client_id": "UUID of the tenant to link to",
  "role": "member",
  "display_name": "optional"
}
```

**Response:** `201 Created`
```json
{ "user_id": "…", "client_id": "…", "email": "…", "role": "member" }
```

**Errors:** `403` (not admin), `422` (bad `client_id` UUID), `400` (auth user
creation failed), `404` (`client_id` has not been provisioned).

---

### 4.8 Health checks

| Endpoint | Response |
|----------|----------|
| `GET /health` | `200` → `{ "status": "ok", "version": "<git sha>" }` |
| `GET /readyz`  | `200` → `{ "status": "ok" }`, or `503` if the database is unreachable |

---

## 5. SSE streaming protocol

`/chat` and `/confirm` return an **EventSourceResponse**. Each SSE message has an
`event` name and a JSON-encoded `data` payload. The graph run is detached from the
HTTP stream: if the browser leaves the page and closes the SSE connection, the
server-side turn continues and writes its final state to the session checkpoint.
Re-posting the same in-flight body for the same session attaches to the existing
turn; posting a different message/action while a turn is running returns `409`.

### Event types

| `event` | When | `data` shape |
|---------|------|--------------|
| `node`  | Emitted after each internal step completes (progress ticks). | `{ "node": "<name>", "partial_state": { … } }` — on `/confirm`, only `{ "node": "<name>" }`. |
| `awaiting_confirmation` | Terminal (only on `/chat`). A recommendation is ready and the run is paused. | `{ "recommendation": "<markdown string>" }` |
| `done`  | Terminal. The turn finished; nothing else required. | `{}` |
| `error` | Terminal. The turn failed. | `{ "detail": "<message>" }` |

### `node` names you may observe (for progress UI)

`route_intent`, `chat_response`, `fetch_doctrine`, `retrieve`, `fetch_source`,
`analyze`, `critique`, `refine`, `recommend`, `post_stub`.

You don't need to handle each one specifically — they're useful for a
"thinking…/searching…/analyzing…" progress indicator. Treat unknown node names
gracefully.

### `partial_state` (on `/chat` `node` events)

A compacted snapshot of what the step produced. Large fields are trimmed
server-side. Notable keys you may see:

- `retrieved_segments_summary`: `{ "hit_count": N, "top_segments": [ { segment_id, video_id, start_seconds, end_seconds, score, text_preview } ] }`
- `intent`: one of `"new_request"`, `"follow_up"`, `"chitchat"`
- `candidate_recommendation`, `critique_result`, `final_recommendation`,
  `awaiting_confirmation` may appear depending on the step.

> `partial_state` is for live UI feedback. The full conversation message text is
> **not** included in `node` events — read it from
> `GET /sessions/{id}/messages` or from the `awaiting_confirmation` payload.

### The recommendation payload

The `awaiting_confirmation` event's `recommendation` is a **Markdown string**
(rendered for the user). Display it, then offer Approve / Reject buttons that
call `POST /confirm`.

---

## 6. End-to-end flow (frontend pseudocode)

```js
// 1. (Optional) pre-create a session, or just generate a UUID client-side.
const sessionId = crypto.randomUUID();

// 2. Send a message and consume the SSE stream.
async function sendMessage(text) {
  const res = await fetch(`${BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
    },
    body: JSON.stringify({ session_id: sessionId, message: text }),
  });

  // Parse the SSE stream (fetch + ReadableStream, or a library).
  for await (const evt of parseSSE(res.body)) {
    switch (evt.event) {
      case "node":
        updateProgress(JSON.parse(evt.data));      // optional "thinking…" UI
        break;
      case "awaiting_confirmation":
        const { recommendation } = JSON.parse(evt.data);
        showRecommendation(recommendation);        // render markdown + Approve/Reject
        break;
      case "done":
        markTurnComplete();
        break;
      case "error":
        showError(JSON.parse(evt.data).detail);
        break;
    }
  }
}

// 3. When the user clicks Approve/Reject on a pending recommendation:
async function confirm(action /* "approved" | "rejected" */) {
  const res = await fetch(`${BASE_URL}/confirm`, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${token}`,
      "Content-Type": "application/json",
      "Accept": "text/event-stream",
    },
    body: JSON.stringify({ session_id: sessionId, action }),
  });
  for await (const evt of parseSSE(res.body)) {
    if (evt.event === "done") markTurnComplete();
    if (evt.event === "error") showError(JSON.parse(evt.data).detail);
  }
}
```

> **SSE note:** the native `EventSource` API only supports GET and can't set
> headers. Since these endpoints are POST with an `Authorization` header, use
> `fetch()` + a streaming parser (e.g. `@microsoft/fetch-event-source`) instead.

---

## 7. State machine summary (for UI affordances)

```
            ┌──────────────────────────── done ────────────────────────────┐
            │                                                               ▼
[idle] ──POST /chat──▶ [streaming nodes] ──▶ awaiting_confirmation? ──no──▶ [idle]
                                                     │ yes
                                                     ▼
                                          [recommendation shown]
                                            │ approve        │ reject
                                            ▼                ▼
                                   POST /confirm        POST /confirm
                                   (action=approved)    (action=rejected)
                                            │                │
                                            ▼                ▼
                                     [streaming nodes] ──▶ done ──▶ [idle]
```

- After `awaiting_confirmation`, the only valid next action for that turn is
  `POST /confirm`. Disable the message input until confirmation resolves (or until
  the user explicitly starts a new request).
- After `done`, the session is idle and ready for the next `/chat` message.

---

## 8. Error handling conventions

- All error responses are JSON: `{ "detail": "<human-readable message>" }`.
- Mid-stream failures arrive as an SSE `error` event (HTTP status is already
  `200` because the stream had started) — always handle `error` inside the
  stream, not just on the HTTP response.
- Common statuses: `401` (auth), `403` (ownership/admin), `404` (missing
  session), `422` (bad request body), `500` (agent error), `503` (DB down).
