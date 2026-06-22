# Frontend Interface Spec

Implementation-ready design for the **video-agent** frontend. This is the
companion to [backend_integration.md](backend_integration.md) — that document is
the API contract; this one describes the UI that consumes it. It is detailed
enough to hand to a coding agent to build.

**Target stack:** React + Next.js (App Router) + TypeScript + Tailwind CSS.
**Product shape:** chat-first. The conversation is the centerpiece; clip
recommendations and the approve/reject gate render **inline** in the chat.

> Read [backend_integration.md](backend_integration.md) first — every API detail
> (endpoints, SSE events, auth) lives there and is not repeated in full here.

---

## 1. Product principles

1. **The chat is the product.** A creator types what they want ("find me a punchy
   hook about consistency"); the agent works, then proposes a clip. Everything
   happens in one conversational thread.
2. **Show the work.** While the agent searches/analyzes/critiques, surface
   lightweight progress so the wait feels intentional, not frozen.
3. **The approval gate is a first-class moment.** When a recommendation arrives,
   it's a rich inline card with clear **Approve** / **Reject** actions — not a
   buried link. Until the user decides, the composer is locked for that turn.
4. **Resumable & multi-session.** A sidebar lists past conversations; reopening one
   restores its full history.
5. **Calm, creator-tool aesthetic.** Dark-mode-friendly, focused, minimal chrome.

---

## 2. Tech stack & libraries

| Concern | Choice | Why |
|---------|--------|-----|
| Framework | **Next.js (App Router) + TypeScript** | File-based routing, server components for the shell, client components for interactivity |
| Styling | **Tailwind CSS** | Utility-first, fast iteration |
| Components | **shadcn/ui** + **lucide-react** | Accessible primitives (Button, Dialog, ScrollArea, Skeleton), icons |
| Auth | **@supabase/supabase-js** | Email/password sign-in + automatic token refresh; the backend trusts Supabase JWTs |
| **SSE consumption** | **@microsoft/fetch-event-source** | **Critical:** `/chat` and `/confirm` are POST endpoints requiring an `Authorization` header. The native `EventSource` API supports only GET and can't set headers, so it cannot be used. This library does POST + headers + streaming. |
| Markdown | **react-markdown** + **remark-gfm** | The recommendation and assistant messages arrive as Markdown strings |
| State | **React Context + hooks** | The app is small; no Redux/Zustand needed. One auth context + one per-conversation hook |

Install sketch:
```bash
npx create-next-app@latest video-agent-web --ts --tailwind --app
cd video-agent-web
npm i @supabase/supabase-js @microsoft/fetch-event-source react-markdown remark-gfm lucide-react
npx shadcn@latest init
npx shadcn@latest add button textarea scroll-area skeleton dialog avatar sonner
```

**Environment variables** (`.env.local`):
```
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000
NEXT_PUBLIC_SUPABASE_URL=<supabase project url>
NEXT_PUBLIC_SUPABASE_ANON_KEY=<supabase anon key>
```

---

## 3. Information architecture & routes

App Router layout:

```
app/
  layout.tsx                 # <html>, fonts, <AuthProvider>, <Toaster>
  (auth)/
    login/page.tsx           # /login — email/password sign-in
  (app)/
    layout.tsx               # protected shell: redirects to /login if no session
    page.tsx                 # / — empty state ("Start a new conversation")
    chat/[sessionId]/page.tsx# /chat/:sessionId — a conversation
middleware.ts                # optional: redirect unauthenticated users to /login
lib/
  supabase.ts                # browser Supabase client (singleton)
  api.ts                     # typed apiClient (REST)
  sse.ts                     # streamTurn() helper over fetch-event-source
  types.ts                   # Session, ChatMessage, NodeEvent, Recommendation, TurnState
components/
  AppShell.tsx
  SessionSidebar.tsx
  SessionListItem.tsx
  Conversation.tsx
  MessageList.tsx
  MessageBubble.tsx
  AgentProgress.tsx
  RecommendationCard.tsx
  Composer.tsx
  AuthProvider.tsx
  LoginForm.tsx
hooks/
  useAuth.ts
  useSessions.ts
  useConversation.ts         # owns the turn state machine + SSE
```

**Route guard:** the `(app)/layout.tsx` checks for a Supabase session
client-side; if absent, redirect to `/login`. (A `middleware.ts` can also
short-circuit, but Supabase tokens live in client storage by default, so the
client-side guard in the layout is the reliable one.)

---

## 4. Screens & wireframes

### 4.1 Login (`/login`)

Minimal centered card. No sign-up link (users are admin-provisioned via
`POST /admin/users`).

```
┌──────────────────────────────────────────────┐
│                                                │
│                 ◆ video-agent                  │
│           Sign in to your workspace            │
│                                                │
│   ┌────────────────────────────────────────┐  │
│   │ Email                                    │  │
│   │ [ you@example.com                     ]  │  │
│   │ Password                                 │  │
│   │ [ ••••••••••                          ]  │  │
│   │                                          │  │
│   │ [          Sign in           ]           │  │
│   │ ⚠ Invalid email or password (on error)   │  │
│   └────────────────────────────────────────┘  │
│                                                │
└──────────────────────────────────────────────┘
```

On submit → `supabase.auth.signInWithPassword(...)`. On success → redirect to `/`.

### 4.2 Main app shell (`/` and `/chat/:sessionId`)

```
┌───────────────────┬──────────────────────────────────────────────┐
│  ◆ video-agent     │  Find me a hook clip about consistency        │
│  [ + New chat ]    │ ──────────────────────────────────────────── │
│                    │                                                │
│  CONVERSATIONS     │   ┌────────────────────────────────────────┐ │
│  ▸ Consistency hook│   │ 🧑 Find me a punchy hook about           │ │
│    2 msgs · 3m     │   │    consistency                           │ │
│  ▸ Morning routine │   └────────────────────────────────────────┘ │
│    6 msgs · 1h     │                                                │
│  ▸ Q&A clip ideas  │   ┌────────────────────────────────────────┐ │
│    4 msgs · 1d     │   │ 🤖 Searching transcripts…  ⟳            │ │ ← AgentProgress
│                    │   └────────────────────────────────────────┘ │
│                    │                                                │
│                    │   ┌────────────────────────────────────────┐ │
│                    │   │ 🤖 RECOMMENDATION                        │ │ ← RecommendationCard
│                    │   │ <rendered markdown: hook quote, time-    │ │
│                    │   │  range, why it fits the brand…>          │ │
│                    │   │                                          │ │
│                    │   │   [ ✓ Approve & post ]  [ ✗ Reject ]      │ │
│                    │   └────────────────────────────────────────┘ │
│  ───────────────   │ ──────────────────────────────────────────── │
│  🧑 user@email      │  [ Type a message…                    ] [↑]  │ ← Composer (locked
│  [ Sign out ]      │                                                │   while awaiting)
└───────────────────┴──────────────────────────────────────────────┘
```

- **Left sidebar:** brand, "New chat", scrollable session list (newest first),
  current user + sign out at the bottom.
- **Center:** the active conversation (message list + composer). Empty `/` shows a
  centered prompt to start typing or pick a conversation.

### 4.3 Conversation states (center panel)

| State | What renders |
|-------|--------------|
| **Empty** | "Ask the agent to find you a clip." + example prompt chips |
| **Streaming** | User bubble + an `AgentProgress` row showing a friendly label derived from the latest `node` event (see §6) with a spinner |
| **Awaiting confirmation** | The `RecommendationCard` (rendered Markdown + Approve/Reject). Composer disabled with hint "Approve or reject the recommendation to continue." |
| **Posted** | Success bubble (e.g. "Clip posted ✓") and composer re-enabled |
| **Rejected** | Acknowledgement bubble and composer re-enabled |
| **Error** | Inline error bubble with a Retry affordance; composer re-enabled |

### 4.4 Recommendation card (inline)

```
┌──────────────────────────────────────────────────┐
│ 🤖  Recommended clip                               │
│ ────────────────────────────────────────────────  │
│  <react-markdown render of the recommendation>     │
│   • Hook: "…"                                       │
│   • Suggested range / b-roll / why it fits brand    │
│ ────────────────────────────────────────────────  │
│         [ ✓ Approve & post ]   [ ✗ Reject ]         │
└──────────────────────────────────────────────────┘
```

Clicking either button opens the `/confirm` SSE stream (see §7), disables both
buttons, and shows an inline spinner until the stream ends with `done`.

---

## 5. Component tree

| Component | Type | Responsibility | Key props |
|-----------|------|----------------|-----------|
| `AuthProvider` | client | Holds Supabase session, exposes `user`, `accessToken`, `signIn`, `signOut`; refreshes token | `children` |
| `LoginForm` | client | Email/password form → `supabase.auth.signInWithPassword` | — |
| `AppShell` | client | Two-pane layout (sidebar + main) | `children` |
| `SessionSidebar` | client | Loads `GET /sessions`, renders list, "New chat", user footer | — |
| `SessionListItem` | client | One session row; active highlight; links to `/chat/:id` | `session`, `active` |
| `Conversation` | client | Orchestrates a session via `useConversation`; renders list + composer | `sessionId` |
| `MessageList` | client | Scrollable message stack; auto-scroll to bottom | `messages`, `turnState` |
| `MessageBubble` | client | One user/assistant message; assistant content via `react-markdown` | `role`, `content` |
| `AgentProgress` | client | Spinner + friendly label from latest node event | `nodeName` |
| `RecommendationCard` | client | Renders Markdown recommendation + Approve/Reject | `markdown`, `onApprove`, `onReject`, `busy` |
| `Composer` | client | Textarea + send; disabled per turn state; Enter to send, Shift+Enter newline | `disabled`, `onSend` |

---

## 6. State management & the turn state machine

A conversation owns one **turn state machine**, implemented in
`useConversation(sessionId)`:

```
        send message
 idle ───────────────▶ streaming
   ▲                      │
   │                      ├── node events ──▶ (update progress label, stay streaming)
   │                      │
   │      done /          ├── "awaiting_confirmation" ──▶ awaiting_confirmation
   │      rejected ◀──────┤                                      │
   │      / error         │                              approve / reject
   │                      │                                      ▼
   └──────────────────────┴──────── done ◀──────────────── confirming
```

States: `"idle" | "streaming" | "awaiting_confirmation" | "confirming" | "error"`.

**Hook shape:**
```ts
type TurnState =
  | { kind: "idle" }
  | { kind: "streaming"; nodeName?: string }
  | { kind: "awaiting_confirmation"; recommendation: string }
  | { kind: "confirming"; action: "approved" | "rejected" }
  | { kind: "error"; detail: string };

function useConversation(sessionId: string) {
  // messages: ChatMessage[]  (loaded from GET /sessions/:id/messages on mount)
  // turn: TurnState
  // sendMessage(text): starts /chat SSE
  // confirm(action): starts /confirm SSE
  return { messages, turn, sendMessage, confirm, reload };
}
```

**Composer lock rule:** the composer is disabled whenever
`turn.kind !== "idle" && turn.kind !== "error"`. After `awaiting_confirmation`,
the *only* valid action is approve/reject.

**Friendly progress labels** (map `node` name → user-facing copy):
| node | label |
|------|-------|
| `route_intent` | "Understanding your request…" |
| `fetch_doctrine` | "Loading brand guidelines…" |
| `retrieve` | "Searching transcripts…" |
| `fetch_source` | "Reading the source video…" |
| `analyze` | "Analyzing the best moment…" |
| `critique` | "Checking it against your brand…" |
| `refine` | "Refining the search…" |
| `recommend` | "Drafting a recommendation…" |
| `post_stub` | "Posting the clip…" |
| `chat_response` | "Thinking…" |
| _unknown_ | "Working…" |

---

## 7. API wiring

### 7.1 REST client (`lib/api.ts`)

A thin typed wrapper around the backend. Every call injects the Bearer token from
the auth context. **No `client_id` header** — the backend derives the tenant from
the token.

| Method | Backend | Wrapper |
|--------|---------|---------|
| `GET /sessions` | list sessions | `listSessions(): Promise<Session[]>` |
| `POST /sessions` | pre-create | `createSession(title?): Promise<Session>` |
| `GET /sessions/:id/messages` | history | `getMessages(id): Promise<ChatMessage[]>` |

On any `401`, attempt a Supabase token refresh once; if that fails, sign out and
redirect to `/login`.

### 7.2 SSE helper (`lib/sse.ts`)

One function handles both streaming endpoints, since they share the event
vocabulary:

```ts
import { fetchEventSource } from "@microsoft/fetch-event-source";

type StreamHandlers = {
  onNode?: (nodeName: string) => void;
  onAwaitingConfirmation?: (recommendation: string) => void;
  onDone?: () => void;
  onError?: (detail: string) => void;
};

export async function streamTurn(
  path: "/chat" | "/confirm",
  body: object,
  token: string,
  handlers: StreamHandlers,
  signal?: AbortSignal,
) {
  await fetchEventSource(`${BASE_URL}${path}`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
    signal,
    openWhenHidden: true, // keep streaming if the tab is backgrounded
    onmessage(ev) {
      const data = ev.data ? JSON.parse(ev.data) : {};
      switch (ev.event) {
        case "node": handlers.onNode?.(data.node); break;
        case "awaiting_confirmation":
          handlers.onAwaitingConfirmation?.(data.recommendation); break;
        case "done": handlers.onDone?.(); break;
        case "error": handlers.onError?.(data.detail); break;
      }
    },
    onerror(err) { handlers.onError?.(String(err)); throw err; /* stop retry */ },
  });
}
```

- `sendMessage(text)` → `streamTurn("/chat", { session_id, message }, ...)`.
  Optimistically append the user's message bubble before opening the stream.
- `confirm(action)` → `streamTurn("/confirm", { session_id, action }, ...)`.
- After a turn ends (`done` or `awaiting_confirmation`), **re-fetch**
  `GET /sessions/:id/messages` to pull the authoritative assistant message text
  (the SSE `node` events deliberately omit full message bodies).
- On page return/reload, `GET /sessions/:id/messages` includes `turn_status`.
  Keep the progress UI visible while it is `"running"` and refresh messages until
  it becomes `"idle"` or `"completed"`.

### 7.3 Message sourcing rule

- **Live turn feedback** → SSE events (`node` labels, the `awaiting_confirmation`
  recommendation markdown).
- **Durable message history** → `GET /sessions/:id/messages` (canonical; used on
  page load and after each turn completes).

---

## 8. Auth flow

1. `AuthProvider` initializes the Supabase browser client and subscribes to
   `supabase.auth.onAuthStateChange`.
2. `LoginForm` calls `signInWithPassword`. On success, the provider holds the
   session; the `(app)` layout renders.
3. `accessToken = session.access_token` is attached to every API/SSE call.
4. Supabase auto-refreshes the token; the provider always reads the latest.
5. On 401 from the backend → force a refresh; if still failing → `signOut()` +
   redirect to `/login`.
6. **No sign-up UI.** Accounts are created by an operator via `POST /admin/users`.

---

## 9. Design system

Tailwind tokens (extend `tailwind.config.ts`), dark-first:

| Token | Value (suggested) |
|-------|-------------------|
| Background | `zinc-950` (app), `zinc-900` (sidebar/cards) |
| Surface border | `zinc-800` |
| Primary accent | `indigo-500` (buttons, active session) |
| User bubble | `indigo-600` text on `indigo-950/40` |
| Assistant bubble | `zinc-100` text on `zinc-800/60` |
| Success (posted) | `emerald-500` |
| Danger (reject/error) | `rose-500` |
| Font | Inter (UI), system mono for timestamps |
| Radius | `rounded-2xl` cards, `rounded-xl` bubbles |

- **Loading:** `Skeleton` rows for the session list; `AgentProgress` uses a small
  spinner + animated label.
- **Markdown:** style `react-markdown` output with a `prose prose-invert`
  Tailwind-typography class so headings/lists/bold from the recommendation render
  cleanly.
- **Responsive:** below `md`, collapse the sidebar into a slide-over drawer
  (hamburger in a top bar); the conversation goes full-width.

---

## 10. Edge cases & error handling

| Case | Handling |
|------|----------|
| Mid-stream `error` event | Render an error bubble with the `detail`; set turn → `error`; re-enable composer; offer **Retry** (re-send last message). |
| Dropped SSE connection | The backend keeps the turn running. Re-fetch messages; if `turn_status` is `"running"`, keep a progress state visible and refresh until the assistant message appears. |
| Token expires mid-stream | A 401 surfaces via `onerror`; refresh token and re-issue the turn, or bounce to `/login`. |
| Reload during `awaiting_confirmation` | On load, `getMessages` restores history. The recommendation lives in the assistant message; show the card with Approve/Reject still actionable (the backend's run is still paused server-side). |
| Empty session list | Sidebar shows "No conversations yet"; center shows the empty-state prompt. |
| Duplicate session id on `POST /sessions` | Avoid by generating `crypto.randomUUID()` client-side and letting `/chat` auto-create; only call `POST /sessions` once per new chat. |
| Long agent run | Keep `AgentProgress` visible; `openWhenHidden: true` keeps the stream alive in background tabs. |
| Approve/Reject double-click | Disable both buttons once `confirming`. |

---

## 11. Suggested build order

1. **Auth shell** — Supabase client, `AuthProvider`, `/login`, route guard.
2. **App shell + sessions** — `AppShell`, `SessionSidebar`, `GET /sessions`,
   "New chat", routing to `/chat/:id`.
3. **Conversation read path** — `useConversation` loads `GET /sessions/:id/messages`;
   `MessageList` + `MessageBubble` with Markdown.
4. **Send + stream** — `Composer`, `streamTurn("/chat")`, `AgentProgress`, turn
   state machine; re-fetch messages on completion.
5. **Approval gate** — `RecommendationCard`, `streamTurn("/confirm")`, composer
   lock, posted/rejected states.
6. **Polish** — design tokens, responsive drawer, skeletons, toasts, error/retry,
   empty states.

---

## 12. TypeScript types (starter)

```ts
export interface Session {
  session_id: string;
  title: string;
  message_count: number;
  last_message_at: string;
  status: string;
  created_at: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string; // markdown
}

export type SSEEvent =
  | { event: "node"; data: { node: string; partial_state?: unknown } }
  | { event: "awaiting_confirmation"; data: { recommendation: string } }
  | { event: "done"; data: Record<string, never> }
  | { event: "error"; data: { detail: string } };
```

---

> **Consistency note:** every endpoint, header, and SSE event above is taken from
> [backend_integration.md](backend_integration.md) and the live code in
> [orchestrator/main.py](orchestrator/main.py). Two constraints are easy to get
> wrong and worth re-reading: (1) **no `client_id` header** — auth is the Bearer
> token alone; (2) **POST + SSE** means native `EventSource` cannot be used.
