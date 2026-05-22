# video-agent

A multi-tenant video clip recommendation agent built with LangGraph, deployed on Render as a FastAPI service with SSE streaming.

Users (video creators) chat with the agent. It searches a Supabase database of transcript chunks and brand doctrine, drafts a clip recommendation, self-reviews against the client's brand rubric until quality is acceptable, then asks the user to approve before posting via OpusClip.

---

## Architecture

```mermaid
graph TD
    Client -->|POST /chat| FastAPI
    FastAPI -->|SSE stream| Client
    FastAPI --> Graph

    subgraph Graph [LangGraph StateGraph]
        RI[route_intent] --> CR[chat_response]
        RI --> FD[fetch_doctrine]
        RI --> REC[recommend]
        FD --> RTV[retrieve]
        RTV --> AN[analyze]
        AN --> CRQ[critique]
        CRQ -->|approved| REC
        CRQ -->|needs_refinement| RF[refine]
        CRQ -->|retry| RTV
        RF --> RTV
        REC -->|interrupt_before| PS[post_stub]
    end

    Graph --> Supabase[(Supabase)]
    Graph --> OpenAI[OpenAI LLM + Embeddings]
```

### Runtime context

`RuntimeContext` (defined in `orchestrator/runtime.py`) carries the Supabase client, tool implementations, LLM, and embedder. It is injected per-request via `config["configurable"]["runtime"]` and is **never stored in `AgentState`** — so the checkpointed state remains fully JSON-serialisable.

---

## Graph flow

```
START
  → route_intent
       ├── chitchat    → chat_response → END
       ├── follow_up   → recommend ──────────────────────────────┐
       └── new_request → fetch_doctrine → retrieve               │
                                             ↓                    │
                                           analyze                │
                                             ↓                    │
                                          critique                │
                                     ┌─ approved ──────────────→ recommend → [interrupt] → post_stub → END
                                     ├─ needs_refinement → refine → retrieve  (loop)
                                     └─ retry             → retrieve  (loop)
```

The graph pauses (`interrupt_before=["post_stub"]`) after `recommend` so the user can approve or reject via `POST /confirm`.

---

## Local setup

```bash
git clone <repo>
cd video-agent

# Python 3.12 (use pyenv or system)
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements-dev.txt

cp .env.example .env
# Fill in .env with your keys

uvicorn orchestrator.main:app --reload
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `SUPABASE_URL` | ✓ | — | Supabase project URL |
| `SUPABASE_SERVICE_KEY` | ✓ | — | Service-role key (bypasses RLS) |
| `SUPABASE_DB_URL` | ✓ | — | Postgres connection string for checkpointer |
| `OPENAI_API_KEY` | ✓ | — | LLM + embeddings |
| `LLM_MODEL` | | `gpt-4o` | OpenAI chat model ID |
| `EMBEDDING_MODEL` | | `text-embedding-3-small` | OpenAI embedding model |
| `MAX_CRITIQUE_ITERATIONS` | | `3` | Max critique/refine loops before forced approval |
| `LANGCHAIN_TRACING_V2` | | `false` | Enable LangSmith tracing |
| `LANGCHAIN_API_KEY` | | — | LangSmith API key |
| `LOG_LEVEL` | | `INFO` | structlog level |

---

## API

### POST /chat

Start or continue a session.

**Headers:** `X-Client-ID: <uuid>`

**Body:**
```json
{ "session_id": "my-session-123", "message": "Find me a great hook clip about consistency" }
```

**SSE events:**
- `node` — emitted after each graph node completes
- `awaiting_confirmation` — graph paused before posting; includes `recommendation` markdown
- `done` — graph completed without posting (e.g., chitchat)

**curl:**
```bash
curl -N -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -H "X-Client-ID: 00000000-0000-0000-0000-000000000001" \
  -d '{"session_id":"sess-1","message":"Find me a hook clip about consistency"}'
```

---

### POST /confirm

Resume a paused session after user approval/rejection.

**Headers:** `X-Client-ID: <uuid>`

**Body:**
```json
{ "session_id": "my-session-123", "action": "approved" }
```

`action` is `"approved"` or `"rejected"`.

**curl:**
```bash
curl -N -X POST http://localhost:8000/confirm \
  -H "Content-Type: application/json" \
  -H "X-Client-ID: 00000000-0000-0000-0000-000000000001" \
  -d '{"session_id":"sess-1","action":"approved"}'
```

---

### GET /health

Returns `{"status": "ok", "version": "<git sha>"}`.

### GET /readyz

Verifies DB reachability. Used by Render health checks.

---

## Running tests

```bash
pytest tests/ -v
```

CI runs ruff + mypy + pytest on every push via `.github/workflows/ci.yml`.

---

## Deployment (Render)

The service is defined in `render.yaml`. It uses Docker, health-checks `/readyz`, and expects all env vars to be set in the Render dashboard.

> **Note:** Render's free tier sleeps after 15 minutes of inactivity. Cold starts break long-lived SSE connections. Use the **Starter plan** for any real workload.

---

## Troubleshooting

**`X-Client-ID header is required` (400)**
Include the header on every request: `-H "X-Client-ID: <uuid>"`.

**`Database not reachable` (503) on `/readyz`**
Check `SUPABASE_DB_URL`. The checkpointer needs a direct Postgres connection, not the Supabase REST URL.

**Graph never reaches `post_stub`**
The graph pauses with `interrupt_before=["post_stub"]`. You must call `POST /confirm` to resume.

**LangSmith traces not appearing**
Set `LANGCHAIN_TRACING_V2=true` and `LANGCHAIN_API_KEY`. LangChain reads these env vars natively.

**Forced approval note in recommendation**
The critique loop ran `MAX_CRITIQUE_ITERATIONS` times or retrieved the same top-5 segments twice. The system approved the best available match and appended an explanatory note.
