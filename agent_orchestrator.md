# Agent Orchestrator, State Logic & Backblaze B2 Usage

This document explains how the backend orchestrates the LangGraph-based agent pipeline, how state is
modeled and persisted across turns, and how Backblaze B2 is used as the source-video store. It reflects
the implementation as of the current codebase (`orchestrator/`).

## Overview

The backend is a FastAPI service that runs a **LangGraph** state machine (`langgraph.graph.StateGraph`)
to answer chat requests about video clip recommendations. A single user message is routed through a
sequence of LLM-driven nodes:

```
route_intent → chat_response | fetch_doctrine → retrieve → fetch_source → analyze → critique → refine|recommend → post_stub
```

Source video/transcript files live in **Backblaze B2**, fetched via B2's native HTTP API (not the
S3-compatible endpoint). The final "publish" step hands OpusClip a signed B2 download URL rather than
proxying video bytes through the backend.

Core libraries: `langgraph` (graph engine + Postgres checkpointing), `langchain-xai` (Grok chat model),
`openai` (embeddings only), `fastapi` + `sse-starlette` (HTTP/SSE transport), `supabase` (async client),
`httpx` (raw calls to B2 and OpusClip), `psycopg`/`psycopg_pool` (checkpointer's Postgres pool).

---

## 1. Agent Orchestrator

### Entry point: `build_graph`

`orchestrator/graph.py:build_graph(checkpointer=None) -> CompiledStateGraph` (lines 65–131) is the single
orchestrator entry point. It is called fresh on every `/chat` and `/confirm` request
(`orchestrator/main.py:227, 316, 554`), always passing the shared Postgres-backed checkpointer — so graph
*definition* is stateless and rebuilt per call, while *state* lives entirely in the checkpointer keyed by
`thread_id = session_id`.

### Node registration

Each pipeline step is registered with `builder.add_node(name, fn)` (`graph.py:70-81`), one per module in
`orchestrator/nodes/`. Every node exposes a single async function with a uniform signature:

```python
async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]
```

This `run()` contract is the entire "agent" abstraction — there is no separate agent-registration class
beyond `StateGraph.add_node`.

### Routing / dispatch logic

Execution is **sequential with conditional branching**, not parallel fan-out. Small pure-Python router
functions are wired via `add_conditional_edges` (`graph.py:25-62`):

| Router | Lines | Behavior |
|---|---|---|
| `_intent_router` | 25 | chitchat → `chat_response`; follow-up with reusable context → `fetch_source` (skips retrieval, still reaches `analyze`, and can ingest the referenced clip's source file); else → `fetch_doctrine` (if missing) or `retrieve` |
| `_retrieve_router` | 39 | source resolution failed → `source_not_found`; forced/pre-approved critique exists → `recommend`; else → `fetch_source` |
| `_fetch_source_router` | 49 | source-answer mode → `source_answer`; else → `analyze` |
| `_critique_router` | 55 | `approved` → `recommend`; `needs_refinement` → `refine` (loops back to `retrieve`); `retry` → back to `retrieve` |

The graph is compiled with `interrupt_before=["post_stub"]` (`graph.py:130`) — this is the
**human-in-the-loop confirmation gate**. LangGraph pauses right before the node that actually calls
OpusClip and waits for an explicit `/confirm` call or a natural-language "yes" reply.

### What triggers orchestration

HTTP endpoints in `orchestrator/main.py` (`app = FastAPI(...)`, line 73):

- **`POST /chat`** (line 352) — main entry point. Resolves a `RuntimeContext`, then runs
  `_run_chat_turn` (line 217), which streams the graph via
  `graph.astream(input_state, config=thread_config, stream_mode="updates")`.
- **`POST /confirm`** (line 398) — resumes a graph paused at the `post_stub` interrupt via
  `_run_confirm_turn` (line 307), calling `graph.aupdate_state(...)` then `graph.astream(None, ...)`.
- Session management (`POST/GET /sessions`, `GET /sessions/{id}/messages`, `DELETE /sessions/{id}`) and
  document upload (`POST /sessions/{id}/documents`) support the chat flow but don't run the graph.
- Admin endpoints (`POST /admin/clients`, `POST /admin/users`) provision tenants/users via Supabase RPCs.

### Turn concurrency

`orchestrator/turn_stream.py`'s `TurnStreamRegistry` / `ActiveTurn` (lines 24-178) wraps each graph run
in a detached `asyncio.Task` and fans out Server-Sent Events to any number of subscribers. `get_or_start`
(line 91) raises `TurnAlreadyActiveError` (→ HTTP 409) if a second distinct turn is requested while one is
already running on that session — **one graph execution per session at a time**, though the app serves
many sessions concurrently. The app `lifespan` (lines 54-70) opens the checkpointer and a Supabase
service-role client once at startup, stored on `app.state`.

### Dependency injection ("agent tools")

`orchestrator/runtime.py` defines `RuntimeContext` — a per-request dataclass (never checkpointed) carrying
`client_id`, `user_id`, and four tool interfaces (`search_tool`, `doctrine_tool`, `publish_tool`,
`file_tool`) plus `llm` and `embedder`. It's built per-request in
`orchestrator/auth.py:resolve_runtime` (lines 99-140) and injected via
`config["configurable"]["runtime"]` (`main.py:_thread_config`, lines 165-169); every node reads it at the
top of `run()` (e.g. `retrieve.py:138`, `analyze.py:23`).

Tools are wired in `orchestrator/tools/registry.py:get_tools(supabase, client_id) -> ToolSet`
(lines 22-50): `SupabaseSearchTool`, `SupabaseDoctrineTool`, `OpusClipTool`/`OpusClipStub` (publish),
`B2FileTool` (file). Interfaces are `typing.Protocol`s in `orchestrator/tools/protocols.py`
(`SearchTool`, `DoctrineTool`, `PublishTool`, `Embedder`, `FileTool`, lines 183-284) — this is the
extension point for adding or swapping agent tools.

---

## 2. State Logic

### Schema: `AgentState`

`orchestrator/state.py:AgentState(TypedDict)` (lines 9-41) is the canonical state shape threaded through
every node. Notable fields:

- `messages: Annotated[list[BaseMessage], add_messages]` — LangGraph's built-in append reducer; the full
  conversation history.
- `intent`, `follow_up_reuse`, `user_query` — output of `route_intent`.
- `source_reference`, `source_task`, `source_video_id`, `source_metadata`, `source_resolution_error` —
  source-video resolution state.
- `session_documents` — uploaded document context, re-injected fresh each turn from Postgres.
- `refined_query`, `iteration_count`, `previous_segment_ids`, `retrieved_segments` — the
  retrieval/refinement loop's working memory.
- `brand_doctrine` — client-specific scoring rubric, fetched once per session.
- `source_file_content` — raw transcript/text pulled from B2.
- `candidate_clips`, `candidate_recommendation`, `critique_result`, `final_recommendation` — outputs of
  the analyze → critique → refine loop.
- `awaiting_confirmation`, `confirmation_response` — the human-in-the-loop gate flags.

There is no separate database table mirroring this schema field-by-field; it is serialized wholesale as
checkpoint blobs by LangGraph's checkpointer.

### Transitions

Every node's `run()` returns a **partial dict** — only the keys it changes — which LangGraph merges into
`AgentState` between node executions (`messages` uses the special `add_messages` appender; everything
else is a plain overwrite).

The most consequential transition logic is in `orchestrator/nodes/route_intent.py:run` (lines 109-227):

- **`"new_request"`** — wipes essentially the entire pipeline-scoped state (`refined_query`,
  `iteration_count`, `retrieved_segments`, `candidate_clips`, `critique_result`, `final_recommendation`,
  `awaiting_confirmation`, `confirmation_response`, etc.) so a new topic starts clean.
- **`"follow_up"` without reuse** — resets the retrieval loop (`retrieved_segments`,
  `previous_segment_ids`, `source_file_content`, `critique_result`) but preserves `brand_doctrine`
  **and the previous turn's answer** (`candidate_clips`, `candidate_recommendation`,
  `final_recommendation`).
- **`"follow_up"` with reuse** — preserves `retrieved_segments` / `source_file_content` **and the
  previous answer**; only clears `refined_query`, `iteration_count`, `critique_result`, and the
  confirmation flags.

The previous turn's `candidate_clips` / `final_recommendation` are deliberately *not* cleared on
follow-ups: `analyze` reads them as `previous_clips` / `previous_recommendation` before overwriting
`candidate_clips`, and `recommend` reads `final_recommendation` before overwriting it. This is what
lets "dive deeper on clip 2" resolve positionally against what the creator was actually shown.
`route_intent` also (a) appends an inventory of that state to the classifier prompt
(`orchestrator/context.py:build_prior_context_summary`) and (b) applies a deterministic override —
an explicit clip reference ("clip 2", "the second one", "that option", via
`context.references_prior_clip`) with clips in context and no new source named forces
`follow_up` + `reuse_context`. A stricter predicate — `context.is_clip_refinement_request`, which
requires a clip reference **and** an ask for more/different — short-circuits
`confirm_intent.classify_confirmation_reply` to `"other"`, so "yes, dive deeper on clip 2" is never
read as approval to publish while "yes, make this clip" still is.

`analyze` gates `previous_clips` on `iteration_count == 0` so the refine loop
(`analyze → critique → refine → retrieve → analyze`) doesn't re-feed its own rejected draft back as
"what the user was shown last turn".

### Retrieve ↔ critique ↔ refine loop (bounded retry state machine)

- `orchestrator/nodes/critique.py:run` (lines 18-78) computes a `weighted_score` against the client's
  rubric and derives `verdict ∈ {"approved", "needs_refinement", "retry"}` from
  `brand_doctrine.minimum_a_tier_score` / `auto_reject_below` (lines 58-63), incrementing
  `iteration_count`.
- `orchestrator/nodes/retrieve.py:run` (lines 137-334) has two escape hatches to prevent infinite loops:
  1. If `iteration_count >= settings.max_critique_iterations` and a prior critique exists, force-approve
     with `forced: True` (lines 153-167).
  2. **Staleness check** — if the top-5 retrieved segment IDs are identical to the previous round's and a
     critique already exists, also force-approve (lines 316-328).
- `_critique_router` (`graph.py:55`) reads `critique_result.verdict` to route to `recommend`, `refine`, or
  back to `retrieve`.

### Human-in-the-loop confirmation state

`build_graph(...).compile(interrupt_before=["post_stub"])` (`graph.py:128-131`) pauses execution before
publishing. `orchestrator/main.py:_run_chat_turn` (lines 217-304) checks
`graph.aget_state(thread_config).next` to detect a paused-at-`post_stub` state, then either:

- classifies the user's reply via `orchestrator/confirm_intent.py:classify_confirmation_reply` (an LLM
  call) and resumes via `graph.aupdate_state(...)` + `graph.astream(None, ...)` on approval
  (lines 246-267), or
- clears the gate via `as_node="post_stub"` state update without executing the node, on rejection
  (lines 271-275).

`POST /confirm` (`main.py:398`) performs the same via an explicit `action` field instead of NL
classification.

### Persistence

| What | Where | Notes |
|---|---|---|
| Full `AgentState` (incl. message history) | Postgres, via LangGraph's `AsyncPostgresSaver` (`orchestrator/checkpointer.py:get_checkpointer`, lines 30-48) | Writes to `checkpoints`, `checkpoint_blobs`, `checkpoint_writes` (see `databaseschema.sql:361-393`), auto-created by `checkpointer.setup()` at startup. `thread_id == session_id`. Pool uses `prepare_threshold=None` to work through the Supabase/PgBouncer pooler (`checkpointer.py:36-44`). |
| Session directory / ownership | `public.chat_sessions` table (`databaseschema.sql:329-340`) | `session_id` (PK), `client_id`, `user_id`, `title`, `message_count`, `last_message_at`, `status`. Managed by `orchestrator/session_manager.py`. Per the schema comment (`databaseschema.sql:325-328`): *"chat_sessions... owns WHO owns WHICH thread. LangGraph checkpoint tables own the actual message/state content."* |
| Uploaded session documents | Separate storage (`orchestrator/session_documents.py`) | Re-fetched fresh each `/chat` call (`main.py:370-374`) and injected into `input_state["session_documents"]` — not itself checkpointed, re-supplied every turn. |
| `RuntimeContext`, SSE turn buffers | In-memory only, per-process | `RuntimeContext` rebuilt every request; `TurnStreamRegistry`/`ActiveTurn` buffers pruned after a 300s TTL (`turn_stream.py:86-178`). |

### State flow per turn

1. The FastAPI handler builds `input_state` (`messages`, `session_id`, `session_documents`) and a
   `RunnableConfig` (`{"configurable": {"thread_id": session_id, "runtime": RuntimeContext}}` —
   `_thread_config`, `main.py:165-169`).
2. `graph.astream(input_state, config=thread_config, stream_mode="updates")` runs nodes per the router
   graph; each node reads `state` + `config["configurable"]["runtime"]`, calls out to the LLM/tools, and
   returns a patch.
3. After each node, LangGraph merges the patch into state, persists a new checkpoint row keyed by
   `thread_id`, and yields the patch to the caller, which is sanitized
   (`orchestrator/sse_sanitize.py:compact_partial_state`) and forwarded to SSE subscribers as a `node`
   event (`main.py:282-287`).
4. On completion, `graph.aget_state(thread_config)` is read back to decide the terminal SSE event
   (`awaiting_confirmation` vs `done`), and `session_manager.touch_session` bumps
   `chat_sessions.message_count` / `last_message_at`.

---

## 3. Backblaze B2 Endpoint Usage

This integration uses Backblaze's **native B2 HTTP API** (`b2_authorize_account`, `b2_list_buckets`,
`b2_list_file_names`, `b2_get_download_authorization`, etc.) — **not** the S3-compatible endpoint. There
is no `boto3`/S3 client anywhere in the repo.

### Credentials / configuration

- Env vars `B2_KEY_ID`, `B2_APPLICATION_KEY` (`.env.example:14-15`) load into `Settings` in
  `orchestrator/config.py:26-27` (`b2_key_id: str = ""`, `b2_application_key: str = ""`) — account-level
  application-key credentials applied across every tenant.
- Per-tenant storage location comes from the `clients_registry` table: `b2_bucket text` and
  `b2_prefix text default ''` (`databaseschema.sql:59-60`), set at client-provisioning time via
  `POST /admin/clients` → RPC `admin_provision_client` (`main.py:593-631`;
  `databaseschema.sql:822-855`).
- `Settings.b2_download_url_ttl_seconds: int = 86400` (`config.py:31-34`) — TTL for signed download URLs
  handed to OpusClip (B2 caps this at 604800s / 7 days). `opusclip_api_url` / `opusclip_api_key`
  configure the downstream consumer of those URLs.
- Legend / concurrency settings (`config.py`): `b2_legend_cache_dir` (default `.b2_legend`),
  `b2_legend_ttl_seconds` (900), `b2_legend_min_refresh_seconds` (60), `b2_fetch_concurrency` (6),
  and `source_file_content_max_chars` (60,000). All are passed explicitly into `B2FileTool` by
  `registry.get_tools`, so tests can construct the tool against a temp directory.

### Endpoints

- **Auth**: hardcoded `_B2_AUTH_URL = "https://api.backblazeb2.com/b2api/v3/b2_authorize_account"`
  (`orchestrator/tools/local/b2_file.py:18`).
- **All other API calls** (`b2_list_buckets`, `b2_list_file_names`, `b2_list_file_versions`,
  `b2_list_keys`, `b2_get_download_authorization`) target `{auth['apiUrl']}/b2api/v3/<method>`, where
  `apiUrl` is returned dynamically by the authorize call and cached (`_authorize`, lines 86-104).
- **File downloads** hit `{auth['downloadUrl']}/file/{bucket}/{full_path}`
  (`_download_text_file`, line 459; signed-URL construction, line 575) — `downloadUrl` is likewise
  returned by B2, not hardcoded.

### `B2FileTool` — wrapper class

File: `orchestrator/tools/local/b2_file.py`, class `B2FileTool` (lines 56-584). Instantiated once per
request in `registry.get_tools` (`orchestrator/tools/registry.py:44-49`) with
`(supabase, client_id, key_id, application_key)`. Implements the `FileTool` Protocol
(`orchestrator/tools/protocols.py:226-284`).

| Method | Purpose | Lines |
|---|---|---|
| `_authorize()` | `b2_authorize_account`; caches `authorizationToken`, `downloadUrl`, `apiUrl`, `accountId` for the request's lifetime | 86-104 |
| `_get_client_storage()` | Loads `b2_bucket` / `b2_prefix` for `client_id` from `clients_registry`, caches result | 114-136 |
| `list_buckets()` | `b2_list_buckets`; refreshes the bucket-name→id cache | 142-182 |
| `_resolve_bucket_id()` | name→id cache lookup, falling back to `list_buckets` | 184-192 |
| `list_file_names()` | `b2_list_file_names` (paginated, GET) | 198-227 |
| `list_file_versions()` | `b2_list_file_versions` (paginated, GET) | 233-270 |
| `list_keys()` | `b2_list_keys` | 276-313 |
| `find_file_by_name()` | **Legend lookup first** (O(1) dict access), then one bounded legend refresh + retry, then the original paginated scan as a last resort | — |
| `refresh_legend()` | Rebuilds and persists the tenant's file index; `min_age_seconds` floors how often a miss may trigger it | — |
| `fetch_files()` | Concurrent multi-file download (shared client + semaphore + `return_exceptions`), one `B2FetchedFile` per path | — |
| `fetch_source_bundle()` | Source file + its legend-resolved text sidecars, fetched in one batch — the "dive deeper" ingestion primitive | — |
| `_resolve_source_path()` | Resolves `source_video_id` → `(bucket, full_path)` via `video_summaries.b2_path` / `source_file`, falling back to `find_file_by_name` | 389-449 |
| `_download_text_file()` | GETs file bytes; returns text only for recognized text content-types/extensions, else `None` | 451-485 |
| `fetch_source_file(source_video_id)` | Public entry point: resolve path → download text (transcript fetch) | 487-497 |
| `fetch_file_by_name(file_name, bucket_name=None, prefix=None)` | Resolve by filename → download text | 499-527 |
| `get_download_url(source_video_id, valid_duration_seconds=86400)` | **Presigned URL generator**: `b2_get_download_authorization` scoped to the exact `fileNamePrefix`, then builds `{downloadUrl}/file/{bucket}/{urlencoded_path}?Authorization={token}`. Returns `None` unless the resolved path is **media** — lookup is deliberately permissive so a source whose video was never uploaded still resolves to its transcript, but this URL goes to OpusClip, which cannot cut a clip out of a `.txt` | 534-584 |

Typed result data classes (`orchestrator/tools/protocols.py`): `B2Bucket`, `B2FileEntry`,
`B2FetchedFile`, `B2KeyInfo`, parsed via `_parse_file_entry` (`b2_file.py`).

### The legend — local file index (`orchestrator/tools/local/b2_legend.py`)

Backblaze's `prefix` is a literal-string match with no server-side basename search, so
`find_file_by_name` used to paginate up to 10 × 1000 entries **per lookup** — far too slow for an
interactive turn. The *legend* replaces that with a one-time sweep, cached locally:

- **Build**: one paginated `b2_list_file_names` sweep of the client's configured bucket + prefix at
  `maxFileCount=1000` (B2 bills listing per 1000 returned, so asking for more buys nothing), following
  `nextFileName` to exhaustion. Only `action == "upload"` rows are indexed — folder placeholders, hide
  markers and in-progress large uploads would resolve to dead paths.
- **Shape**: `{"built_at": <iso8601>, "bucket", "prefix", "files": {<normalized basename>: [entry, ...]}}`
  where each entry carries the **full B2 path**, `file_id`, `content_length`, `content_type` and
  `upload_timestamp`. The value is a *list*: the normalization key drops the extension on purpose (so
  `talk.mp4`, `talk.txt` and `talk.srt` collide), and the same basename can live under two folders.
  Collisions resolve at lookup: exact full path → exact basename → most recent upload.
- **Role-suffix pairing**: a second index, derived in memory and never persisted, keys entries by that
  name minus a trailing `transcript` / `summary` token. Transcripts in practice are *not* the video's
  name with another extension — they are `<video>_transcript.txt`, so keying on the extension-stripped
  basename alone files a video apart from its own transcript. Against the live `TCP-MASTER` bucket that
  produced **0 sidecar matches across 236 resolved files**; with the widened key it is 269 of 271. The
  token is end-anchored, so a title that merely contains the word (`… - TRANSCRIPT ONLY.mp4`) keeps it.
  The same key lets a source whose video was never uploaded still resolve to its transcript.
- **Where**: `{b2_legend_cache_dir}/{client_id}.json`, written atomically (temp file + `os.replace`) so
  concurrent sessions for one tenant can't shred it. An unreadable or corrupt file is treated as "no
  legend" and rebuilt, never raised.
- **Refresh**: in-memory (per `B2FileTool`, i.e. per request) → on-disk → rebuild, at each step honouring
  `b2_legend_ttl_seconds`. A lookup **miss** triggers one rebuild-and-retry, rate-limited by
  `b2_legend_min_refresh_seconds` so a genuinely absent file doesn't re-sweep the bucket every turn.
- **Fallback**: `find_file_by_name` still runs the original paginated scan when the legend misses, so
  correctness never depends on the cache being current. The legend is bypassed entirely when a caller
  overrides bucket/prefix away from the client's configured location.

### Concurrent fetches and auth retry

- `fetch_files(paths=[...], bucket_name=...)` downloads several files in one batch: a **single
  `httpx.AsyncClient`** (one connection pool, one B2 auth token — which B2 documents as safe to share
  across concurrent downloads) plus an `asyncio.Semaphore(b2_fetch_concurrency)` and
  `asyncio.gather(..., return_exceptions=True)`. The client is scoped to the *batch*, not the tool
  instance: nothing in the request path disposes tools, and a turn runs in a detached task that outlives
  its HTTP request. Each path returns a `B2FetchedFile` distinguishing three outcomes — text downloaded,
  file exists but isn't text, download failed — so a missing sidecar doesn't read like a broken B2 call.
- `fetch_source_bundle(source_video_id)` resolves the source path, asks the legend for the text sidecars
  belonging to its stem (`.md`/`.srt`/`.text`/`.txt`/`.vtt`), and fetches the lot concurrently. It
  **never requests the media file**: a source resolves to its video far more often than to a transcript,
  a video carries no ingestible text, and content-type is only readable once the whole body has arrived —
  so asking for it downloaded the entire file only to discard it (19.3s and an empty bundle against the
  real bucket; the same call now returns transcript text in ~0.9s). `is_media_file()` rules a path out by
  name, before any request. A source with no text anywhere returns `[]` without downloading anything.
- **Transcript first**: the batch is sorted by `sidecar_sort_rank` so a transcript always precedes a
  summary. Ordering is a property of the filename, not of which file resolved first, because a source
  whose video is missing resolves to whichever sidecar the index yields.
- **Truncation** (`fetch_source._combine_bundle`): parts are joined in order and capped at
  `source_file_content_max_chars` (200,000 — at the previous 60,000, 36% of this tenant's transcripts
  were cut mid-sentence; the largest is 184k, and the cap is env-overridable if prompt cost matters
  more than recall). When the cap does bite, only the *longest* part is shrunk and the shorter ones are
  kept whole: truncating the joined string instead dropped the summary off the end, leaving the model a
  source's opening minutes with no idea what the rest covered.
- Every download goes through `_download_file`, which percent-encodes the path (`quote(..., safe="/")`,
  matching `get_download_url`) and, on a 401 carrying `expired_auth_token` / `bad_auth_token`,
  re-authorizes and retries **that one call** — a late-batch expiry doesn't discard completed work.
  `missing_auth_token` is deliberately not retried: it means a client bug.

### "Dive deeper on clip 2" — source ingestion on the reuse path

`_intent_router`'s `"reuse"` branch now enters at **`fetch_source`** instead of `analyze`
(`graph.py`). Retrieval is still skipped; `_fetch_source_router` still lands on `analyze` (or
`source_answer`), so the topology is otherwise unchanged. What this buys: when the clip the creator
points at belongs to a source whose file has never been ingested, `fetch_source` pulls it now.

- `fetch_source._target_source_video_id` picks the referenced clip's `video_id` — resolving "clip 2" /
  "the second one" via `context.referenced_clip_position` against `candidate_clips` — instead of
  defaulting to the top-ranked segment's video, which is usually a different file.
- `AgentState.source_content_video_id` records which source `source_file_content` was ingested for, so
  the node can tell "already have this transcript" from "have a *different* video's transcript".
  `None` means unknown provenance (legacy checkpoints, or the pseudo-transcript `retrieve` builds for a
  B2-only source) and is treated as a match — it is never refetched.
- Multiple text files are joined under `## <basename>` headings and capped at
  `source_file_content_max_chars` with a truncation marker.

### Buckets and their purpose

One bucket per tenant (`b2_bucket` + `b2_prefix` on `clients_registry`), storing **source video/audio
files and their transcripts** — mp4/mov/m4v/mp3/wav/m4a/aac/flac plus text sidecars
(csv/json/log/md/srt/txt/vtt; extension sets at `b2_file.py:24-53`). There is no separate
"artifacts" bucket: OpusClip generates clips itself, pulling the source video directly from B2 via a
signed URL rather than the orchestrator storing or uploading generated output.

**Consumers:**

1. `orchestrator/nodes/fetch_source.py:run` (lines 15-48) — calls
   `runtime.file_tool.fetch_source_file(source_video_id)` to pull transcript/text content used by
   `analyze` when full-transcript context is needed; falls back to `None` if the file isn't text.
2. `orchestrator/nodes/retrieve.py` — `_find_b2_source_file` / `_resolve_source_video` (lines 85-134)
   use `find_file_by_name` / `fetch_file_by_name` to resolve a user-typed source reference (URL,
   filename, "the video called X") directly against B2 when Supabase's `resolve_source_video` search
   comes up empty — including synthesizing a `source_video_id = f"b2:{file_id_or_name}"` and a full-text
   pseudo-segment when only a B2 file (no indexed transcript row) is found (lines 184-227).
3. `orchestrator/nodes/post_stub.py:run` (lines 18-82) — the confirmation-gated publish step. Calls
   `runtime.file_tool.get_download_url(video_id, valid_duration_seconds=settings.b2_download_url_ttl_seconds)`
   (lines 36-38) to mint a signed URL, attaches it as `metadata["video_url"]` on a `ClipPayload`, and
   passes it to `runtime.publish_tool.create_and_post_clip(payload)` (the OpusClip tool). **B2 is the
   source of truth for video; OpusClip reads it directly, and the orchestrator never proxies video
   bytes.**

### Retry / error handling

No explicit retry/backoff logic (no `tenacity`, no manual retry loop) wraps B2 calls. Each
`httpx.AsyncClient` call uses `resp.raise_for_status()` (e.g. `b2_file.py:94, 161, 222, 260, 293, 466,
570`) with per-call timeouts (15s for auth/list/URL-signing, 30s for file listing, 60s for downloads).
Failures propagate as `httpx.HTTPStatusError` / connection exceptions and are caught at the **node
level**, re-raised as domain errors:

- `fetch_source.py:36-38` → `RetrievalError(f"B2 file download failed for {source_video_id}: {exc}")`
- `retrieve.py:90-93, 131-132` → `RetrievalError` for B2 lookup/download failures during source
  resolution
- `post_stub.py:39-40` → `ToolError(f"Resolving source video URL failed for {video_id}: {exc}")`

These `AgentError` subclasses (`orchestrator/errors.py`) are caught centrally — FastAPI exception
handlers for HTTP paths (`main.py:131-134`) and a `try/except AgentError` inside
`_run_chat_turn`/`_run_confirm_turn` (`main.py:299-304, 341-346`) — turning any failure into a terminal
SSE `error` event rather than crashing the turn. Retry, in effect, means "fail the turn cleanly and let
the user re-ask," not automatic retries.

Missing/unresolvable B2 data degrades gracefully rather than erroring:
- `_get_client_storage` logs a warning and returns `None` if no bucket is configured
  (`b2_file.py:126-133`).
- `find_file_by_name` logs a warning and returns `None` if nothing matches after the page cap
  (lines 377-383).
- `_download_text_file` returns `None` (not an error) for non-text content types (lines 469-477).
