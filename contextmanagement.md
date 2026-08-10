# Context Management

How conversation/agent state is managed across the backend, based on `agent_orchestrator.md` and `databaseschema.sql`.

## Agent orchestration

A FastAPI backend runs a LangGraph state machine per chat turn:

```
route_intent → chat_response | fetch_doctrine → retrieve → fetch_source → analyze → critique → refine|recommend → post_stub
```

- **Graph is stateless, state is not**: `build_graph()` is rebuilt fresh on every request, but all actual state (`AgentState`) lives in a Postgres-backed LangGraph checkpointer keyed by `thread_id = session_id`.
- **Nodes are uniform**: each is just `async def run(state, config) -> dict`, returning a partial patch that LangGraph merges in (`messages` appends via `add_messages`, everything else overwrites).
- **Follow-up memory**: only `messages`, `session_id`, and `session_documents` are passed as graph input each turn — anything else in the input dict would overwrite the checkpointed value, since non-`messages` channels have no reducer. `route_intent` preserves the previous turn's `candidate_clips` / `final_recommendation` on follow-ups so `analyze` and `recommend` can resolve references like "dive deeper on clip 2" against what was actually shown.
- **Bounded retry loop**: `retrieve → analyze → critique → refine` cycles with two escape hatches — a max iteration count and a "staleness" check (same top-5 segments retrieved twice) — both force-approve to prevent infinite loops.
- **Human-in-the-loop**: the graph is compiled with `interrupt_before=["post_stub"]`, pausing right before the OpusClip publish call until `/confirm` or a classified "yes" reply resumes it.
- **Dependency injection**: a per-request `RuntimeContext` (never checkpointed) carries tool interfaces (search/doctrine/publish/file) plus LLM/embedder, injected via `config["configurable"]["runtime"]`.
- **B2 usage**: native B2 HTTP API (not S3-compatible) for source video/transcript storage, one bucket per tenant. `post_stub` mints a signed download URL for OpusClip rather than proxying video bytes. Filename lookups go through a **legend** — a per-tenant JSON index of the bucket cached on local disk — so resolving a scattered file is a dict access rather than a multi-page bucket scan, and a "dive deeper" follow-up pulls the source file plus its transcript sidecars concurrently through `fetch_source`.

## Database schema

Two persistence systems work together:

- `chat_sessions` table owns **who owns which thread** (`client_id`, `user_id`, `session_id` as PK) — pure directory/ownership metadata.
- LangGraph's own tables (`checkpoints`, `checkpoint_blobs`, `checkpoint_writes`) own the **actual conversation/state content**, linked by `thread_id == session_id`. These are auto-created by `AsyncPostgresSaver.setup()`, not hand-maintained.
- Everything else (`video_summaries`, `transcript_segments` with pgvector embeddings, `brand_doctrine` rubrics, `asset_catalog`) is tenant-scoped via RLS using `current_client_id()`, which resolves from a JWT claim or a `user_profiles` lookup.
- Hybrid search (`hybrid_search_transcripts` / `hybrid_search_summaries`) combines pgvector cosine similarity with Postgres full-text search via Reciprocal Rank Fusion — this is what the `retrieve` node calls.

## Summary

The split is clean:

- **Conversation/agent state** → LangGraph checkpoints
- **Session ownership/directory** → `chat_sessions`
- **Domain data** (videos, transcripts, doctrine) → tenant-isolated tables with RLS
