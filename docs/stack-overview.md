# TCP Stack Overview

Snapshot of the live doc as of 2026-10-01: https://claude.ai/code/artifact/1a09cd2f-ed66-44a9-a703-bf8a8c876063 (diagram is there).

TCP runs two AI agents, VP (video clip agent) and Co-P (The Collaborative Pilot). Each has its own backend, its own Supabase project and its own sign-in. Only the frontend is shared: `joint-platform-frontend` puts both behind one sidebar switcher, and each half calls its own backend directly. The three backends now share one repo, joint-platform-backend, but still deploy and run separately; merging their logic has not started.

|  | VP | Co-P |
| --- | --- | --- |
| Backend repo | `-vp-collaborative` (FastAPI + LangGraph) | `TCPBackend` (FastAPI) |
| Backend host | Render service `-vp-collaborative` (Docker) | Render service `TCPBackend` (Docker) |
| Chat model | Grok 4 via xAI | Grok 4.1 fast reasoning via xAI |
| Embeddings | OpenAI `text-embedding-3-small` | OpenAI `text-embedding-3-small` |
| Database | Supabase project "VP db" | Supabase project "Co-p DB" (Nano tier) |
| Sign-in | Supabase Auth (email + password, or Google) | Google sign-in, then GoHighLevel entitlement check |
| Content in | Upload or Drive, transcribed by `BackBlazeTranscription`, indexed by n8n | Unknown loader, not in any repo; being rebuilt in code |
| Old frontend | `vp_frontend` | `v1-collaborative-pilot-ui` |


The frontend is the only shared piece: VP calls go down the left lane and Co-P calls down the right, and Co-P sign-in goes straight to the Co-p DB edge functions.

## Frontends

`joint-platform-frontend` is the one we are building toward; the other three are the old single-purpose UIs it replaces.

| Repo | Stack | What it is | Hosted |
| --- | --- | --- | --- |
| `joint-platform-frontend` | Next.js 15, React 19, Tailwind | Combined UI. Sidebar: agent switcher (VP / Co-P), workspace picker, New chat, Chat / Knowledge / Ingest tabs, conversation list, account. VP lives under `/vp/...`, Co-P under `/cop/...` | Vercel project joint-platform-frontend at joint-platform-frontend.vercel.app. Env vars are set; last deployed 2026-09-26 from a CLI upload, not a git push. CORS is not set up yet |
| `vp_frontend` | Next.js | Old VP UI: chat, knowledge base (media library), workspaces, ingest | Vercel project vp-frontend at vp-frontend-smoky.vercel.app; last deployed 2026-09-19 |
| `v1-collaborative-pilot-ui` | Next.js 15, React 19, generated with v0.app | Old Co-P UI: Google sign-in (or the legacy ?contactId= link), chat with a sessions sidebar. Calls tcpbackend.onrender.com | Vercel project v1-collaborative-pilot-ui at www.co-p.ai (co-p.ai redirects there); last deployed 2026-01-29 |
| `ingestionhub-frontend` | Vite + React | Older standalone VP ingest tool: drop files or links, transcribe, send to Drive via n8n. Its flow now lives in the Ingest tab | Vercel project ingestionhub-frontend at ingestionhub-frontend.vercel.app; still deployed, last on 2026-08-03 |
| `TCPFrontEnd` | Next.js | Earliest Co-P UI, replaced by v1-collaborative-pilot-ui. No env vars set | Vercel project collaborativepilotorigin at collaborativepilot.vercel.app; last deployed 2025-10-09 |

All five Vercel projects sit in the `thecollaborativeprocess` scope of the `tcpsysadmin` Vercel account (Hobby plan). Every `*.vercel.app` address is behind Vercel's login wall, so only www.co-p.ai is open to the public. The joint frontend needs a custom domain, or that protection turned off, before outside users can reach it.

How the joint frontend is organised:

- `components/shell` and `components/chat` are shared by both agents.
- `vp/` and `cop/` folders under `app`, `components`, `lib` and `hooks` hold per-agent code.
- `lib/agents.ts` is the agent registry: names, paths and capability flags. Co-P has workspaces, knowledge browser and ingestion switched off, so those tabs show "not available yet".
- `app/api/cop/auth/route.ts` is the only server code: it turns a Google sign-in into a Co-P token (see User auth).
- Env vars: `NEXT_PUBLIC_VP_*` for VP, `NEXT_PUBLIC_COP_API_BASE_URL` and `NEXT_PUBLIC_GOOGLE_CLIENT_ID` for Co-P, and the server-only `COP_SUPABASE_FUNCTIONS_URL` and `COP_EDGE_APP_SECRET`.
- It has never been built: there is no `package-lock.json` yet.

## Backends

Three Python services, all FastAPI on Render. VP's agent is a multi-step LangGraph pipeline; Co-P's is a single retrieve-then-answer RAG call.

### VP: `-vp-collaborative` (Render: `-vp-collaborative`)

- **What it does:** multi-tenant clip recommender. A user chats; it searches that workspace's transcripts and brand doctrine, drafts a clip, critiques it against the brand rubric, refines until it passes, then waits for the user to approve before posting through OpusClip.
- **Graph** (`orchestrator/graph.py`): `route_intent` sends a turn to chitchat (`chat_response`), `expand_clip`, reuse of an earlier clip (`fetch_source`) or a new search (`fetch_doctrine` → `retrieve`). Search then runs `fetch_source` → `analyze` → `critique`, looping through `refine` → `retrieve` until approved, then `recommend` → `post_stub`. The graph pauses before `post_stub` for `/confirm`.
- **State:** LangGraph checkpoints in Postgres (`AsyncPostgresSaver` on `SUPABASE_DB_URL`).
- **Models:** chat `grok-4` via xAI; embeddings OpenAI `text-embedding-3-small`. OpusClip runs as a stub when `OPUSCLIP_API_KEY` is unset.
- **Endpoints:** `/chat` (SSE) and `/confirm`; `/sessions` and session documents; `/media` (list, detail, signed video URL, bulk move and delete, storage); `/ingestion/b2/media` and `/ingestion/b2/text`; `/workspaces` (create, members, invites, join); `/admin/clients` and `/admin/users`; `/health` and `/readyz`.
- **CORS:** env var `CORS_ORIGINS`.

### Co-P: `TCPBackend` (Render: `TCPBackend`)

- **What it does:** one chat agent. Each turn embeds the question, pulls the top 3 chunks from `tcp_db_v2`, and streams an answer from Grok with the recent chat history.
- **Models:** `grok-4-1-fast-reasoning` via xAI (`main.py` passes `grok-4-fast-reasoning`, but `rag_service.py` overrides it); `gpt-4o-mini` for session summaries; OpenAI `text-embedding-3-small`.
- **Chat history:** stored encrypted with AES-256-GCM (`ENCRYPTION_KEY`) in `encrypted_sessions` and `encrypted_messages`; session titles are plaintext.
- **Endpoints:** `/chat/stream` (SSE; needs a `tier` in the token), `/session/new`, `/sessions`, `/session/{id}/history`, `/session/{id}/summary`, `DELETE /session/{id}`, `/health`, `/metrics`.
- **Limits:** token-bucket rate limit per IP (`RATE_LIMIT_RPM` 60, burst 10), max 1000 connections, 1 uvicorn worker.
- **CORS:** env var `CORS_ALLOWED_ORIGINS` (currently co-p.ai, www.co-p.ai, the old Vercel URL and localhost).
- Its git branches are `main`, `task/multi-workspace-management` (same commit as `main`) and `new-db` (Oct 2025). Justin's merged-backend branch was never pushed.

### VP transcription: `BackBlazeTranscription` (Render: `BackBlazeTranscription`)

- **What it does:** turns audio or video into a timestamped transcript. It takes a B2 file path, a URL or an uploaded file, extracts audio with ffmpeg, splits it into 10-minute chunks, and sends each to OpenAI `whisper-1`.
- **Endpoints:** `/transcribe`, `/transcribeHTTP`, `/fetchText`, `/transcribe-file`, resumable `/uploads/...` (init, chunk, complete, abort), `/queue`, `/jobs/{id}`, `/health`.
- **Auth:** one shared `X-API-KEY` header.
- **Host:** Render Docker service in Ohio with a 12 GB disk at `/data` for temp files.
- Justin's fix for transcripts missing their final chunk was merged to `main` on 2026-09-30, but it is not live: the Render service builds from task/multi-workspace-management with auto-deploy off, and last deployed on 2026-08-27.

## Databases and storage

There are two separate Supabase projects (Postgres + pgvector), one per agent; they share no tables or users. VP keeps its video files in Backblaze B2; both agents use Google Drive as an intake inbox.

### "VP db" (Supabase)

Everything is scoped by `client_id`, which is what the UI calls a workspace.

| Table | Holds |
| --- | --- |
| `clients_registry` | One row per workspace: B2 bucket and prefix, the four Drive folder IDs (transcript / summary × intake / completed), plan tier, status |
| `workspace_memberships`, `workspace_invites`, `user_profiles` | Who belongs to which workspace and with what role |
| `transcript_segments` | Transcript chunks with timestamps, speaker and embedding: the main search index |
| `video_summaries` | One summary per video with its own embedding, B2 path and thumbnail |
| `ingestion_manifests` | One row per ingested video: B2 path, Drive file IDs of its transcript and summary, status |
| `brand_doctrine` | Each workspace's rubric, voice guidelines and forbidden topics, used by the critique step |
| `asset_catalog` | Reusable assets with embeddings, searched by the `match_assets` RPC |
| `chat_sessions`, `blueprints`, `job_runs` | Chat list, drafted clip plans, and a log of runs with tokens and cost |
| `checkpoints*` | LangGraph conversation state |

Schema and migrations are checked into the repo (`databaseschema.sql` plus the `*_migration.sql` files).

### "Co-p DB" (Supabase, Nano tier)

| Table | Holds |
| --- | --- |
| `tcp_db_v2` | Knowledge chunks with embeddings, searched by the `tcpdb_v2_search_ids` RPC |
| `encrypted_sessions` | Chat sessions per GoHighLevel contact; title in plaintext, summary encrypted |
| `encrypted_messages` | Chat messages, AES-256-GCM encrypted by the backend |
| `entitlements`, `login_events`, `payment_events` | Who has paid access, and logs of logins and payments. Seen only in Supabase's advisor output; columns and row counts not yet visible |
| `chat_memory_v2` | Older chat history table from langchain\_setup.sql; probably unused |

Co-p DB runs Postgres 17 in us-east-1 and was created on 2025-08-05. Our Supabase connection can read project settings and Supabase's advisor reports, but not table contents, migrations or edge functions, so this list is incomplete.

The checked-in `services/langchain_setup.sql` is out of date: it defines `chat_memory_v2` and `langchain_chat_history`, not the `encrypted_*` tables or `tcpdb_v2_search_ids` the code calls. The real schema lives only in the Supabase project. This project also hosts the login edge functions (see User auth).

### Backblaze B2

- VP stores original videos and thumbnails in B2, one bucket per workspace (bucket names start with `vpstorage`; the bucket is recorded on `clients_registry`).
- `BackBlazeTranscription` reads media from B2 to transcribe it.
- The backend hands the browser short-lived signed URLs to play a video (`/media/{id}/video-url`).

### Google Drive

- Each VP workspace has transcript and summary intake folders. n8n picks files up from there, indexes them, then moves them to the matching completed folder.
- How Co-P's knowledge got into tcp\_db\_v2 is unknown. The n8n export we thought was Co-P's turned out to be VP's.

## User auth

The two agents have separate user lists, so signing in to one does not sign you in to the other. VP uses Supabase Auth; Co-P checks a paid GoHighLevel (GHL) entitlement and issues its own token.

### VP sign-in

1. The user signs in on `/vp/login` with email + password or Google, through Supabase Auth on "VP db".
2. The browser sends the Supabase access token as `Authorization: Bearer` on every backend call.
3. The backend checks the token by calling Supabase's `/auth/v1/user`.
4. The workspace comes from the `X-Workspace-ID` header and must have a `workspace_memberships` row for that user. Without the header, the backend falls back to the user's default workspace in `user_profiles`.

### Co-P sign-in

1. The user signs in with Google in the browser, which returns a Google ID token.
2. The browser posts it to our server route `/api/cop/auth`, which verifies it against Google's public keys and requires a verified email.
3. The route calls the Co-P edge function `google-email-lookup` with that email. It returns the user's GHL contact ID and whether their access is active.
4. The route calls `ghl-login-handler` with the contact ID. It returns a signed token plus the user's tier and access end date.
5. Both calls carry an app secret header that only exists on the server (`COP_EDGE_APP_SECRET`).
6. The browser keeps the token and sends it as `Authorization: Bearer` to TCPBackend.
7. TCPBackend verifies it as an HS256 JWT with `JWT_SECRET` and `JWT_ISSUER`. The token's `sub` is the GHL contact ID, and `/chat/stream` also requires a `tier`.

So the edge function and TCPBackend share one signing secret. The source of both edge functions is not in any repo we have; it exists only in the Co-p DB Supabase project.

## Ingestion and pipelines

VP content goes in through the Ingest tab, then a scheduled n8n workflow indexes it into VP db. Co-P has no ingestion in any repo; how tcp\_db\_v2 was filled is unknown, so Co-P ingestion is being rebuilt in code (plan below).

### VP: one video, start to finish

1. **Upload.** The Ingest tab uploads the file in resumable chunks to `BackBlazeTranscription`. A B2 link or a ready-made transcript also works.
2. **Transcribe.** The service saves the original and a thumbnail in the workspace's B2 bucket, transcribes with Whisper, and the tab polls `/jobs/{id}` until it finishes.
3. **Summarise.** The tab calls the n8n webhook `/summarize-text`.
4. **Review.** The user edits the transcript and summary before saving.
5. **Write to Drive.** The n8n webhook `/ingest-to-drive` saves both files to each chosen workspace's intake folders and records the B2 path.
6. **Index.** The scheduled n8n workflow "ingestion v3" walks every active workspace in `clients_registry`, up to 10 newest files each run. It embeds with OpenAI and upserts `video_summaries`, `transcript_segments` and `ingestion_manifests`. Then it moves the files to the completed folders and logs failures to `job_runs`.

A video with no audio skips steps 3–5 and is registered as video-only through the VP backend's `/media/storage`.

### n8n workflows we have exports for

All exports are JSON files in `BackBlazeTranscription`:

- `n8n-vp-ingestion-v3-all-workspaces.json`: the indexer in step 6.
- `n8n-vp-drive-to-supabase.json`: the older single-workspace indexer.
- `n8n-flow3-summarize-to-drive.json`: webhooks `/summarize-text`, `/ingest-to-drive` and `/list-clients` (steps 3 and 5).
- `n8n-delete-media-drive-files.json`: trashes Drive files when media is deleted in the Knowledge tab.
- `n8n-repair-ingestion-file.json`: re-runs one failed file.

Workspace storage provisioning (creating the B2 bucket and Drive folders) is a separate n8n workflow, called by the VP backend's `STORAGE_PROVISIONING_WEBHOOK_URL`.

### Plan: keep n8n for VP, rebuild Co-P ingestion in code

VP keeps its n8n workflows because they work. Co-P's ingestion will be rebuilt in code in `joint-platform-backend`, because nothing records how Co-P gets its content today.

1. **Add.** Video and audio go through the transcription service, and the media stays in B2. PDF and DOCX are converted to text; markdown is used as is. Each item becomes one markdown file with a permanent id.
2. **Store.** Files live in git under `knowledge/workspaces/cop/library/`, in whatever folders people choose. An AI-maintained `wiki/` beside it comes later and cites files by id, so reorganizing never breaks it.
3. **Sync.** On each push, a GitHub Action chunks changed files (about 1,000 tokens each), embeds only those, and writes them to new `kb_files` and `kb_chunks` tables in Co-p DB.
4. **Answer.** TCPBackend searches the new tables instead of `tcp_db_v2`. The table and search function become env vars, so switching back is a config change.

`tcp_db_v2` stays untouched until the new path is tested side by side on real questions.

Open for the team: Co-P should support multiple user profiles, each submitting its own data into its own B2 bucket and Drive folders. How profiles map to storage, who can see whose data, and whether client data lives in this repo or a separate private data repo are not decided.

## Deployment hosts and config

The frontend is on Vercel and the three backends are on Render; the data lives in Supabase, Backblaze B2 and Google Drive. Render config is checked in as `render.yaml` in each old backend repo. joint-platform-backend has one render.yaml for all three; each Render service switches over by pointing it at the new repo and its folder (steps in that repo's README). Which branch each Render service deploys from is set in the Render dashboard, not in the repo.

| What | Host | Service / project | Key config (names only) |
| --- | --- | --- | --- |
| `joint-platform-frontend` | Vercel | joint-platform-frontend.vercel.app (behind Vercel login) | `NEXT_PUBLIC_VP_*`, `NEXT_PUBLIC_COP_API_BASE_URL`, `NEXT_PUBLIC_GOOGLE_CLIENT_ID`, `COP_SUPABASE_FUNCTIONS_URL`, `COP_EDGE_APP_SECRET` |
| `v1-collaborative-pilot-ui` (old Co-P) | Vercel | www.co-p.ai | NEXT\_PUBLIC\_GOOGLE\_CLIENT\_ID, NEXT\_PUBLIC\_BACKEND\_URL |
| `-vp-collaborative` | Render (Docker, starter, Ohio) | `-vp-collaborative`, health `/readyz` | `SUPABASE_*`, `XAI_API_KEY`, `OPENAI_API_KEY`, `B2_*`, `TRANSCRIPTION_API_URL/KEY`, provisioning and media-deletion webhooks, `CORS_ORIGINS` |
| `TCPBackend` | Render (Docker, starter, Ohio) | `TCPBackend`, health `/health` | `SUPABASE_URL/KEY`, `OPENAI_API_KEY`, `GROK_API_KEY`, `JWT_SECRET`, `JWT_ISSUER`, `ENCRYPTION_KEY`, `CORS_ALLOWED_ORIGINS` |
| `BackBlazeTranscription` | Render (Docker, starter, Ohio, 12 GB disk) | `BackBlazeTranscription`, health `/health` | `API_KEY`, `OPENAI_API_KEY`, `B2_*`, `N8N_DRIVE_WEBHOOK_URL` |
| `-vp-collaborative` (second copy) | Render (Docker, free, Ohio) | `-vp-collaborative-1`, branch task/multi-workspace-management, last deployed 2026-08-27 | Same repo as -vp-collaborative; nothing in the code points at it |
| VP data and auth | Supabase | "VP db" | Auth redirect URLs must include the frontend's URL |
| Co-P data and login functions | Supabase | "Co-p DB" (Nano) | Edge functions `google-email-lookup`, `ghl-login-handler` |
| Workflows | n8n | n8n Cloud: thecollaborativeprocess.app.n8n.cloud | Webhooks listed under Ingestion |
| Video files | Backblaze B2 | one private bucket per VP workspace |  |
| Third-party APIs | xAI, OpenAI, GoHighLevel, Google OAuth, OpusClip |  |  |

What Render shows (checked 2026-10-01): one workspace, "My Workspace", with four Docker web services in Ohio and no databases, cron jobs or static sites. `-vp-collaborative` and `TCPBackend` deploy `main` on every commit; `TCPBackend` last deployed on 2026-01-29. No service has a health check path set in Render, so the health endpoints above exist in the code but Render does not use them. The API does not return env var names or custom domains, so the config column comes from the code.

To put the joint frontend live, its Vercel URL has to be added in four places:

- [ ] `CORS_ALLOWED_ORIGINS` on `TCPBackend`
- [ ] `CORS_ORIGINS` on `-vp-collaborative`
- [ ] VP db auth redirect URLs
- [ ] Google OAuth authorized JavaScript origins

## Known issues and open to-dos

Security findings are kept in the private live doc (link at the top), not in this public repo.

### Co-P quality (from the TCPBackend audit)

- Retrieval takes the top 3 chunks from `tcp_db_v2` with an empty metadata filter: no filtering by project or file, and no citations.
- The checked-in SQL does not match the tables and RPCs the code calls.
- The system prompt says "It's okay to hallucinate".
- There is no ingestion pipeline in the repo.

### Platform

- [ ] Switch each Render service to `joint-platform-backend` (steps in its README). Justin's merged-backend branch was never pushed, so the new repo was built from the three existing ones.
- [ ] Run `npm install` and `npm run build` on `joint-platform-frontend`, and commit `package-lock.json`.
- [ ] Finish the Vercel setup (checklist under Deployment).
- [ ] Add a read endpoint on TCPBackend so Co-P can have a Knowledge view.
- [ ] Pick the final app name ("TCP Agents" is a placeholder in `lib/agents.ts`).

* [ ] Deploy Justin's transcript fix: point the BackBlazeTranscription service at `main` (or merge into its branch) and turn auto-deploy on.
* [ ] Delete or document the unused-looking copies: Render `-vp-collaborative-1` (free plan) and Vercel `collaborativepilotorigin`.
* [ ] Set health check paths on the Render services (`/readyz` for VP, `/health` for the other two).
* [ ] Give the Supabase connection read access to tables and edge functions, and find the VP db project, so the database sections can be checked.
