# joint-platform-backend

All backend services for the TCP joint platform (VP agent + Co-P) in one repo.
The frontend lives in `joint-platform-frontend`.

New here? Start with the [stack overview](docs/stack-overview.md): how the
frontends, backends, databases, auth, ingestion and hosting fit together. It
covers architecture only; known issues and to-dos live in the team's private
doc.

| Folder | Render service | Was | Stack |
|---|---|---|---|
| `services/vp-agent` | `-vp-collaborative` (Docker) | `TCPsysadmin/-vp-collaborative` | FastAPI + LangGraph, grok-4, Supabase Postgres checkpoints |
| `services/cop-chat` | `TCPBackend` (Docker) | `TCPsysadmin/TCPBackend` | FastAPI RAG over `tcp_db_v2`, grok-4-1-fast-reasoning |
| `services/transcription` | `BackBlazeTranscription` (Docker, 12 GB disk) | `TCPsysadmin/BackBlazeTranscription` | FastAPI + ffmpeg + whisper-1 |
| `knowledge/` | n/a | new | Workspace data in git: user library + LLM wiki ([README](knowledge/README.md)) |
| `tools/kb/` | n/a | new | Index and check the knowledge base |

Render names are the live service names (workspace "My Workspace", Ohio).
The old per-service `render.yaml` files used different names that Render never had.

Each service was imported with `git subtree`, so its full history is preserved
(`git log -- services/vp-agent`). Each still has its own README, `.env.example`,
Dockerfile, and tests; run it from its folder exactly as before.

The services are deployed separately and do not share code yet. Merging VP and
Co-P logic happens incrementally inside this repo.

## Deploying

`render.yaml` describes all three services, each with a `rootDir`, so a push
only redeploys services whose folder changed. To move the existing Render
services over without recreating them (keeps their env vars and disk):

1. In each Render service → Settings → change the repository to
   `TCPsysadmin/joint-platform-backend`, branch `main`.
2. Set **Root Directory** to the folder in the table above.
3. For the Docker services, set Dockerfile path `./Dockerfile` (relative to the
   root directory) and confirm the build log shows the right context.
4. Deploy and hit the health check (`/readyz`, `/health`, `/health`).

Until that is done, the old repos keep deploying. Avoid landing changes in both
places; once a service is cut over, archive its old repo.

## CI

GitHub only reads workflows from the repo root, so each service has its own
workflow in `.github/workflows/`, filtered to its folder.

## Knowledge base

```
python tools/kb/kb.py new-id    # id for a new library file
python tools/kb/kb.py index     # rebuild index.json for every workspace
python tools/kb/kb.py check     # what CI runs
```
