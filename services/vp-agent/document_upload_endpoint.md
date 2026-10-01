# Document Upload Endpoint

## Endpoint

`POST /sessions/{session_id}/documents`

Uploads a document into an authenticated user's chat session so the agent can use it as session-scoped memory/context on later `/chat` turns.

## Request

- Auth: `Authorization: Bearer <supabase_access_token>`
- Content type: `multipart/form-data`
- Form field: `file`

Supported uploads:

- `.pdf`
- `.docx`
- UTF-8 text-like files: `.txt`, `.md`, `.markdown`, `.csv`, `.json`, `.jsonl`, `.log`, `.srt`, `.vtt`
- `text/*` content types

## Backend Flow

1. Verifies the Supabase bearer token.
2. Resolves the authenticated user's `client_id`.
3. Verifies or creates the target `chat_sessions` row for that user.
4. Streams the uploaded file in chunks up to `SESSION_DOCUMENT_MAX_UPLOAD_BYTES`.
5. Extracts text:
   - PDF via `pypdf`
   - DOCX via `python-docx`
   - text-like files via UTF-8 decoding
6. Stores the processed result in `chat_session_documents`.

## Stored Fields

The endpoint stores normalized `content_text`, a short `summary`, file metadata, `sha256`, byte/character counts, `session_id`, `user_id`, and `client_id`.

## Agent Usage

Before each `/chat` turn, the backend loads recent ready documents for that session and places them into `AgentState.session_documents`. The chat, analysis, and recommendation nodes format those documents as private session context for the LLM.

Uploaded documents can guide preferences, terminology, campaign details, and constraints. They are not treated as transcript evidence for video quotes or timestamps.
