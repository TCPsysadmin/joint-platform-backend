-- Deduplicate retries of one upload request while allowing a user to attach the
-- exact same file again intentionally under a new request ID.

drop index if exists public.chat_session_documents_session_sha256_uidx;

alter table public.chat_session_documents
    add column if not exists upload_request_id text;

create unique index if not exists chat_session_documents_session_request_uidx
    on public.chat_session_documents (session_id, upload_request_id)
    where upload_request_id is not null;
