# Session Document Upload Database Setup

The API expects a Supabase table named `public.chat_session_documents`. The backend writes processed text content into this table after `POST /sessions/{session_id}/documents`, then reads the newest ready documents for that session before each `/chat` turn.

## Required Table

```sql
create table if not exists public.chat_session_documents (
    doc_id       uuid primary key default uuid_generate_v4(),
    session_id   text not null references public.chat_sessions(session_id) on delete cascade,
    user_id      uuid not null,
    client_id    uuid not null references public.clients_registry(client_id) on delete cascade,
    filename     text not null,
    content_type text,
    byte_size    integer not null,
    char_count   integer not null,
    sha256       text not null,
    upload_request_id text,
    content_text text not null,
    summary      text,
    metadata     jsonb not null default '{}'::jsonb,
    status       text not null default 'ready',
    created_at   timestamptz not null default now(),
    updated_at   timestamptz not null default now(),
    constraint chat_session_documents_status_check
      check (status in ('processing', 'ready', 'failed'))
);
```

## Indexes

```sql
create index if not exists chat_session_documents_session_idx
    on public.chat_session_documents (session_id, created_at desc);

create index if not exists chat_session_documents_user_idx
    on public.chat_session_documents (user_id, created_at desc);

create index if not exists chat_session_documents_client_idx
    on public.chat_session_documents (client_id);

create unique index if not exists chat_session_documents_session_request_uidx
    on public.chat_session_documents (session_id, upload_request_id)
    where upload_request_id is not null;

create index if not exists chat_session_documents_text_fts_idx
    on public.chat_session_documents
    using gin (to_tsvector('english', coalesce(content_text, '')));
```

## Updated At Trigger

Reuse the existing `public.set_updated_at()` function from `databaseschema.sql`:

```sql
drop trigger if exists trg_chat_session_documents_updated on public.chat_session_documents;
create trigger trg_chat_session_documents_updated
    before update on public.chat_session_documents
    for each row execute function public.set_updated_at();
```

## Row Level Security

The backend uses the service-role key and always filters by `session_id`, `user_id`, and `client_id`. These RLS policies are for direct Supabase access from authenticated clients if you expose it later.

```sql
alter table public.chat_session_documents enable row level security;

drop policy if exists tenant_select on public.chat_session_documents;
create policy tenant_select on public.chat_session_documents
for select using (
  user_id = auth.uid()
);

drop policy if exists tenant_insert on public.chat_session_documents;
create policy tenant_insert on public.chat_session_documents
for insert with check (
  user_id = auth.uid()
);

drop policy if exists tenant_update on public.chat_session_documents;
create policy tenant_update on public.chat_session_documents
for update using (
  user_id = auth.uid()
) with check (
  user_id = auth.uid()
);

drop policy if exists tenant_delete on public.chat_session_documents;
create policy tenant_delete on public.chat_session_documents
for delete using (
  user_id = auth.uid()
);
```

## Grants

```sql
grant select, insert, update, delete on public.chat_session_documents to authenticated;
```

## Backend Contract

The endpoint stores extracted text from `.pdf`, `.docx`, and UTF-8 text-like files: `.txt`, `.md`, `.markdown`, `.csv`, `.json`, `.jsonl`, `.log`, `.srt`, and `.vtt`, plus `text/*` content types. PDF extraction uses `pypdf`; DOCX extraction uses `python-docx`. The backend writes:

- `content_text`: normalized extracted text used as agent context.
- `summary`: a short first-pass summary/excerpt for UI and quick inspection.
- `sha256`: hash of the original uploaded bytes for integrity checks and audits.
- `upload_request_id`: optional frontend-generated idempotency key. Retrying one
  upload reuses the row, while intentionally attaching identical bytes again
  with a new key creates a new row.
- `metadata`: currently includes the file extension, detected document kind, and extraction method.
- `status`: currently written as `ready`.

No Supabase Edge Function is required for the current backend path. If the frontend later uploads directly to Supabase Storage for large files, add an Edge Function or background worker to extract text and insert rows into this same table using the same columns.
