-- Prevent identical document bytes from being stored more than once in a chat.
--
-- Before running this migration, the duplicate audit must return zero rows:
--
-- select session_id, sha256, count(*)
-- from public.chat_session_documents
-- group by session_id, sha256
-- having count(*) > 1;

create unique index if not exists chat_session_documents_session_sha256_uidx
    on public.chat_session_documents (session_id, sha256);
