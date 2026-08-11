-- Enforce tenant isolation inside search RPCs.
--
-- The orchestrator intentionally uses the service-role key for server-side
-- database access. Service role bypasses RLS, so search functions must filter
-- on the authenticated user's resolved client_id explicitly.

begin;

-- A user can be reassigned to another client. Keep old chat history bound to
-- its original client even though the user_id itself has not changed.
--
-- Keep the legacy overloads during deployment so the currently running
-- backend continues to work until Render has switched to the tenant-aware
-- release. Remove them with tenant_search_isolation_cleanup.sql afterward.

create or replace function public.archive_session(
    p_session_id text,
    p_user_id uuid,
    p_client_id uuid
)
returns boolean
language plpgsql
security definer
set search_path = public
as $$
declare
  v_count integer;
begin
  update public.chat_sessions
  set status     = 'archived',
      updated_at = now()
  where session_id = p_session_id
    and user_id    = p_user_id
    and client_id  = p_client_id
    and status     = 'active';
  get diagnostics v_count = row_count;
  return v_count > 0;
end;
$$;

revoke all on function public.archive_session(text, uuid, uuid)
    from public, anon, authenticated;
grant execute on function public.archive_session(text, uuid, uuid)
    to service_role;

create or replace function public.hybrid_search_transcripts(
    p_client_id        uuid,
    p_query_text       text,
    p_query_embedding  vector(1536),
    p_match_count      integer default 40,
    p_rrf_k            integer default 60,
    p_min_seconds      numeric default 0
)
returns table (
    segment_id      uuid,
    source_video_id text,
    source_title    text,
    source_file     text,
    has_timestamps  boolean,
    chunk_index     integer,
    start_seconds   numeric,
    end_seconds     numeric,
    transcript_text text,
    speaker         text,
    fused_score     numeric
)
language sql
stable
security invoker
as $$
with semantic as (
    select ts.segment_id,
           row_number() over (order by ts.embedding <=> p_query_embedding) as rnk
    from public.transcript_segments ts
    where ts.client_id = p_client_id
      and ts.embedding is not null
      and (ts.end_seconds - ts.start_seconds) >= p_min_seconds
    order by ts.embedding <=> p_query_embedding
    limit p_match_count * 2
),
lexical as (
    select ts.segment_id,
           row_number() over (
             order by ts_rank_cd(
               to_tsvector('english', ts.transcript_text),
               websearch_to_tsquery('english', p_query_text)
             ) desc
           ) as rnk
    from public.transcript_segments ts
    where ts.client_id = p_client_id
      and to_tsvector('english', ts.transcript_text)
          @@ websearch_to_tsquery('english', p_query_text)
    limit p_match_count * 2
),
fused as (
    select coalesce(s.segment_id, l.segment_id) as segment_id,
           coalesce(1.0/(p_rrf_k + s.rnk), 0)
         + coalesce(1.0/(p_rrf_k + l.rnk), 0) as score
    from semantic s
    full outer join lexical l using (segment_id)
)
select ts.segment_id,
       ts.source_video_id,
       ts.source_title,
       ts.source_file,
       ts.has_timestamps,
       ts.chunk_index,
       ts.start_seconds,
       ts.end_seconds,
       ts.transcript_text,
       ts.speaker,
       f.score
from fused f
join public.transcript_segments ts using (segment_id)
where ts.client_id = p_client_id
order by f.score desc
limit p_match_count;
$$;

revoke all on function public.hybrid_search_transcripts(
    uuid, text, vector, integer, integer, numeric
) from public, anon, authenticated;
grant execute on function public.hybrid_search_transcripts(
    uuid, text, vector, integer, integer, numeric
) to service_role;

create or replace function public.hybrid_search_summaries(
    p_client_id        uuid,
    p_query_text       text,
    p_query_embedding  vector(1536),
    p_match_count      integer default 20,
    p_rrf_k            integer default 60
)
returns table (
    summary_id      uuid,
    source_video_id text,
    title           text,
    source_file     text,
    has_timestamps  boolean,
    summary_text    text,
    fused_score     numeric
)
language sql
stable
security invoker
as $$
with semantic as (
    select vs.summary_id,
           row_number() over (order by vs.summary_embedding <=> p_query_embedding) as rnk
    from public.video_summaries vs
    where vs.client_id = p_client_id
      and vs.summary_embedding is not null
    order by vs.summary_embedding <=> p_query_embedding
    limit p_match_count * 2
),
lexical as (
    select vs.summary_id,
           row_number() over (
             order by ts_rank_cd(
               to_tsvector('english', vs.summary_text),
               websearch_to_tsquery('english', p_query_text)
             ) desc
           ) as rnk
    from public.video_summaries vs
    where vs.client_id = p_client_id
      and to_tsvector('english', vs.summary_text)
          @@ websearch_to_tsquery('english', p_query_text)
    limit p_match_count * 2
),
fused as (
    select coalesce(s.summary_id, l.summary_id) as summary_id,
           coalesce(1.0/(p_rrf_k + s.rnk), 0)
         + coalesce(1.0/(p_rrf_k + l.rnk), 0) as score
    from semantic s
    full outer join lexical l using (summary_id)
)
select vs.summary_id,
       vs.source_video_id,
       vs.title,
       vs.source_file,
       vs.has_timestamps,
       vs.summary_text,
       f.score
from fused f
join public.video_summaries vs using (summary_id)
where vs.client_id = p_client_id
order by f.score desc
limit p_match_count;
$$;

revoke all on function public.hybrid_search_summaries(
    uuid, text, vector, integer, integer
) from public, anon, authenticated;
grant execute on function public.hybrid_search_summaries(
    uuid, text, vector, integer, integer
) to service_role;

create or replace function public.match_assets(
    p_client_id       uuid,
    p_query_embedding vector(1536),
    p_asset_types     text[] default null,
    p_match_count     integer default 8
)
returns table (
    asset_id    uuid,
    asset_type  text,
    name        text,
    description text,
    parameters  jsonb,
    similarity  numeric
)
language sql
stable
security invoker
as $$
  select ac.asset_id, ac.asset_type, ac.name, ac.description, ac.parameters,
         (1 - (ac.embedding <=> p_query_embedding))::numeric as similarity
  from public.asset_catalog ac
  where ac.client_id = p_client_id
    and ac.is_active = true
    and ac.embedding is not null
    and (p_asset_types is null or ac.asset_type = any(p_asset_types))
  order by ac.embedding <=> p_query_embedding
  limit p_match_count;
$$;

revoke all on function public.match_assets(
    uuid, vector, text[], integer
) from public, anon, authenticated;
grant execute on function public.match_assets(
    uuid, vector, text[], integer
) to service_role;

commit;
