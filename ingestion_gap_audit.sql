-- Read-only ingestion health audit. This file does not create or update rows.
-- Change the slug once if a different workspace needs to be inspected.

with target_client as (
    select client_id, display_name, slug
    from public.clients_registry
    where slug = 'collaborative-process'
      and status = 'active'
),
segment_counts as (
    select ts.client_id, ts.source_video_id, count(*)::integer as segment_count
    from public.transcript_segments ts
    join target_client tc using (client_id)
    group by ts.client_id, ts.source_video_id
),
all_sources as (
    select vs.client_id, vs.source_video_id
    from public.video_summaries vs
    join target_client tc using (client_id)

    union

    select sc.client_id, sc.source_video_id
    from segment_counts sc
)
select
    tc.display_name as workspace,
    src.source_video_id,
    vs.title,
    vs.source_file,
    vs.b2_path,
    coalesce(sc.segment_count, 0) as transcript_segment_count,
    nullif(length(trim(vs.summary_text)), 0) is not null as has_summary,
    case
        when coalesce(sc.segment_count, 0) = 0
             and nullif(length(trim(vs.summary_text)), 0) is null
            then 'missing transcript and summary'
        when coalesce(sc.segment_count, 0) = 0
            then 'missing transcript'
        when nullif(length(trim(vs.summary_text)), 0) is null
            then 'missing summary'
    end as gap,
    vs.updated_at
from all_sources src
join target_client tc using (client_id)
left join public.video_summaries vs
    on vs.client_id = src.client_id
   and vs.source_video_id = src.source_video_id
left join segment_counts sc
    on sc.client_id = src.client_id
   and sc.source_video_id = src.source_video_id
where coalesce(sc.segment_count, 0) = 0
   or nullif(length(trim(vs.summary_text)), 0) is null
order by gap, vs.updated_at, src.source_video_id;

-- Recent failed scheduled-ingestion jobs. These explain processing failures,
-- while the first result set shows the current data that still needs repair.
select
    jr.started_at,
    jr.error,
    jr.payload ->> 'source_video_id' as source_video_id,
    jr.payload ->> 'file_name' as file_name,
    jr.payload ->> 'ingestion_type' as ingestion_type
from public.job_runs jr
join public.clients_registry c using (client_id)
where c.slug = 'collaborative-process'
  and jr.kind = 'ingest'
  and jr.status = 'failed'
order by jr.started_at desc
limit 200;
