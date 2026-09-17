-- Atomic database operations for bulk Knowledge Base deletion and workspace moves.
-- Run once in Supabase SQL Editor before deploying task/bulk-media-management.

create or replace function public.admin_delete_media(
    p_client_id uuid,
    p_source_video_ids text[]
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    v_deleted integer;
begin
    if coalesce(array_length(p_source_video_ids, 1), 0) = 0 then
        return 0;
    end if;

    delete from public.ingestion_manifests
     where client_id = p_client_id
       and source_video_id = any(p_source_video_ids);

    -- Some legacy transcript rows have no summary_id, so remove by tenant/source
    -- explicitly instead of relying only on the foreign-key cascade.
    delete from public.transcript_segments
     where client_id = p_client_id
       and source_video_id = any(p_source_video_ids);

    delete from public.video_summaries
     where client_id = p_client_id
       and source_video_id = any(p_source_video_ids);
    get diagnostics v_deleted = row_count;

    return v_deleted;
end;
$$;

revoke all on function public.admin_delete_media(uuid, text[])
from public, anon, authenticated;
grant execute on function public.admin_delete_media(uuid, text[])
to service_role;


create or replace function public.admin_move_media(
    p_source_client_id uuid,
    p_destination_client_id uuid,
    p_source_video_ids text[],
    p_destination_b2_bucket text
)
returns integer
language plpgsql
security definer
set search_path = public
as $$
declare
    v_requested integer;
    v_found integer;
begin
    v_requested := coalesce(array_length(p_source_video_ids, 1), 0);
    if v_requested = 0 then
        return 0;
    end if;
    if p_source_client_id = p_destination_client_id then
        raise exception 'Source and destination workspaces must be different';
    end if;

    select count(*) into v_found
      from public.video_summaries
     where client_id = p_source_client_id
       and source_video_id = any(p_source_video_ids);
    if v_found <> v_requested then
        raise exception 'One or more selected videos no longer exist in the source workspace';
    end if;

    if exists (
        select 1
          from public.video_summaries
         where client_id = p_destination_client_id
           and source_video_id = any(p_source_video_ids)
    ) then
        raise exception 'One or more selected videos already exist in the destination workspace';
    end if;

    update public.ingestion_manifests
       set client_id = p_destination_client_id,
           b2_bucket = p_destination_b2_bucket,
           updated_at = now()
     where client_id = p_source_client_id
       and source_video_id = any(p_source_video_ids);

    update public.video_summaries
       set client_id = p_destination_client_id,
           updated_at = now()
     where client_id = p_source_client_id
       and source_video_id = any(p_source_video_ids);

    update public.transcript_segments
       set client_id = p_destination_client_id
     where client_id = p_source_client_id
       and source_video_id = any(p_source_video_ids);

    return v_found;
end;
$$;

revoke all on function public.admin_move_media(uuid, uuid, text[], text)
from public, anon, authenticated;
grant execute on function public.admin_move_media(uuid, uuid, text[], text)
to service_role;
