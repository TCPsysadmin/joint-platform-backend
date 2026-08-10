-- Durable Drive/B2 ingestion handoff.
-- Run once in the Supabase SQL editor before importing the revised n8n workflows.

alter table public.clients_registry
    add column if not exists drive_transcripts_intake_folder_id text,
    add column if not exists drive_summaries_intake_folder_id text,
    add column if not exists drive_transcripts_completed_folder_id text,
    add column if not exists drive_summaries_completed_folder_id text;

-- Databases created before the media-library UI do not have thumbnail fields.
-- Keep this additive and idempotent so it is safe to rerun during rollout.
alter table public.video_summaries
    add column if not exists thumbnail_url text,
    add column if not exists thumbnail_b2_path text;

create unique index if not exists clients_registry_transcripts_intake_uidx
    on public.clients_registry (drive_transcripts_intake_folder_id)
    where drive_transcripts_intake_folder_id is not null;

create unique index if not exists clients_registry_summaries_intake_uidx
    on public.clients_registry (drive_summaries_intake_folder_id)
    where drive_summaries_intake_folder_id is not null;

create table if not exists public.ingestion_manifests (
    manifest_id               uuid primary key default uuid_generate_v4(),
    idempotency_key           text unique not null,
    client_id                 uuid not null references public.clients_registry(client_id) on delete cascade,
    source_video_id           text not null,
    title                     text,
    source_file               text,
    b2_bucket                 text,
    b2_path                   text,
    thumbnail_b2_path         text,
    transcript_drive_file_id  text,
    transcript_url            text,
    summary_drive_file_id     text,
    summary_url               text,
    status                    text not null default 'completed'
                              check (status in ('processing','completed','failed')),
    error                     text,
    created_at                timestamptz default now(),
    updated_at                timestamptz default now(),
    unique (client_id, source_video_id)
);

create unique index if not exists ingestion_manifests_transcript_file_uidx
    on public.ingestion_manifests (transcript_drive_file_id)
    where transcript_drive_file_id is not null;

create unique index if not exists ingestion_manifests_summary_file_uidx
    on public.ingestion_manifests (summary_drive_file_id)
    where summary_drive_file_id is not null;

create index if not exists ingestion_manifests_client_idx
    on public.ingestion_manifests (client_id, updated_at desc);

drop trigger if exists trg_ingestion_manifests_updated on public.ingestion_manifests;
create trigger trg_ingestion_manifests_updated
    before update on public.ingestion_manifests
    for each row execute function public.set_updated_at();

drop trigger if exists trg_validate_client_ingestion_manifests on public.ingestion_manifests;
create trigger trg_validate_client_ingestion_manifests
    before insert or update on public.ingestion_manifests
    for each row execute function public.validate_client_id();

alter table public.ingestion_manifests enable row level security;
revoke all on public.ingestion_manifests from authenticated, anon;

create or replace function public.admin_complete_ingestion(
    p_idempotency_key text,
    p_client_id uuid,
    p_source_video_id text,
    p_title text,
    p_source_file text,
    p_b2_bucket text,
    p_b2_path text,
    p_thumbnail_b2_path text,
    p_transcripts_folder_id text,
    p_summaries_folder_id text,
    p_transcript_drive_file_id text,
    p_transcript_url text,
    p_summary_drive_file_id text,
    p_summary_url text
)
returns table (
    ok boolean,
    source_video_id text,
    transcript_file_id text,
    transcript_url text,
    summary_file_id text,
    summary_url text
)
language plpgsql
security definer
set search_path = public
as $$
#variable_conflict use_column
declare
    v_client public.clients_registry%rowtype;
begin
    select *
      into v_client
      from public.clients_registry
     where client_id = p_client_id
       and status = 'active'
       and drive_transcripts_intake_folder_id = p_transcripts_folder_id
       and drive_summaries_intake_folder_id = p_summaries_folder_id;

    if not found then
        raise exception 'Ingestion destination is not registered for client %', p_client_id;
    end if;

    if p_b2_path is not null
       and coalesce(v_client.b2_bucket, '') <> coalesce(p_b2_bucket, '') then
        raise exception 'B2 bucket does not match clients_registry for client %', p_client_id;
    end if;

    insert into public.ingestion_manifests (
        idempotency_key, client_id, source_video_id, title, source_file,
        b2_bucket, b2_path, thumbnail_b2_path,
        transcript_drive_file_id, transcript_url,
        summary_drive_file_id, summary_url, status, error
    )
    values (
        p_idempotency_key, p_client_id, p_source_video_id, p_title, p_source_file,
        p_b2_bucket, p_b2_path, p_thumbnail_b2_path,
        p_transcript_drive_file_id, p_transcript_url,
        p_summary_drive_file_id, p_summary_url, 'completed', null
    )
    on conflict (client_id, source_video_id) do update set
        idempotency_key          = excluded.idempotency_key,
        title                    = excluded.title,
        source_file              = excluded.source_file,
        b2_bucket                = coalesce(excluded.b2_bucket, ingestion_manifests.b2_bucket),
        b2_path                  = coalesce(excluded.b2_path, ingestion_manifests.b2_path),
        thumbnail_b2_path        = coalesce(excluded.thumbnail_b2_path, ingestion_manifests.thumbnail_b2_path),
        transcript_drive_file_id = excluded.transcript_drive_file_id,
        transcript_url           = excluded.transcript_url,
        summary_drive_file_id    = excluded.summary_drive_file_id,
        summary_url              = excluded.summary_url,
        status                   = 'completed',
        error                    = null;

    insert into public.video_summaries (
        client_id, source_video_id, title, source_file, b2_path, thumbnail_b2_path
    )
    values (
        p_client_id, p_source_video_id, p_title, p_source_file, p_b2_path, p_thumbnail_b2_path
    )
    on conflict (client_id, source_video_id) do update set
        title             = coalesce(excluded.title, video_summaries.title),
        source_file       = coalesce(excluded.source_file, video_summaries.source_file),
        b2_path           = coalesce(excluded.b2_path, video_summaries.b2_path),
        thumbnail_b2_path = coalesce(excluded.thumbnail_b2_path, video_summaries.thumbnail_b2_path),
        updated_at        = now();

    return query
    select true, p_source_video_id, p_transcript_drive_file_id, p_transcript_url,
           p_summary_drive_file_id, p_summary_url;
end;
$$;

revoke all on function public.admin_complete_ingestion(
    text, uuid, text, text, text, text, text, text, text, text, text, text, text, text
) from public, anon, authenticated;
grant execute on function public.admin_complete_ingestion(
    text, uuid, text, text, text, text, text, text, text, text, text, text, text, text
) to service_role;

-- Existing lowercase test-company mapping. Replace the two placeholders before
-- running this update; the intake IDs came from the current list-clients response.
--
-- update public.clients_registry
-- set b2_bucket = 'VPStorage-testcompany',
--     b2_prefix = '',
--     drive_transcripts_intake_folder_id = '151MoUx2IgMFmgFQ-fo-yyncJf865m1Sk',
--     drive_summaries_intake_folder_id = '150mp-jn2U6O8YT_KjPNqkYImGXDLsdzE',
--     drive_transcripts_completed_folder_id = '<TEST_COMPANY_TRANSCRIPTS_COMPLETED_ID>',
--     drive_summaries_completed_folder_id = '<TEST_COMPANY_SUMMARIES_COMPLETED_ID>'
-- where client_id = 'fdefa887-b08f-43ba-8efc-b5c6bcf3e25a';
