-- ============================================================================
--  VIDEO PILOT — COMPLETE DATABASE SCHEMA
--  One-shot, copy-paste safe for the Supabase SQL Editor.
--  Uses IF NOT EXISTS / CREATE OR REPLACE / DROP IF EXISTS throughout so it
--  can be run on a fresh database or re-applied to an existing one safely.
--
--  Ordering matters: tables are created before any function that references
--  them, so forward-reference errors cannot occur.
-- ============================================================================


-- ============================================================================
--  EXTENSIONS
-- ============================================================================

create extension if not exists "uuid-ossp";
create extension if not exists "vector";
create extension if not exists "pg_trgm";


-- ============================================================================
--  UTILITY FUNCTIONS — no table references (safe to create first)
-- ============================================================================

-- Auto-updates updated_at on any row modification.
create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

-- Returns the Supabase Auth user UUID for the current request (auth.uid() wrapper).
create or replace function public.current_user_id()
returns uuid
language sql
stable
as $$
  select auth.uid()
$$;

grant execute on function public.current_user_id() to authenticated, anon;


-- ============================================================================
--  TABLES — in dependency order
-- ============================================================================

-- ---------- CLIENTS REGISTRY ----------
-- The tenant directory. Admin-only access (no RLS — access denied via REVOKE).
-- Holds tenant identity, plan info, and ingestion source preferences.
create table if not exists public.clients_registry (
    client_id          uuid primary key default uuid_generate_v4(),
    slug               text unique not null,
    display_name       text not null,
    b2_bucket          text,
    b2_prefix          text default '',
    drive_transcripts_intake_folder_id    text,
    drive_summaries_intake_folder_id      text,
    drive_transcripts_completed_folder_id text,
    drive_summaries_completed_folder_id   text,
    source_kind        text not null default 'managed'
                       check (source_kind in ('managed','b2','gdrive','dropbox','manual')),
    status             text not null default 'active'
                       check (status in ('active','paused','archived')),
    plan_tier          text not null default 'standard'
                       check (plan_tier in ('standard','premium','enterprise')),
    reasoning_budget_s integer not null default 600,
    last_sync_at       timestamptz,
    created_at         timestamptz default now(),
    updated_at         timestamptz default now(),
    metadata           jsonb default '{}'::jsonb
);

-- Keep upgrades idempotent for clients created before Drive ingestion mapping.
alter table public.clients_registry
    add column if not exists drive_transcripts_intake_folder_id text,
    add column if not exists drive_summaries_intake_folder_id text,
    add column if not exists drive_transcripts_completed_folder_id text,
    add column if not exists drive_summaries_completed_folder_id text;

create unique index if not exists clients_registry_transcripts_intake_uidx
    on public.clients_registry (drive_transcripts_intake_folder_id)
    where drive_transcripts_intake_folder_id is not null;

create unique index if not exists clients_registry_summaries_intake_uidx
    on public.clients_registry (drive_summaries_intake_folder_id)
    where drive_summaries_intake_folder_id is not null;

create index if not exists clients_registry_active_idx
    on public.clients_registry (status) where status = 'active';

drop trigger if exists trg_clients_updated on public.clients_registry;
create trigger trg_clients_updated
    before update on public.clients_registry
    for each row execute function public.set_updated_at();

-- Only service_role touches this table directly.
revoke all on public.clients_registry from authenticated, anon;


-- ---------- USER PROFILES ----------
-- Links a Supabase Auth user (auth.users) to exactly one client (tenant).
-- Created by the backend POST /admin/users endpoint or admin_link_user_to_client().
-- One user → one client (enforced by primary key on user_id).
-- Role scopes admin vs. member access within the tenant.
create table if not exists public.user_profiles (
    user_id       uuid primary key references auth.users(id) on delete cascade,
    client_id     uuid not null references public.clients_registry(client_id) on delete cascade,
    role          text not null default 'member'
                  check (role in ('admin', 'member')),
    display_name  text,
    created_at    timestamptz default now(),
    updated_at    timestamptz default now()
);

create index if not exists user_profiles_client_idx
    on public.user_profiles (client_id);

drop trigger if exists trg_user_profiles_updated on public.user_profiles;
create trigger trg_user_profiles_updated
    before update on public.user_profiles
    for each row execute function public.set_updated_at();


-- ============================================================================
--  current_client_id() — defined AFTER user_profiles so the table reference
--  inside the SQL function body resolves without error.
-- ============================================================================

-- Returns the client_id for the current request.
-- Priority 1: explicit client_id JWT claim in the JWT before calling Supabase.
-- Priority 2: user_profiles lookup for Supabase Auth users (JWT has sub, not client_id).
-- security definer so the user_profiles subquery bypasses user_profiles' own RLS.
create or replace function public.current_client_id()
returns uuid
language sql
stable
security definer
set search_path = public
as $$
  select coalesce(
    -- Priority 1: explicit claim in JWT (service-role)
    nullif(
      coalesce(current_setting('request.jwt.claims', true)::jsonb ->> 'client_id', ''),
      ''
    )::uuid,
    -- Priority 2: derive from user_profiles for Supabase Auth users
    (select client_id from public.user_profiles where user_id = auth.uid())
  )
$$;

grant execute on function public.current_client_id() to authenticated, anon;


-- ---------- VIDEO SUMMARIES ----------
-- One row per source video. Holds metadata + optional summary.
-- summary_text and summary_embedding are nullable — filled in when a summary
-- file is ingested. Transcripts can be ingested without a summary.
create table if not exists public.video_summaries (
    summary_id        uuid primary key default uuid_generate_v4(),
    client_id         uuid not null references public.clients_registry(client_id) on delete cascade,
    source_video_id   text not null,
    title             text,
    source_file       text,
    has_timestamps    boolean default true,
    duration_seconds  integer,
    recorded_at       date,
    summary_text      text,
    summary_embedding vector(1536),
    topics            text[],
    speakers          text[],
    quality_score     numeric(3,2),
    b2_path           text,
    thumbnail_url     text,
    thumbnail_b2_path text,
    created_at        timestamptz default now(),
    updated_at        timestamptz default now(),
    unique (client_id, source_video_id)
);

-- Keep upgrades idempotent for databases created before media-library thumbnails.
alter table public.video_summaries
    add column if not exists thumbnail_url text,
    add column if not exists thumbnail_b2_path text;

create index if not exists video_summaries_client_idx
    on public.video_summaries (client_id);

create index if not exists video_summaries_embedding_idx
    on public.video_summaries using hnsw (summary_embedding vector_cosine_ops);

create index if not exists video_summaries_fts_idx
    on public.video_summaries using gin (to_tsvector('english', coalesce(summary_text, '')));

create index if not exists video_summaries_trgm_idx
    on public.video_summaries using gin (summary_text gin_trgm_ops);

drop trigger if exists trg_video_summaries_updated on public.video_summaries;
create trigger trg_video_summaries_updated
    before update on public.video_summaries
    for each row execute function public.set_updated_at();


-- ---------- INGESTION MANIFESTS ----------
-- Durable handoff between the immediate Drive upload workflow and the scheduled
-- embedding/indexing workflow. One row binds the reviewed Drive artifacts to the
-- original B2 video and thumbnail.
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

-- Completes the immediate Drive handoff and creates/updates the lightweight
-- media row in one transaction. The scheduled workflow later adds embeddings
-- and transcript chunks to the same (client_id, source_video_id) record.
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


-- ---------- TRANSCRIPT SEGMENTS ----------
-- Chunks of transcript text with embeddings (~600 words, 100-word overlap).
-- Carries denormalized source_title and source_file for direct agent citation
-- without needing a join back to video_summaries.
create table if not exists public.transcript_segments (
    segment_id        uuid primary key default uuid_generate_v4(),
    client_id         uuid not null references public.clients_registry(client_id) on delete cascade,
    source_video_id   text not null,
    source_title      text,
    source_file       text,
    has_timestamps    boolean default true,
    summary_id        uuid references public.video_summaries(summary_id) on delete cascade,
    chunk_index       integer not null,
    start_seconds     numeric(10,2) not null,
    end_seconds       numeric(10,2) not null,
    speaker           text,
    transcript_text   text not null,
    embedding         vector(1536),
    word_count        integer,
    created_at        timestamptz default now(),
    unique (client_id, source_video_id, chunk_index)
);

create index if not exists transcript_segments_client_idx
    on public.transcript_segments (client_id);

create index if not exists transcript_segments_video_idx
    on public.transcript_segments (client_id, source_video_id);

create index if not exists transcript_segments_embedding_idx
    on public.transcript_segments using hnsw (embedding vector_cosine_ops);

create index if not exists transcript_segments_fts_idx
    on public.transcript_segments using gin (to_tsvector('english', coalesce(transcript_text, '')));

create index if not exists transcript_segments_trgm_idx
    on public.transcript_segments using gin (transcript_text gin_trgm_ops);


-- ---------- BRAND DOCTRINE ----------
-- Per-tenant evaluation rubric used by the Critique agent.
-- Versioned so doctrine can evolve without losing decision history.
-- Auto-seeded with a default rubric when admin_provision_client runs.
create table if not exists public.brand_doctrine (
    doctrine_id       uuid primary key default uuid_generate_v4(),
    client_id         uuid not null references public.clients_registry(client_id) on delete cascade,
    version           integer not null default 1,
    name              text not null,
    description       text not null,
    rubric            jsonb not null,
    voice_guidelines  text,
    forbidden_topics  text[],
    is_active         boolean default true,
    created_at        timestamptz default now(),
    updated_at        timestamptz default now(),
    unique (client_id, version, name)
);

create index if not exists brand_doctrine_active_idx
    on public.brand_doctrine (client_id) where is_active = true;

drop trigger if exists trg_brand_doctrine_updated on public.brand_doctrine;
create trigger trg_brand_doctrine_updated
    before update on public.brand_doctrine
    for each row execute function public.set_updated_at();


-- ---------- ASSET CATALOG ----------
-- Reusable creative building blocks per tenant.
-- The Director agent maps clip beats to these assets via embedding match.
create table if not exists public.asset_catalog (
    asset_id          uuid primary key default uuid_generate_v4(),
    client_id         uuid not null references public.clients_registry(client_id) on delete cascade,
    asset_type        text not null
                      check (asset_type in ('character','broll_prompt','voice','music','overlay')),
    name              text not null,
    description       text,
    parameters        jsonb default '{}'::jsonb,
    tags              text[],
    embedding         vector(1536),
    is_active         boolean default true,
    created_at        timestamptz default now(),
    updated_at        timestamptz default now()
);

create index if not exists asset_catalog_client_type_idx
    on public.asset_catalog (client_id, asset_type) where is_active = true;

create index if not exists asset_catalog_embedding_idx
    on public.asset_catalog using hnsw (embedding vector_cosine_ops);

drop trigger if exists trg_asset_catalog_updated on public.asset_catalog;
create trigger trg_asset_catalog_updated
    before update on public.asset_catalog
    for each row execute function public.set_updated_at();


-- ---------- JOB RUNS ----------
-- Audit log for ingestions, blueprints, and reindexes.
-- Tracks tokens, cost, errors. Used for cost attribution and debugging.
create table if not exists public.job_runs (
    job_run_id        uuid primary key default uuid_generate_v4(),
    client_id         uuid not null references public.clients_registry(client_id) on delete cascade,
    kind              text not null check (kind in ('sync','blueprint','reindex','ingest')),
    status            text not null default 'running'
                      check (status in ('running','succeeded','failed','cancelled')),
    started_at        timestamptz default now(),
    finished_at       timestamptz,
    tokens_input      integer default 0,
    tokens_output     integer default 0,
    cost_usd          numeric(10,4) default 0,
    error             text,
    payload           jsonb default '{}'::jsonb
);

create index if not exists job_runs_client_idx
    on public.job_runs (client_id, started_at desc);


-- ---------- BLUEPRINTS ----------
-- Final Production Blueprint outputs from the Director agent.
create table if not exists public.blueprints (
    blueprint_id      uuid primary key default uuid_generate_v4(),
    client_id         uuid not null references public.clients_registry(client_id) on delete cascade,
    job_run_id        uuid references public.job_runs(job_run_id) on delete set null,
    title             text not null,
    brief             text,
    blueprint_json    jsonb not null,
    status            text not null default 'draft'
                      check (status in ('draft','approved','rejected','published')),
    created_at        timestamptz default now(),
    updated_at        timestamptz default now()
);

create index if not exists blueprints_client_idx
    on public.blueprints (client_id, created_at desc);

drop trigger if exists trg_blueprints_updated on public.blueprints;
create trigger trg_blueprints_updated
    before update on public.blueprints
    for each row execute function public.set_updated_at();


-- ---------- CHAT SESSIONS ----------
-- Per-user session directory. This table owns WHO owns WHICH thread.
-- LangGraph checkpoint tables (below) own the actual message/state content.
-- session_id here == thread_id in LangGraph — the link between the two systems.
create table if not exists public.chat_sessions (
    session_id       text primary key,
    client_id        uuid not null references public.clients_registry(client_id) on delete cascade,
    user_id          uuid not null references auth.users(id) on delete cascade,
    title            text,
    message_count    integer not null default 0,
    last_message_at  timestamptz,
    status           text not null default 'active'
                     check (status in ('active', 'archived')),
    created_at       timestamptz default now(),
    updated_at       timestamptz default now()
);

create index if not exists chat_sessions_user_idx
    on public.chat_sessions (user_id, last_message_at desc);

create index if not exists chat_sessions_client_idx
    on public.chat_sessions (client_id);

drop trigger if exists trg_chat_sessions_updated on public.chat_sessions;
create trigger trg_chat_sessions_updated
    before update on public.chat_sessions
    for each row execute function public.set_updated_at();


-- ============================================================================
--  LANGGRAPH CHECKPOINT TABLES
--  Auto-created by AsyncPostgresSaver.setup() at backend startup.
--  Included here as reference — IF NOT EXISTS makes them safe to re-run.
--  thread_id in these tables == session_id in chat_sessions.
-- ============================================================================

create table if not exists checkpoints (
    thread_id            text    not null,
    checkpoint_ns        text    not null default '',
    checkpoint_id        text    not null,
    parent_checkpoint_id text,
    type                 text,
    checkpoint           jsonb   not null default '{}',
    metadata             jsonb   not null default '{}',
    primary key (thread_id, checkpoint_ns, checkpoint_id)
);

create table if not exists checkpoint_blobs (
    thread_id     text  not null,
    checkpoint_ns text  not null default '',
    channel       text  not null,
    version       text  not null,
    type          text  not null,
    blob          bytea,
    primary key (thread_id, checkpoint_ns, channel, version)
);

create table if not exists checkpoint_writes (
    thread_id     text    not null,
    checkpoint_ns text    not null default '',
    checkpoint_id text    not null,
    task_id       text    not null,
    idx           integer not null,
    channel       text    not null,
    type          text,
    blob          bytea   not null,
    primary key (thread_id, checkpoint_ns, checkpoint_id, task_id, idx)
);

create table if not exists checkpoint_migrations (
    v integer primary key
);


-- ============================================================================
--  VALIDATION + DENORMALIZATION TRIGGERS
-- ============================================================================

-- Validates that any client_id used in inserts exists and is active in
-- clients_registry. Safety net for service_role ingestion (bypasses RLS).
create or replace function public.validate_client_id()
returns trigger
language plpgsql
as $$
begin
  if not exists (
    select 1 from public.clients_registry
    where client_id = new.client_id and status = 'active'
  ) then
    raise exception 'Invalid or inactive client_id: %', new.client_id;
  end if;
  return new;
end;
$$;

drop trigger if exists trg_validate_client_video_summaries on public.video_summaries;
create trigger trg_validate_client_video_summaries
  before insert or update on public.video_summaries
  for each row execute function public.validate_client_id();

drop trigger if exists trg_validate_client_transcript_segments on public.transcript_segments;
create trigger trg_validate_client_transcript_segments
  before insert or update on public.transcript_segments
  for each row execute function public.validate_client_id();

drop trigger if exists trg_validate_client_ingestion_manifests on public.ingestion_manifests;
create trigger trg_validate_client_ingestion_manifests
  before insert or update on public.ingestion_manifests
  for each row execute function public.validate_client_id();


-- Auto-fills source_title and source_file on new transcript_segments from
-- the parent video_summaries row. Saves the agent from needing a join.
create or replace function public.sync_segment_denorm()
returns trigger
language plpgsql
as $$
begin
  if new.summary_id is not null and (new.source_title is null or new.source_file is null) then
    select vs.title, vs.source_file
      into new.source_title, new.source_file
    from public.video_summaries vs
    where vs.summary_id = new.summary_id;
  end if;
  return new;
end;
$$;

drop trigger if exists trg_sync_segment_denorm on public.transcript_segments;
create trigger trg_sync_segment_denorm
  before insert or update on public.transcript_segments
  for each row execute function public.sync_segment_denorm();


-- Propagates title/source_file changes from video_summaries to all
-- linked transcript_segments rows. Keeps denormalized data in sync.
create or replace function public.propagate_summary_changes()
returns trigger
language plpgsql
as $$
begin
  if (new.title is distinct from old.title) or (new.source_file is distinct from old.source_file) then
    update public.transcript_segments
       set source_title = new.title,
           source_file  = new.source_file
     where summary_id = new.summary_id;
  end if;
  return new;
end;
$$;

drop trigger if exists trg_propagate_summary_changes on public.video_summaries;
create trigger trg_propagate_summary_changes
  after update on public.video_summaries
  for each row execute function public.propagate_summary_changes();


-- ============================================================================
--  ROW-LEVEL SECURITY
-- ============================================================================

alter table public.video_summaries     enable row level security;
alter table public.transcript_segments enable row level security;
alter table public.ingestion_manifests enable row level security;
alter table public.brand_doctrine      enable row level security;
alter table public.asset_catalog       enable row level security;
alter table public.blueprints          enable row level security;
alter table public.job_runs            enable row level security;
alter table public.user_profiles       enable row level security;
alter table public.chat_sessions       enable row level security;

revoke all on public.ingestion_manifests from authenticated, anon;

-- Tenant-scoped tables: read/write only your own client's rows.
-- current_client_id() resolves from JWT claim (service-role) or
-- user_profiles lookup (Supabase Auth users) — transparent to the policy.

drop policy if exists tenant_select on public.video_summaries;
drop policy if exists tenant_modify on public.video_summaries;
create policy tenant_select on public.video_summaries
    for select using (client_id = public.current_client_id());
create policy tenant_modify on public.video_summaries
    for all using (client_id = public.current_client_id())
            with check (client_id = public.current_client_id());

drop policy if exists tenant_select on public.transcript_segments;
drop policy if exists tenant_modify on public.transcript_segments;
create policy tenant_select on public.transcript_segments
    for select using (client_id = public.current_client_id());
create policy tenant_modify on public.transcript_segments
    for all using (client_id = public.current_client_id())
            with check (client_id = public.current_client_id());

drop policy if exists tenant_select on public.brand_doctrine;
drop policy if exists tenant_modify on public.brand_doctrine;
create policy tenant_select on public.brand_doctrine
    for select using (client_id = public.current_client_id());
create policy tenant_modify on public.brand_doctrine
    for all using (client_id = public.current_client_id())
            with check (client_id = public.current_client_id());

drop policy if exists tenant_select on public.asset_catalog;
drop policy if exists tenant_modify on public.asset_catalog;
create policy tenant_select on public.asset_catalog
    for select using (client_id = public.current_client_id());
create policy tenant_modify on public.asset_catalog
    for all using (client_id = public.current_client_id())
            with check (client_id = public.current_client_id());

drop policy if exists tenant_select on public.blueprints;
drop policy if exists tenant_modify on public.blueprints;
create policy tenant_select on public.blueprints
    for select using (client_id = public.current_client_id());
create policy tenant_modify on public.blueprints
    for all using (client_id = public.current_client_id())
            with check (client_id = public.current_client_id());

drop policy if exists tenant_select on public.job_runs;
drop policy if exists tenant_modify on public.job_runs;
create policy tenant_select on public.job_runs
    for select using (client_id = public.current_client_id());
create policy tenant_modify on public.job_runs
    for all using (client_id = public.current_client_id())
            with check (client_id = public.current_client_id());

-- User profiles: each user sees and edits only their own row.
-- Inserts are service-role only (via admin_link_user_to_client / POST /admin/users).
drop policy if exists own_profile_select on public.user_profiles;
drop policy if exists own_profile_update on public.user_profiles;
create policy own_profile_select on public.user_profiles
    for select using (user_id = auth.uid());
create policy own_profile_update on public.user_profiles
    for update using (user_id = auth.uid())
             with check (user_id = auth.uid());

-- Chat sessions: each user sees and modifies only their own sessions.
-- Inserts are service-role only (backend manages session creation on first message).
drop policy if exists own_sessions_select on public.chat_sessions;
drop policy if exists own_sessions_modify on public.chat_sessions;
create policy own_sessions_select on public.chat_sessions
    for select using (user_id = auth.uid());
create policy own_sessions_modify on public.chat_sessions
    for all using (user_id = auth.uid())
             with check (user_id = auth.uid());


-- ============================================================================
--  GRANTS
-- ============================================================================

grant usage on schema public to authenticated, anon;

grant select, insert, update, delete on
    public.video_summaries,
    public.transcript_segments,
    public.brand_doctrine,
    public.asset_catalog,
    public.blueprints,
    public.job_runs
to authenticated;

-- Users can read their own profile; update display_name only (not client_id/role).
grant select on public.user_profiles to authenticated;
grant update (display_name) on public.user_profiles to authenticated;

-- Users can read their own sessions (backend inserts/updates via service role).
grant select on public.chat_sessions to authenticated;


-- ============================================================================
--  SESSION HELPER
-- ============================================================================

-- Atomically increments message_count and bumps last_message_at.
-- Called by the backend via svc.rpc("increment_session_count", ...) after each
-- chat turn. Service-role only — revoked from all user-facing roles.
create or replace function public.increment_session_count(p_session_id text)
returns void
language sql
security definer
set search_path = public
as $$
  update public.chat_sessions
  set message_count   = message_count + 1,
      last_message_at = now(),
      updated_at      = now()
  where session_id = p_session_id
$$;

revoke execute on function public.increment_session_count(text) from public, anon, authenticated;


-- Soft-deletes a session by flipping its status to 'archived'. Archived sessions
-- are excluded from list_sessions (GET /sessions), so they disappear from the
-- sidebar while their LangGraph checkpoint history is preserved.
-- Called by the backend via svc.rpc("archive_session", ...) on DELETE /sessions/{id}.
-- Takes p_user_id and scopes the update to that owner so the service-role call
-- (which bypasses RLS) still cannot archive another user's session.
-- Returns true if a matching, still-active session was archived; false otherwise.
drop function if exists public.archive_session(text, uuid);
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


-- ============================================================================
--  SEARCH FUNCTIONS
-- ============================================================================

-- Hybrid semantic + lexical search over transcript chunks.
-- Combines pgvector cosine similarity with full-text search via Reciprocal Rank Fusion.
-- The orchestrator calls this RPC with the service-role key, which bypasses RLS.
-- Keep an explicit client predicate in every CTE as the authorization boundary.
drop function if exists public.hybrid_search_transcripts(text, vector, integer, integer, numeric);
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

revoke all on function public.hybrid_search_transcripts(uuid, text, vector, integer, integer, numeric)
    from public, anon, authenticated;
grant execute on function public.hybrid_search_transcripts(uuid, text, vector, integer, integer, numeric)
    to service_role;


-- Hybrid search over video summaries — broad discovery ("which videos are about X").
drop function if exists public.hybrid_search_summaries(text, vector, integer, integer);
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

revoke all on function public.hybrid_search_summaries(uuid, text, vector, integer, integer)
    from public, anon, authenticated;
grant execute on function public.hybrid_search_summaries(uuid, text, vector, integer, integer)
    to service_role;


-- Semantic match over the asset catalog — used by the Director agent to find
-- creative assets (characters, B-roll prompts, voices, etc.) matching a beat.
drop function if exists public.match_assets(vector, text[], integer);
create or replace function public.match_assets(
    p_client_id      uuid,
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

revoke all on function public.match_assets(uuid, vector, text[], integer)
    from public, anon, authenticated;
grant execute on function public.match_assets(uuid, vector, text[], integer)
    to service_role;


-- ============================================================================
--  ADMIN PROVISIONING
-- ============================================================================

-- Single-call client provisioning. Creates the clients_registry row and
-- seeds a default brand_doctrine. Call from the Supabase SQL Editor or the
-- backend admin flow.
drop function if exists public.admin_provision_client(text, text, text, text);
create or replace function public.admin_provision_client(
    p_slug text,
    p_display_name text,
    p_source_kind text default 'gdrive',
    p_b2_bucket text default null,
    p_b2_prefix text default null,
    p_plan_tier text default 'standard'
)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare
    v_client_id uuid;
begin
    insert into public.clients_registry (
        slug,
        display_name,
        source_kind,
        b2_bucket,
        b2_prefix,
        plan_tier
    )
    values (
        p_slug,
        p_display_name,
        p_source_kind,
        p_b2_bucket,
        coalesce(p_b2_prefix, ''),
        p_plan_tier
    )
    returning client_id into v_client_id;

    insert into public.brand_doctrine (client_id, version, name, description, rubric)
    values (
      v_client_id, 1, 'Core Doctrine',
      'Default rubric — replace with client-specific doctrine.',
      '{
        "dimensions": [
          {"key":"narrative_tension","weight":0.25,"description":"Conflict and stakes"},
          {"key":"doctrinal_alignment","weight":0.30,"description":"Brand fit"},
          {"key":"hook_strength","weight":0.20,"description":"Opening 3 seconds"},
          {"key":"quotability","weight":0.15,"description":"Memorable phrasing"},
          {"key":"production_quality","weight":0.10,"description":"Audio/video clarity"}
        ],
        "minimum_a_tier_score": 0.78,
        "auto_reject_below": 0.45
      }'::jsonb
    );

    return v_client_id;
end;
$$;

revoke execute on function public.admin_provision_client(text, text, text, text, text, text)
    from public, anon, authenticated;


-- Links an existing Supabase Auth user to a client tenant in user_profiles.
-- The auth user must already exist in auth.users (create via Supabase dashboard
-- or the backend POST /admin/users endpoint, then call this to link them).
-- Upserts: safe to call again if the user already has a profile.
create or replace function public.admin_link_user_to_client(
    p_user_id      uuid,
    p_client_id    uuid,
    p_role         text default 'member',
    p_display_name text default null
)
returns void
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.user_profiles (user_id, client_id, role, display_name)
    values (p_user_id, p_client_id, p_role, p_display_name)
    on conflict (user_id) do update
    set client_id    = excluded.client_id,
        role         = excluded.role,
        display_name = coalesce(excluded.display_name, user_profiles.display_name),
        updated_at   = now();
end;
$$;

revoke execute on function public.admin_link_user_to_client(uuid, uuid, text, text)
    from public, anon, authenticated;


-- ============================================================================
--  END OF SCHEMA
-- ============================================================================
