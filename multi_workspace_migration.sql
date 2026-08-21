-- Multi-workspace memberships, invitations, and ingestion attribution.
-- Run once in Supabase SQL Editor before deploying task/multi-workspace-management.

create table if not exists public.workspace_memberships (
    client_id uuid not null references public.clients_registry(client_id) on delete cascade,
    user_id uuid not null references auth.users(id) on delete cascade,
    role text not null default 'member' check (role in ('owner', 'admin', 'member')),
    display_name text,
    email text,
    invited_by uuid references auth.users(id) on delete set null,
    joined_at timestamptz not null default now(),
    updated_at timestamptz not null default now(),
    primary key (client_id, user_id)
);

create index if not exists workspace_memberships_user_id_idx
    on public.workspace_memberships(user_id);

-- Preserve every existing user's current workspace and role.
insert into public.workspace_memberships (client_id, user_id, role, display_name, email)
select client_id,
       profiles.user_id,
       case when profiles.role = 'admin' then 'admin' else 'member' end,
       profiles.display_name,
       users.email
  from public.user_profiles profiles
  left join auth.users users on users.id = profiles.user_id
on conflict (client_id, user_id) do nothing;

-- Give each existing self-service workspace one non-removable owner.
with first_admin as (
    select distinct on (members.client_id) members.client_id, members.user_id
      from public.workspace_memberships members
      join public.clients_registry clients using (client_id)
     where members.role = 'admin'
       and coalesce((clients.metadata ->> 'self_service')::boolean, false)
     order by members.client_id, members.joined_at, members.user_id
)
update public.workspace_memberships members
   set role = 'owner', updated_at = now()
  from first_admin
 where members.client_id = first_admin.client_id
   and members.user_id = first_admin.user_id;

create table if not exists public.workspace_invites (
    invite_id uuid primary key default gen_random_uuid(),
    client_id uuid not null references public.clients_registry(client_id) on delete cascade,
    token_hash text not null unique,
    role text not null default 'member' check (role in ('admin', 'member')),
    created_by uuid not null references auth.users(id) on delete cascade,
    expires_at timestamptz not null,
    max_uses integer not null default 1 check (max_uses > 0),
    use_count integer not null default 0 check (use_count >= 0),
    revoked_at timestamptz,
    created_at timestamptz not null default now()
);

create index if not exists workspace_invites_client_id_idx
    on public.workspace_invites(client_id);

alter table public.video_summaries
    add column if not exists uploaded_by_user_id uuid references auth.users(id) on delete set null,
    add column if not exists uploaded_by_email text;

-- Transactionally create an additional workspace without replacing the
-- caller's legacy user_profiles/default-workspace row.
create or replace function public.create_additional_workspace(
    p_user_id uuid,
    p_slug text,
    p_display_name text
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
        slug, display_name, source_kind, plan_tier, b2_prefix, metadata
    ) values (
        p_slug, p_display_name, 'managed', 'standard', '',
        '{"self_service":true,"storage_provisioning_status":"pending"}'::jsonb
    ) returning client_id into v_client_id;

    insert into public.brand_doctrine (client_id, version, name, description, rubric)
    values (
      v_client_id, 1, 'Core Doctrine',
      'Default rubric - replace with client-specific doctrine.',
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

    insert into public.workspace_memberships (
        client_id, user_id, role, display_name, email
    ) values (
        v_client_id, p_user_id, 'owner', p_display_name,
        (select email from auth.users where id = p_user_id)
    );

    return v_client_id;
end;
$$;

revoke execute on function public.create_additional_workspace(uuid, text, text)
    from public, anon, authenticated;
grant execute on function public.create_additional_workspace(uuid, text, text)
    to service_role;

create or replace function public.accept_workspace_invite(
    p_token_hash text,
    p_user_id uuid,
    p_email text,
    p_display_name text
)
returns uuid
language plpgsql
security definer
set search_path = public
as $$
declare
    v_invite public.workspace_invites%rowtype;
begin
    select * into v_invite
      from public.workspace_invites
     where token_hash = p_token_hash
     for update;

    if v_invite.invite_id is null
       or v_invite.revoked_at is not null
       or v_invite.expires_at <= now()
       or v_invite.use_count >= v_invite.max_uses then
        raise exception 'Invite is invalid or expired';
    end if;

    insert into public.workspace_memberships (
        client_id, user_id, role, display_name, email, invited_by
    ) values (
        v_invite.client_id, p_user_id, v_invite.role,
        nullif(p_display_name, ''), nullif(p_email, ''), v_invite.created_by
    ) on conflict (client_id, user_id) do nothing;

    if found then
        update public.workspace_invites
           set use_count = use_count + 1
         where invite_id = v_invite.invite_id;
    end if;

    return v_invite.client_id;
end;
$$;

revoke execute on function public.accept_workspace_invite(text, uuid, text, text)
    from public, anon, authenticated;
grant execute on function public.accept_workspace_invite(text, uuid, text, text)
    to service_role;

-- Keep the first-workspace RPC compatible with the membership model.
create or replace function public.sync_legacy_workspace_membership()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
    insert into public.workspace_memberships (client_id, user_id, role, display_name, email)
    values (
        new.client_id,
        new.user_id,
        case
          when new.role = 'admin'
           and coalesce((select (metadata ->> 'self_service')::boolean
                           from public.clients_registry
                          where client_id = new.client_id), false)
           and not exists (
               select 1 from public.workspace_memberships
                where client_id = new.client_id and role = 'owner'
           ) then 'owner'
          when new.role = 'admin' then 'admin'
          else 'member'
        end,
        new.display_name,
        (select email from auth.users where id = new.user_id)
    )
    on conflict (client_id, user_id) do update set
        display_name = excluded.display_name,
        updated_at = now();
    return new;
end;
$$;

drop trigger if exists user_profiles_sync_workspace_membership on public.user_profiles;
create trigger user_profiles_sync_workspace_membership
after insert or update of client_id, role, display_name on public.user_profiles
for each row execute function public.sync_legacy_workspace_membership();

alter table public.workspace_memberships enable row level security;
alter table public.workspace_invites enable row level security;

revoke all on public.workspace_memberships from anon, authenticated;
revoke all on public.workspace_invites from anon, authenticated;
grant all on public.workspace_memberships to service_role;
grant all on public.workspace_invites to service_role;
