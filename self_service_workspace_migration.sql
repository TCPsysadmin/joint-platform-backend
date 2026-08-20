-- Self-service workspace provisioning for authenticated Supabase users.
-- Run this once in the Supabase SQL Editor before deploying the frontend flow.

create or replace function public.provision_user_workspace(
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
    -- Serialize retries for the same auth user. This prevents two concurrent
    -- onboarding requests from creating two clients before user_profiles exists.
    perform pg_advisory_xact_lock(hashtext(p_user_id::text));

    select client_id
      into v_client_id
      from public.user_profiles
     where user_id = p_user_id;

    if v_client_id is not null then
        return v_client_id;
    end if;

    insert into public.clients_registry (
        slug,
        display_name,
        source_kind,
        plan_tier,
        b2_prefix,
        metadata
    )
    values (
        p_slug,
        p_display_name,
        'managed',
        'standard',
        '',
        '{"self_service":true,"storage_provisioning_status":"pending"}'::jsonb
    )
    returning client_id into v_client_id;

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

    insert into public.user_profiles (user_id, client_id, role, display_name)
    values (p_user_id, v_client_id, 'admin', p_display_name);

    return v_client_id;
end;
$$;

revoke execute on function public.provision_user_workspace(uuid, text, text)
    from public, anon, authenticated;
grant execute on function public.provision_user_workspace(uuid, text, text)
    to service_role;
