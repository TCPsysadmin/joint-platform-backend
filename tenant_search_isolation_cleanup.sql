-- Run only after the tenant-aware backend release is live on Render.
--
-- tenant_search_isolation_migration.sql deliberately leaves these legacy
-- overloads available during deployment so the previous backend version can
-- continue serving requests. Once the new backend is healthy, remove the old
-- signatures to close the temporary compatibility window.

begin;

drop function if exists public.archive_session(text, uuid);
drop function if exists public.hybrid_search_transcripts(
    text, vector, integer, integer, numeric
);
drop function if exists public.hybrid_search_summaries(
    text, vector, integer, integer
);
drop function if exists public.match_assets(vector, text[], integer);

commit;
