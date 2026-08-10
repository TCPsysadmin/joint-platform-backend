from __future__ import annotations

import asyncio
import posixpath
from pathlib import Path
from typing import Any
from urllib.parse import quote
from uuid import UUID

import httpx
import structlog
from supabase import AsyncClient

from orchestrator.supabase_json import as_dict
from orchestrator.tools.local.b2_legend import (
    B2Legend,
    is_media_file,
    is_text_file,
    legend_path,
    load_legend,
    normalize_file_match_text,
    save_legend,
    sidecar_sort_rank,
    without_known_extension,
)
from orchestrator.tools.protocols import B2Bucket, B2FetchedFile, B2FileEntry, B2KeyInfo

logger = structlog.get_logger(__name__)

_B2_AUTH_URL = "https://api.backblazeb2.com/b2api/v3/b2_authorize_account"

# Cap pagination during filename-based search to avoid runaway scans of
# extremely large buckets. Most clients store far fewer than this.
_FIND_FILE_PAGE_SIZE = 1000
_FIND_FILE_MAX_PAGES = 10
# The legend sweep pages the whole bucket once; the cap only guards against a
# pathologically large tenant, not against per-lookup cost.
_LEGEND_MAX_PAGES = 100
# 401 codes that mean "your token went stale", i.e. re-authorize and retry once.
# `missing_auth_token` is deliberately excluded — that one is a client bug.
_RETRYABLE_AUTH_CODES = {"expired_auth_token", "bad_auth_token"}

# Re-exported for callers/tests that already import these names from this module.
_is_text_file = is_text_file
_normalize_file_match_text = normalize_file_match_text
_without_known_extension = without_known_extension

DEFAULT_LEGEND_CACHE_DIR = ".b2_legend"
DEFAULT_LEGEND_TTL_SECONDS = 900
DEFAULT_LEGEND_MIN_REFRESH_SECONDS = 60
DEFAULT_FETCH_CONCURRENCY = 6


class B2FileTool:
    """B2 client for downloading source transcripts and searching files.

    Auth (`authorizationToken`, `apiUrl`, `downloadUrl`, `accountId`) is fetched
    once on first use and cached for the lifetime of this instance — which is
    one request, since RuntimeContext is per-request. A 401 carrying
    `expired_auth_token` / `bad_auth_token` re-authorizes and retries that one
    call, so a batch that outlives its token doesn't lose the work already done.

    Bucket id resolution is also cached so listing operations after the first
    one don't re-hit b2_list_buckets.

    Filename lookups go through a *legend* — a JSON file-index of the tenant's
    bucket cached on local disk (see `b2_legend`) — instead of re-scanning the
    bucket on every lookup. `find_file_by_name` still falls back to the
    paginated scan, so a file uploaded since the last refresh is still found.
    """

    def __init__(
        self,
        supabase: AsyncClient,
        client_id: UUID,
        key_id: str,
        application_key: str,
        *,
        legend_cache_dir: str | Path = DEFAULT_LEGEND_CACHE_DIR,
        legend_ttl_seconds: int = DEFAULT_LEGEND_TTL_SECONDS,
        legend_min_refresh_seconds: int = DEFAULT_LEGEND_MIN_REFRESH_SECONDS,
        fetch_concurrency: int = DEFAULT_FETCH_CONCURRENCY,
    ) -> None:
        self._supabase = supabase
        self._client_id = client_id
        self._key_id = key_id
        self._application_key = application_key
        self._auth: dict[str, str] | None = None
        self._bucket_id_by_name: dict[str, str] = {}
        self._client_storage: dict[str, str] | None = None  # {"bucket": ..., "prefix": ...}
        self._legend_path = legend_path(legend_cache_dir, client_id)
        self._legend_ttl_seconds = legend_ttl_seconds
        self._legend_min_refresh_seconds = legend_min_refresh_seconds
        self._fetch_concurrency = max(1, fetch_concurrency)
        self._legend: B2Legend | None = None
        self._legend_disk_checked = False

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    async def _authorize(self) -> dict[str, str]:
        if self._auth:
            return self._auth
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.get(
                _B2_AUTH_URL,
                auth=(self._key_id, self._application_key),
            )
            resp.raise_for_status()
            data = resp.json()
        storage = data["apiInfo"]["storageApi"]
        self._auth = {
            "authorizationToken": data["authorizationToken"],
            "downloadUrl": storage["downloadUrl"],
            "apiUrl": storage["apiUrl"],
            "accountId": data["accountId"],
        }
        logger.debug("b2_authorized", account_id=data.get("accountId"))
        return self._auth

    async def _headers(self) -> dict[str, str]:
        auth = await self._authorize()
        return {"Authorization": auth["authorizationToken"]}

    # ------------------------------------------------------------------
    # Client storage config (bucket + prefix from clients_registry)
    # ------------------------------------------------------------------

    async def _get_client_storage(self) -> dict[str, str] | None:
        if self._client_storage is not None:
            return self._client_storage or None
        cr_resp = (
            await self._supabase.table("clients_registry")
            .select("b2_bucket, b2_prefix")
            .eq("client_id", str(self._client_id))
            .maybe_single()
            .execute()
        )
        cr_row = as_dict(cr_resp.data if cr_resp else None)
        if cr_row is None:
            logger.warning("b2_no_client_registry_row", client_id=str(self._client_id))
            self._client_storage = {}
            return None
        bucket = str(cr_row.get("b2_bucket") or "")
        if not bucket:
            logger.warning("b2_no_bucket_for_client", client_id=str(self._client_id))
            self._client_storage = {}
            return None
        prefix = str(cr_row.get("b2_prefix") or "").strip("/")
        self._client_storage = {"bucket": bucket, "prefix": prefix}
        return self._client_storage

    # ------------------------------------------------------------------
    # b2_list_buckets (POST) — JSON body
    # ------------------------------------------------------------------

    async def list_buckets(
        self,
        *,
        bucket_id: str | None = None,
        bucket_name: str | None = None,
        bucket_types: list[str] | None = None,
    ) -> list[B2Bucket]:
        auth = await self._authorize()
        body: dict[str, Any] = {"accountId": auth["accountId"]}
        if bucket_id:
            body["bucketId"] = bucket_id
        if bucket_name:
            body["bucketName"] = bucket_name
        if bucket_types:
            body["bucketTypes"] = bucket_types

        url = f"{auth['apiUrl']}/b2api/v3/b2_list_buckets"
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.post(url, json=body, headers=await self._headers())
            resp.raise_for_status()
            data = resp.json()

        buckets: list[B2Bucket] = []
        for row in data.get("buckets") or []:
            buckets.append(
                B2Bucket(
                    bucket_id=str(row["bucketId"]),
                    bucket_name=str(row["bucketName"]),
                    bucket_type=str(row.get("bucketType") or ""),
                    account_id=str(row["accountId"]) if row.get("accountId") else None,
                    metadata={
                        k: row[k]
                        for k in ("bucketInfo", "lifecycleRules", "revision", "options")
                        if k in row and row[k] is not None
                    },
                )
            )
        # Refresh the name → id cache opportunistically.
        for b in buckets:
            self._bucket_id_by_name[b.bucket_name] = b.bucket_id
        return buckets

    async def _resolve_bucket_id(self, bucket_name: str) -> str | None:
        cached = self._bucket_id_by_name.get(bucket_name)
        if cached:
            return cached
        buckets = await self.list_buckets(bucket_name=bucket_name)
        if not buckets:
            logger.warning("b2_bucket_not_found", bucket_name=bucket_name)
            return None
        return buckets[0].bucket_id

    # ------------------------------------------------------------------
    # b2_list_file_names (GET) — query params
    # ------------------------------------------------------------------

    async def list_file_names(
        self,
        *,
        bucket_id: str,
        prefix: str | None = None,
        delimiter: str | None = None,
        start_file_name: str | None = None,
        max_file_count: int = 100,
    ) -> tuple[list[B2FileEntry], str | None]:
        auth = await self._authorize()
        params: dict[str, Any] = {
            "bucketId": bucket_id,
            "maxFileCount": max_file_count,
        }
        if prefix:
            params["prefix"] = prefix
        if delimiter:
            params["delimiter"] = delimiter
        if start_file_name:
            params["startFileName"] = start_file_name

        url = f"{auth['apiUrl']}/b2api/v3/b2_list_file_names"
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(url, params=params, headers=await self._headers())
            resp.raise_for_status()
            data = resp.json()

        entries = [_parse_file_entry(row) for row in data.get("files") or []]
        next_file_name = data.get("nextFileName")
        return entries, (str(next_file_name) if next_file_name else None)

    # ------------------------------------------------------------------
    # b2_list_file_versions (GET) — query params
    # ------------------------------------------------------------------

    async def list_file_versions(
        self,
        *,
        bucket_id: str,
        prefix: str | None = None,
        delimiter: str | None = None,
        start_file_name: str | None = None,
        start_file_id: str | None = None,
        max_file_count: int = 100,
    ) -> tuple[list[B2FileEntry], str | None, str | None]:
        auth = await self._authorize()
        params: dict[str, Any] = {
            "bucketId": bucket_id,
            "maxFileCount": max_file_count,
        }
        if prefix:
            params["prefix"] = prefix
        if delimiter:
            params["delimiter"] = delimiter
        if start_file_name:
            params["startFileName"] = start_file_name
        if start_file_id:
            params["startFileId"] = start_file_id

        url = f"{auth['apiUrl']}/b2api/v3/b2_list_file_versions"
        async with httpx.AsyncClient(timeout=30.0) as http:
            resp = await http.get(url, params=params, headers=await self._headers())
            resp.raise_for_status()
            data = resp.json()

        entries = [_parse_file_entry(row) for row in data.get("files") or []]
        next_file_name = data.get("nextFileName")
        next_file_id = data.get("nextFileId")
        return (
            entries,
            str(next_file_name) if next_file_name else None,
            str(next_file_id) if next_file_id else None,
        )

    # ------------------------------------------------------------------
    # b2_list_keys (GET) — query params
    # ------------------------------------------------------------------

    async def list_keys(
        self,
        *,
        max_key_count: int = 100,
        start_application_key_id: str | None = None,
    ) -> tuple[list[B2KeyInfo], str | None]:
        auth = await self._authorize()
        params: dict[str, Any] = {
            "accountId": auth["accountId"],
            "maxKeyCount": max_key_count,
        }
        if start_application_key_id:
            params["startApplicationKeyId"] = start_application_key_id

        url = f"{auth['apiUrl']}/b2api/v3/b2_list_keys"
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.get(url, params=params, headers=await self._headers())
            resp.raise_for_status()
            data = resp.json()

        keys: list[B2KeyInfo] = []
        for row in data.get("keys") or []:
            keys.append(
                B2KeyInfo(
                    application_key_id=str(row["applicationKeyId"]),
                    key_name=str(row.get("keyName") or ""),
                    capabilities=list(row.get("capabilities") or []),
                    account_id=str(row["accountId"]) if row.get("accountId") else None,
                    bucket_id=str(row["bucketId"]) if row.get("bucketId") else None,
                    name_prefix=str(row["namePrefix"]) if row.get("namePrefix") else None,
                    expiration_timestamp=(
                        int(row["expirationTimestamp"]) if row.get("expirationTimestamp") else None
                    ),
                    metadata={"options": row.get("options") or []},
                )
            )
        next_key = data.get("nextApplicationKeyId")
        return keys, (str(next_key) if next_key else None)

    # ------------------------------------------------------------------
    # Legend — local file index, so a lookup is a dict access instead of a
    # multi-page bucket scan. See orchestrator/tools/local/b2_legend.py.
    # ------------------------------------------------------------------

    async def _build_legend(self, *, bucket: str, prefix: str) -> B2Legend | None:
        """One full paginated sweep of the tenant's bucket+prefix.

        Requests 1000 per page: B2 bills listing per 1000 entries returned, so
        asking for less just buys extra round trips at the same price.
        """
        bucket_id = await self._resolve_bucket_id(bucket)
        if not bucket_id:
            return None

        entries: list[B2FileEntry] = []
        cursor: str | None = None
        for _ in range(_LEGEND_MAX_PAGES):
            page, cursor = await self.list_file_names(
                bucket_id=bucket_id,
                prefix=prefix or None,
                start_file_name=cursor,
                max_file_count=_FIND_FILE_PAGE_SIZE,
            )
            entries.extend(page)
            if not cursor:
                break

        legend = B2Legend.from_entries(entries, bucket=bucket, prefix=prefix)
        logger.info(
            "b2_legend_built",
            client_id=str(self._client_id),
            bucket=bucket,
            prefix=prefix,
            listed=len(entries),
            indexed=legend.entry_count,
            truncated=bool(cursor),
        )
        return legend

    async def refresh_legend(
        self,
        *,
        bucket: str,
        prefix: str,
        min_age_seconds: int = 0,
    ) -> B2Legend | None:
        """Rebuild and persist the legend.

        `min_age_seconds` floors how often a rebuild may happen. Without it, a
        lookup for a file that genuinely does not exist would sweep the whole
        bucket *and* run the fallback scan on every single turn — strictly worse
        than the behaviour this feature replaces.

        A failed rebuild is logged and returns None: callers fall back to the
        paginated scan rather than failing the turn.
        """
        current = self._legend
        if (
            current is not None
            and current.matches(bucket, prefix)
            and min_age_seconds > 0
            and current.age_seconds() < min_age_seconds
        ):
            return current
        try:
            legend = await self._build_legend(bucket=bucket, prefix=prefix)
        except Exception as exc:
            logger.warning("b2_legend_build_failed", bucket=bucket, error=str(exc))
            return None
        if legend is None:
            return None
        self._legend = legend
        save_legend(self._legend_path, legend)
        return legend

    async def _get_legend(self, *, bucket: str, prefix: str) -> B2Legend | None:
        """In-memory → on-disk → rebuild, honouring the TTL at each step."""
        cached = self._legend
        if (
            cached is not None
            and cached.matches(bucket, prefix)
            and not cached.is_stale(self._legend_ttl_seconds)
        ):
            return cached

        if not self._legend_disk_checked:
            self._legend_disk_checked = True
            from_disk = load_legend(self._legend_path)
            if (
                from_disk is not None
                and from_disk.matches(bucket, prefix)
                and not from_disk.is_stale(self._legend_ttl_seconds)
            ):
                self._legend = from_disk
                return from_disk

        return await self.refresh_legend(bucket=bucket, prefix=prefix)

    async def _resolve_client_location(
        self,
        bucket_name: str | None,
        prefix: str | None,
    ) -> tuple[str, str, bool] | None:
        """(bucket, prefix, is_client_default) for a lookup's target location.

        The legend indexes exactly one bucket+prefix — the client's configured
        one — so a caller that overrides either must bypass it.
        """
        storage = await self._get_client_storage()
        if bucket_name and prefix is not None:
            is_default = storage is not None and (
                bucket_name == storage["bucket"] and prefix == (storage.get("prefix") or "")
            )
            return bucket_name, prefix, is_default
        if storage is None:
            return None
        return (
            bucket_name or storage["bucket"],
            prefix if prefix is not None else (storage.get("prefix") or ""),
            bucket_name in (None, storage["bucket"]),
        )

    # ------------------------------------------------------------------
    # High-level search: find a file when only the basename is known
    # ------------------------------------------------------------------

    async def find_file_by_name(
        self,
        *,
        file_name: str,
        bucket_name: str | None = None,
        prefix: str | None = None,
    ) -> B2FileEntry | None:
        """Locate a file by its basename within the client's bucket.

        Legend first (O(1) dict access). On a miss, one bounded legend refresh
        and a retry — the file may have been uploaded since the last sweep.
        Only then does it fall back to the original paginated b2_list_file_names
        scan, so correctness never depends on the cache being current.
        """
        target = file_name.strip().strip("/")
        if not target:
            return None
        target_basename = posixpath.basename(target)
        target_norm = _normalize_file_match_text(target_basename)

        location = await self._resolve_client_location(bucket_name, prefix)
        if location is None:
            return None
        resolved_bucket_name, resolved_prefix, legend_applies = location

        if legend_applies:
            legend = await self._get_legend(bucket=resolved_bucket_name, prefix=resolved_prefix)
            hit = legend.lookup(target) if legend else None
            if hit is None and legend is not None:
                legend = await self.refresh_legend(
                    bucket=resolved_bucket_name,
                    prefix=resolved_prefix,
                    min_age_seconds=self._legend_min_refresh_seconds,
                )
                hit = legend.lookup(target) if legend else None
            if hit is not None:
                logger.debug("b2_legend_hit", file_name=file_name, b2_path=hit.file_name)
                return hit

        bucket_id = await self._resolve_bucket_id(resolved_bucket_name)
        if not bucket_id:
            return None

        # Backblaze prefix is exact-prefix match. We can't search globally for a
        # basename, so we scan everything under the client's prefix.
        cursor: str | None = None
        for _ in range(_FIND_FILE_MAX_PAGES):
            entries, cursor = await self.list_file_names(
                bucket_id=bucket_id,
                prefix=resolved_prefix or None,
                start_file_name=cursor,
                max_file_count=_FIND_FILE_PAGE_SIZE,
            )
            for entry in entries:
                if entry.action != "upload":
                    continue
                entry_basename = posixpath.basename(entry.file_name)
                # Match either the full path or just the basename.
                if (
                    entry.file_name == target
                    or entry_basename == target_basename
                    or (target_norm and _normalize_file_match_text(entry_basename) == target_norm)
                ):
                    return entry
            if not cursor:
                break

        logger.warning(
            "b2_find_file_not_found",
            file_name=file_name,
            bucket=resolved_bucket_name,
            prefix=resolved_prefix,
        )
        return None

    # ------------------------------------------------------------------
    # Download — primary entry point used by fetch_source node
    # ------------------------------------------------------------------

    async def _resolve_source_path(self, source_video_id: str) -> tuple[str, str] | None:
        """Resolve a source_video_id to its (bucket_name, full_path) in B2.

        Looks up b2_path/source_file from video_summaries, joins the client's
        configured bucket + prefix, and falls back to a basename search when
        only source_file is known. Returns None if nothing resolves.
        """
        # 1. Look up b2_path + source_file from video_summaries.
        vs_resp = (
            await self._supabase.table("video_summaries")
            .select("b2_path, source_file")
            .eq("client_id", str(self._client_id))
            .eq("source_video_id", source_video_id)
            .maybe_single()
            .execute()
        )
        vs_row = as_dict(vs_resp.data if vs_resp else None)
        if vs_row is None:
            logger.warning("b2_fetch_no_summary_row", source_video_id=source_video_id)
            return None
        b2_path = vs_row.get("b2_path")
        source_file = vs_row.get("source_file")

        # 2. Resolve client bucket + prefix.
        storage = await self._get_client_storage()
        if storage is None:
            return None
        bucket = storage["bucket"]
        prefix = storage.get("prefix") or ""

        # 3. Decide the full path.
        full_path: str | None = None
        if b2_path:
            # b2_path in the DB is relative to b2_prefix.
            full_path = f"{prefix}/{b2_path}".lstrip("/") if prefix else str(b2_path)
        elif source_file:
            # Fallback: search the bucket for the filename.
            logger.info(
                "b2_fetch_falling_back_to_search",
                source_video_id=source_video_id,
                source_file=source_file,
            )
            entry = await self.find_file_by_name(
                file_name=str(source_file),
                bucket_name=bucket,
                prefix=prefix,
            )
            if entry:
                # entry.file_name is already the full path within the bucket.
                full_path = entry.file_name

        if not full_path:
            logger.warning(
                "b2_fetch_no_path_resolved",
                source_video_id=source_video_id,
                had_b2_path=bool(b2_path),
                had_source_file=bool(source_file),
            )
            return None

        return bucket, full_path

    async def _download_file(
        self,
        http: httpx.AsyncClient,
        *,
        bucket: str,
        full_path: str,
    ) -> httpx.Response:
        """GET one file, re-authorizing and retrying once on an expired token.

        The path is percent-encoded the same way `get_download_url` does it —
        without this, any filename containing a space or a non-ASCII character
        produces a malformed URL.
        """
        for attempt in (0, 1):
            auth = await self._authorize()
            url = f"{auth['downloadUrl']}/file/{bucket}/{quote(full_path, safe='/')}"
            resp = await http.get(url, headers={"Authorization": auth["authorizationToken"]})
            if attempt == 0 and _is_retryable_auth_failure(resp):
                logger.info("b2_auth_token_refreshed", path=full_path)
                self._auth = None
                continue
            resp.raise_for_status()
            return resp
        raise AssertionError("unreachable")  # pragma: no cover

    async def _download_text_file(
        self,
        *,
        bucket: str,
        full_path: str,
        source_label: str,
    ) -> str | None:
        if is_media_file(full_path):
            logger.info("b2_skipped_media_download", source=source_label, path=full_path)
            return None
        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await self._download_file(http, bucket=bucket, full_path=full_path)

        content_type = resp.headers.get("content-type", "").split(";", 1)[0].lower()
        if not _is_text_file(full_path, content_type):
            logger.info(
                "b2_file_skipped_non_text",
                source=source_label,
                path=full_path,
                content_type=content_type,
                bytes=len(resp.content),
            )
            return None

        logger.info(
            "b2_text_file_downloaded",
            source=source_label,
            path=full_path,
            bytes=len(resp.content),
        )
        return resp.text

    # ------------------------------------------------------------------
    # Concurrent multi-file fetch — "dive deeper on clip X" pulls a source
    # plus its transcript sidecars in one bounded batch.
    # ------------------------------------------------------------------

    async def fetch_files(
        self,
        *,
        paths: list[str],
        bucket_name: str | None = None,
    ) -> list[B2FetchedFile]:
        """Download several files concurrently, one result per requested path.

        Shares a single `httpx.AsyncClient` — and therefore one connection pool
        and one B2 auth token, which B2 documents as safe across concurrent
        downloads — for the whole batch, and bounds fan-out with a semaphore.
        `return_exceptions=True`: a missing subtitle sidecar must not take down
        the transcript that was fetched fine alongside it.
        """
        wanted: list[str] = []
        for path in paths:
            cleaned = path.strip().strip("/")
            if cleaned and cleaned not in wanted:
                wanted.append(cleaned)
        if not wanted:
            return []

        if bucket_name:
            bucket = bucket_name
        else:
            storage = await self._get_client_storage()
            if storage is None:
                return []
            bucket = storage["bucket"]

        semaphore = asyncio.Semaphore(self._fetch_concurrency)
        limit = httpx.Limits(
            max_connections=self._fetch_concurrency,
            max_keepalive_connections=self._fetch_concurrency,
        )

        async def _fetch_one(http: httpx.AsyncClient, full_path: str) -> B2FetchedFile:
            # Content-type is only readable after the whole body has arrived, so
            # a video would be downloaded in full and then discarded. Its name is
            # enough to rule it out first.
            if is_media_file(full_path):
                logger.info("b2_skipped_media_download", path=full_path)
                return B2FetchedFile(file_name=full_path, is_text=False)
            async with semaphore:
                resp = await self._download_file(http, bucket=bucket, full_path=full_path)
            content_type = resp.headers.get("content-type", "").split(";", 1)[0].lower()
            if not _is_text_file(full_path, content_type):
                return B2FetchedFile(file_name=full_path, is_text=False)
            return B2FetchedFile(file_name=full_path, content=resp.text, is_text=True)

        # Scoped to the batch rather than to the tool instance: nothing in the
        # request path disposes tools, and a turn runs in a detached task that
        # outlives its HTTP request — an instance-level client would either leak
        # or be closed underneath a running fetch.
        async with httpx.AsyncClient(timeout=60.0, limits=limit) as http:
            outcomes = await asyncio.gather(
                *(_fetch_one(http, p) for p in wanted),
                return_exceptions=True,
            )

        results: list[B2FetchedFile] = []
        for full_path, outcome in zip(wanted, outcomes, strict=True):
            if isinstance(outcome, BaseException):
                logger.warning("b2_batch_fetch_failed", path=full_path, error=str(outcome))
                results.append(B2FetchedFile(file_name=full_path, error=str(outcome)))
            else:
                results.append(outcome)
        logger.info(
            "b2_batch_fetch_completed",
            requested=len(wanted),
            with_text=sum(1 for r in results if r.content),
            failed=sum(1 for r in results if r.error),
        )
        return results

    async def fetch_source_bundle(self, source_video_id: str) -> list[B2FetchedFile]:
        """The readable text belonging to a source, fetched concurrently.

        A source resolves to its *video* far more often than to a transcript, and
        a video carries no ingestible text — so this deliberately fetches the
        transcript/summary sidecars the legend pairs with that stem, and the
        resolved file itself only when it is already text. Downloading the video
        to discover it is not text costs tens of seconds and yields nothing.
        """
        resolved = await self._resolve_source_path(source_video_id)
        if resolved is None:
            return []
        bucket, full_path = resolved

        paths: list[str] = [full_path] if not is_media_file(full_path) else []
        location = await self._resolve_client_location(bucket, None)
        if location is not None and location[2]:
            legend = await self._get_legend(bucket=location[0], prefix=location[1])
            if legend is not None:
                paths.extend(e.file_name for e in legend.text_siblings(full_path))

        if not paths:
            logger.info(
                "b2_no_text_for_source",
                source_video_id=source_video_id,
                resolved_path=full_path,
            )
            return []

        paths.sort(key=sidecar_sort_rank)
        return await self.fetch_files(paths=paths, bucket_name=bucket)

    async def fetch_source_file(self, source_video_id: str) -> str | None:
        resolved = await self._resolve_source_path(source_video_id)
        if resolved is None:
            return None
        bucket, full_path = resolved

        return await self._download_text_file(
            bucket=bucket,
            full_path=full_path,
            source_label=source_video_id,
        )

    async def fetch_file_by_name(
        self,
        *,
        file_name: str,
        bucket_name: str | None = None,
        prefix: str | None = None,
    ) -> str | None:
        storage = await self._get_client_storage()
        if bucket_name:
            resolved_bucket_name = bucket_name
        elif storage is not None:
            resolved_bucket_name = storage["bucket"]
        else:
            return None
        resolved_prefix = prefix if prefix is not None else (storage or {}).get("prefix", "")

        entry = await self.find_file_by_name(
            file_name=file_name,
            bucket_name=resolved_bucket_name,
            prefix=resolved_prefix,
        )
        if entry is None:
            return None

        return await self._download_text_file(
            bucket=resolved_bucket_name,
            full_path=entry.file_name,
            source_label=file_name,
        )

    # ------------------------------------------------------------------
    # Signed download URL — lets a third party (e.g. OpusClip) fetch a
    # private file directly without proxying the bytes through us.
    # ------------------------------------------------------------------

    async def get_download_url(
        self, source_video_id: str, *, valid_duration_seconds: int = 86400
    ) -> str | None:
        """Build a time-limited, self-contained download URL for a source video.

        Calls b2_get_download_authorization to mint a token scoped to the exact
        file path, then returns a URL with the token embedded as the
        ``Authorization`` query parameter. The URL needs no extra headers, so an
        external service can fetch the private file directly from B2.

        `valid_duration_seconds` must be in [1, 604800] (B2's 7-day cap).
        Returns None if the source_video_id can't be resolved to a *video* file.
        """
        resolved = await self._resolve_source_path(source_video_id)
        if resolved is None:
            return None
        bucket_name, full_path = resolved
        return await self._sign_resolved_path(
            bucket_name,
            full_path,
            valid_duration_seconds=valid_duration_seconds,
            source_label=source_video_id,
        )

    async def get_path_download_urls(
        self,
        b2_paths: list[str],
        *,
        valid_duration_seconds: int = 86400,
    ) -> dict[str, str]:
        """Sign tenant-relative B2 paths in a bounded concurrent batch.

        Used by the media-library list endpoint for private thumbnails. Paths
        are resolved beneath the authenticated client's configured b2_prefix;
        absolute/traversal paths are rejected.
        """
        clean_paths: list[str] = []
        for raw_path in dict.fromkeys(b2_paths):
            path = str(raw_path or "").strip()
            if path.startswith("/") or "\\" in path:
                continue
            normalized = posixpath.normpath(path)
            if not path or normalized in {".", ".."} or normalized.startswith("../"):
                continue
            clean_paths.append(normalized)
        if not clean_paths:
            return {}

        storage = await self._get_client_storage()
        if storage is None:
            return {}
        bucket_name = storage["bucket"]
        prefix = storage.get("prefix") or ""

        # Warm the shared authorization and bucket-id caches before concurrent
        # signing so a page of cards does not repeat account authorization.
        await self._authorize()
        if not await self._resolve_bucket_id(bucket_name):
            return {}

        semaphore = asyncio.Semaphore(8)

        async def sign(relative_path: str) -> tuple[str, str | None]:
            full_path = f"{prefix}/{relative_path}".lstrip("/") if prefix else relative_path
            async with semaphore:
                url = await self._sign_resolved_path(
                    bucket_name,
                    full_path,
                    valid_duration_seconds=valid_duration_seconds,
                    source_label=relative_path,
                )
            return relative_path, url

        signed = await asyncio.gather(*(sign(path) for path in clean_paths))
        return {path: url for path, url in signed if url}

    async def _sign_resolved_path(
        self,
        bucket_name: str,
        full_path: str,
        *,
        valid_duration_seconds: int,
        source_label: str,
    ) -> str | None:
        """Mint a download token for an already tenant-resolved bucket/path."""

        # Resolution is deliberately permissive so a source whose video was never
        # uploaded can still be *read* via its transcript. Publishing is the one
        # caller that must not benefit from that: this URL goes to OpusClip, which
        # would be asked to cut a video out of a .txt. Fail the gate cleanly, the
        # way an unresolvable source always has.
        if not is_media_file(full_path):
            logger.warning(
                "b2_signed_url_not_media",
                source_video_id=source_video_id,
                resolved_path=full_path,
            )
            return None

        bucket_id = await self._resolve_bucket_id(bucket_name)
        if not bucket_id:
            logger.warning(
                "b2_signed_url_no_bucket_id",
                source=source_label,
                bucket=bucket_name,
            )
            return None

        auth = await self._authorize()
        url = f"{auth['apiUrl']}/b2api/v3/b2_get_download_authorization"
        body = {
            "bucketId": bucket_id,
            "fileNamePrefix": full_path,
            "validDurationInSeconds": valid_duration_seconds,
        }
        async with httpx.AsyncClient(timeout=15.0) as http:
            resp = await http.post(url, json=body, headers=await self._headers())
            resp.raise_for_status()
            token = str(resp.json()["authorizationToken"])

        encoded_path = quote(full_path, safe="/")
        signed_url = (
            f"{auth['downloadUrl']}/file/{bucket_name}/{encoded_path}"
            f"?Authorization={quote(token, safe='')}"
        )
        logger.info(
            "b2_signed_url_created",
            source=source_label,
            path=full_path,
            ttl=valid_duration_seconds,
        )
        return signed_url


def _parse_file_entry(row: dict[str, Any]) -> B2FileEntry:
    """Map a B2 file JSON row to our B2FileEntry dataclass."""
    action = str(row.get("action") or "upload")
    raw_size = row.get("contentLength")
    if raw_size is None:
        raw_size = row.get("size") or 0
    return B2FileEntry(
        file_name=str(row.get("fileName") or ""),
        file_id=str(row["fileId"]) if row.get("fileId") else None,
        action=action,
        content_length=int(raw_size or 0),
        content_type=str(row["contentType"]) if row.get("contentType") else None,
        upload_timestamp=int(row["uploadTimestamp"])
        if row.get("uploadTimestamp") is not None
        else None,
        bucket_id=str(row["bucketId"]) if row.get("bucketId") else None,
        metadata={
            k: row[k]
            for k in ("fileInfo", "contentSha1", "contentMd5", "fileRetention", "legalHold")
            if k in row and row[k] is not None
        },
    )


def _is_retryable_auth_failure(resp: httpx.Response) -> bool:
    """True for a 401 that a fresh b2_authorize_account would fix.

    B2 returns 401 for `missing_auth_token` (a client bug), `bad_auth_token` and
    `expired_auth_token`; only the last two are worth retrying.
    """
    if resp.status_code != 401:
        return False
    try:
        code = str((resp.json() or {}).get("code") or "")
    except ValueError:
        return False
    return code in _RETRYABLE_AUTH_CODES
