from __future__ import annotations

import posixpath
from typing import Any
from uuid import UUID

import httpx
import structlog
from supabase import AsyncClient

from orchestrator.supabase_json import as_dict
from orchestrator.tools.protocols import B2Bucket, B2FileEntry, B2KeyInfo

logger = structlog.get_logger(__name__)

_B2_AUTH_URL = "https://api.backblazeb2.com/b2api/v3/b2_authorize_account"

# Cap pagination during filename-based search to avoid runaway scans of
# extremely large buckets. Most clients store far fewer than this.
_FIND_FILE_PAGE_SIZE = 1000
_FIND_FILE_MAX_PAGES = 10


class B2FileTool:
    """B2 client for downloading source transcripts and searching files.

    Auth (`authorizationToken`, `apiUrl`, `downloadUrl`, `accountId`) is fetched
    once on first use and cached for the lifetime of this instance — which is
    one request, since RuntimeContext is per-request.

    Bucket id resolution is also cached so listing operations after the first
    one don't re-hit b2_list_buckets.
    """

    def __init__(
        self,
        supabase: AsyncClient,
        client_id: UUID,
        key_id: str,
        application_key: str,
    ) -> None:
        self._supabase = supabase
        self._client_id = client_id
        self._key_id = key_id
        self._application_key = application_key
        self._auth: dict[str, str] | None = None
        self._bucket_id_by_name: dict[str, str] = {}
        self._client_storage: dict[str, str] | None = None  # {"bucket": ..., "prefix": ...}

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

        Paginates through b2_list_file_names under `prefix` (defaults to the
        client's configured prefix) and returns the first entry whose basename
        matches `file_name`. Returns None if not found within the page cap.
        """
        target = file_name.strip().strip("/")
        if not target:
            return None
        target_basename = posixpath.basename(target)

        # Resolve bucket name → bucket id.
        resolved_bucket_name = bucket_name
        resolved_prefix = prefix
        if not resolved_bucket_name or resolved_prefix is None:
            storage = await self._get_client_storage()
            if storage is None:
                return None
            resolved_bucket_name = resolved_bucket_name or storage["bucket"]
            if resolved_prefix is None:
                resolved_prefix = storage.get("prefix") or ""

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
                # Match either the full path or just the basename.
                if entry.file_name == target or posixpath.basename(entry.file_name) == target_basename:
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

    async def fetch_source_file(self, source_video_id: str) -> str | None:
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

        # 3. Decide the full path to download.
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

        # 4. Authorize and download.
        auth = await self._authorize()
        url = f"{auth['downloadUrl']}/file/{bucket}/{full_path}"

        async with httpx.AsyncClient(timeout=60.0) as http:
            resp = await http.get(
                url,
                headers={"Authorization": auth["authorizationToken"]},
            )
            resp.raise_for_status()

        logger.info(
            "b2_file_downloaded",
            source_video_id=source_video_id,
            path=full_path,
            bytes=len(resp.content),
            via_search=not bool(b2_path),
        )
        return resp.text


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
        upload_timestamp=int(row["uploadTimestamp"]) if row.get("uploadTimestamp") is not None else None,
        bucket_id=str(row["bucketId"]) if row.get("bucketId") else None,
        metadata={
            k: row[k]
            for k in ("fileInfo", "contentSha1", "contentMd5", "fileRetention", "legalHold")
            if k in row and row[k] is not None
        },
    )
