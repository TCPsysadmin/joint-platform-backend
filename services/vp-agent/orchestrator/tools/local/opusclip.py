"""OpusClip REST API client.

Mirrors the bundled `scripts/opusclip` bash CLI from opus-skills, but as a
typed async Python tool. Implements PublishTool via `create_and_post_clip`
plus a wide surface of additional methods covering projects, clips,
templates, uploads, collections, censor jobs, and social posting.

Auth: every request sends `Authorization: Bearer {api_key}`.
Base URL: https://api.opus.pro/api (overridable).
Rate limits: 30 req/min global; see api-reference.md for per-endpoint caps.
The client does not enforce client-side throttling — callers should handle
HTTP 429 responses if they batch operations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import httpx
import structlog

from orchestrator.tools.protocols import (
    ClipPayload,
    OpusBrandTemplate,
    OpusClip,
    OpusCollection,
    OpusJob,
    OpusProject,
    OpusSocialAccount,
    OpusUploadLink,
    PublishResult,
)

logger = structlog.get_logger(__name__)

_DEFAULT_BASE_URL = "https://api.opus.pro/api"
_DEFAULT_TIMEOUT = 30.0
_UPLOAD_TIMEOUT = 600.0  # local-file uploads can be slow


class OpusClipError(RuntimeError):
    """Raised on non-2xx responses from the OpusClip API."""


class OpusClipTool:
    """Async client for the OpusClip REST API.

    Mirrors the `scripts/opusclip` bash CLI surface — projects, clips,
    templates, uploads, collections, censor jobs, and social posting.

    Also implements the existing `PublishTool` protocol via
    `create_and_post_clip`, so it can drop into RuntimeContext as
    `publish_tool` and the post_stub node continues to work.
    """

    def __init__(
        self,
        api_key: str,
        base_url: str = _DEFAULT_BASE_URL,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        if not api_key:
            raise ValueError(
                "OpusClipTool requires an API key. "
                "Set OPUSCLIP_API_KEY (Enterprise/Pro plan required)."
            )
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._extra_headers = dict(extra_headers or {})

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _headers(self, *, json_body: bool = True) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }
        if json_body:
            headers["Content-Type"] = "application/json"
        headers.update(self._extra_headers)
        return headers

    async def _request(
        self,
        method: str,
        path: str,
        *,
        json: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        expect_json: bool = True,
    ) -> Any:
        url = path if path.startswith("http") else f"{self._base_url}{path}"
        async with httpx.AsyncClient(timeout=timeout) as http:
            resp = await http.request(
                method,
                url,
                json=json,
                params=params,
                headers=self._headers(json_body=json is not None),
            )
        if resp.status_code >= 400:
            body = resp.text
            logger.warning(
                "opusclip_http_error",
                method=method,
                url=url,
                status=resp.status_code,
                body=body[:500],
            )
            raise OpusClipError(f"{method} {url} → {resp.status_code}: {body[:500]}")
        if not expect_json or not resp.content:
            return None
        return resp.json()

    async def _request_dict(self, method: str, path: str, **kwargs: Any) -> dict[str, Any]:
        """`_request` for endpoints whose raw response body we hand straight back.

        Normalises an empty/non-object body to `{}` so callers always get a dict.
        """
        data = await self._request(method, path, **kwargs)
        return data if isinstance(data, dict) else {}

    # ------------------------------------------------------------------
    # Projects (POST /clip-projects, POST /clip-projects/{id}/update-visibility)
    # ------------------------------------------------------------------

    async def create_project(
        self,
        *,
        video_url: str,
        clip_durations: list[int] | None = None,
        model: str | None = None,
        custom_prompt: str | None = None,
        topic_keywords: list[str] | None = None,
        genre: str | None = None,
        aspect_ratio: str = "portrait",
        skip_curate: bool = False,
        remove_filler_words: bool = False,
        brand_template_id: str | None = None,
        source_lang: str | None = None,
        title: str | None = None,
        range_start_sec: float | None = None,
        range_end_sec: float | None = None,
        webhook_url: str | None = None,
    ) -> OpusProject:
        """Submit a video URL for clipping.

        `clip_durations` are converted to `[[0, N], ...]` ranges as the API
        requires. The API rejects projects without `curationPref.clipDurations`.
        """
        payload: dict[str, Any] = {"videoUrl": video_url}
        if brand_template_id:
            payload["brandTemplateId"] = brand_template_id
        if title:
            payload["uploadedVideoAttr"] = {"title": title}

        curation: dict[str, Any] = {}
        if model:
            curation["model"] = model
        if genre:
            curation["genre"] = genre
        if topic_keywords:
            curation["topicKeywords"] = list(topic_keywords)
        if custom_prompt:
            curation["customPrompt"] = custom_prompt
        if clip_durations:
            curation["clipDurations"] = [[0, int(d)] for d in clip_durations]
        if skip_curate:
            curation["skipCurate"] = True
        if range_start_sec is not None or range_end_sec is not None:
            rng: dict[str, float] = {}
            if range_start_sec is not None:
                rng["startSec"] = float(range_start_sec)
            if range_end_sec is not None:
                rng["endSec"] = float(range_end_sec)
            curation["range"] = rng
        if curation:
            payload["curationPref"] = curation

        render: dict[str, Any] = {"layoutAspectRatio": aspect_ratio}
        if remove_filler_words:
            render["quickstartConfig"] = {"enableRemoveFillerWords": True}
        payload["renderPref"] = render

        if source_lang:
            payload["importPreference"] = {"sourceLang": source_lang}
        if webhook_url:
            payload["conclusionActions"] = [{"type": "WEBHOOK", "url": webhook_url}]

        data = await self._request("POST", "/clip-projects", json=payload)
        return _parse_project(data)

    async def share_project(self, project_id: str, *, visibility: str = "PUBLIC") -> dict[str, Any]:
        """Toggle a project's visibility (PUBLIC or DEFAULT)."""
        return await self._request_dict(
            "POST",
            f"/clip-projects/{project_id}/update-visibility",
            json={"visibility": visibility},
        )

    # ------------------------------------------------------------------
    # Clips (GET /exportable-clips)
    # ------------------------------------------------------------------

    async def list_clips(
        self,
        *,
        project_id: str | None = None,
        collection_id: str | None = None,
    ) -> list[OpusClip]:
        """List clips for a project OR collection. Exactly one of the two ids is required."""
        if not project_id and not collection_id:
            raise ValueError("list_clips requires project_id or collection_id")
        if project_id and collection_id:
            raise ValueError("list_clips: pass project_id OR collection_id, not both")

        params: dict[str, Any]
        if project_id:
            params = {"q": "findByProjectId", "projectId": project_id}
        else:
            params = {"q": "findByCollectionId", "collectionId": collection_id}

        data = await self._request("GET", "/exportable-clips", params=params)
        rows: list[dict[str, Any]] = (data or {}).get("data") or []
        return [_parse_clip(r) for r in rows]

    async def describe_clip(self, *, project_id: str, clip_id: str) -> OpusClip | None:
        """Find a single clip in a project by its clip id (or composite id)."""
        clips = await self.list_clips(project_id=project_id)
        suffix = clip_id.split(".", 1)[1] if "." in clip_id else clip_id
        for c in clips:
            if c.clip_id == suffix or c.clip_id == clip_id:
                return c
        return None

    # ------------------------------------------------------------------
    # Brand templates (GET /brand-templates)
    # ------------------------------------------------------------------

    async def list_brand_templates(self) -> list[OpusBrandTemplate]:
        data = await self._request("GET", "/brand-templates", params={"q": "mine"})
        rows: list[dict[str, Any]] = (data or {}).get("data") or data or []
        if isinstance(rows, dict):
            rows = rows.get("list") or []
        return [
            OpusBrandTemplate(
                template_id=str(r.get("brandTemplateId") or r.get("id") or ""),
                name=r.get("name"),
                metadata={k: v for k, v in r.items() if k not in {"brandTemplateId", "id", "name"}},
            )
            for r in rows
            if r.get("brandTemplateId") or r.get("id")
        ]

    # ------------------------------------------------------------------
    # Upload (4-step GCS resumable flow)
    # ------------------------------------------------------------------

    async def request_upload_link(self) -> OpusUploadLink:
        """Step 1 of upload — request a GCS upload URL and uploadId."""
        data = await self._request(
            "POST",
            "/upload-links",
            json={"video": {"usecase": "LocalUpload"}},
        )
        return OpusUploadLink(
            upload_id=str(data["uploadId"]),
            url=str(data["url"]),
            dns_url=data.get("dnsUrl"),
            use_amount=data.get("useAmount"),
            total_amount=data.get("totalAmount"),
        )

    async def upload_local_file(
        self,
        *,
        file_path: str | Path,
        create_project_kwargs: dict[str, Any] | None = None,
    ) -> OpusProject:
        """Upload a local file and create a clip project from it.

        Runs the full 4-step GCS resumable flow then submits a /clip-projects
        request using the resulting uploadId as the videoUrl. Extra args for
        create_project go via `create_project_kwargs`.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"upload_local_file: {path} not found")

        link = await self.request_upload_link()

        # Step 2 — initiate resumable session on the GCS URL.
        async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as http:
            init_resp = await http.post(
                link.url,
                headers={"x-goog-resumable": "start", "Content-Length": "0"},
                content=b"",
            )
        if init_resp.status_code >= 400:
            raise OpusClipError(
                f"GCS resumable init failed ({init_resp.status_code}): {init_resp.text[:200]}"
            )
        session_url = init_resp.headers.get("Location") or init_resp.headers.get("location")
        if not session_url:
            raise OpusClipError("GCS resumable init returned no Location header")

        # Step 3 — PUT the file bytes.
        size = path.stat().st_size
        with path.open("rb") as fh:
            async with httpx.AsyncClient(timeout=_UPLOAD_TIMEOUT) as http:
                put_resp = await http.put(
                    session_url,
                    content=fh.read(),
                    headers={
                        "Content-Type": "application/octet-stream",
                        "Content-Length": str(size),
                    },
                )
        if put_resp.status_code >= 400:
            raise OpusClipError(
                f"GCS upload PUT failed ({put_resp.status_code}): {put_resp.text[:200]}"
            )

        # Step 4 — create the project from the uploadId.
        kwargs = dict(create_project_kwargs or {})
        kwargs.setdefault("video_url", link.upload_id)
        logger.info("opusclip_uploaded", file=str(path), bytes=size, upload_id=link.upload_id)
        return await self.create_project(**kwargs)

    # ------------------------------------------------------------------
    # Collections
    # ------------------------------------------------------------------

    async def list_collections(self, *, content_id: str | None = None) -> list[OpusCollection]:
        params: dict[str, Any] = (
            {"q": "findByContentId", "contentId": content_id} if content_id else {"q": "mine"}
        )
        data = await self._request("GET", "/collections", params=params)
        rows: list[dict[str, Any]] = ((data or {}).get("data") or {}).get("list") or []
        return [_parse_collection(r) for r in rows]

    async def create_collection(self, *, name: str) -> OpusCollection:
        data = await self._request("POST", "/collections", json={"collectionName": name})
        return _parse_collection(((data or {}).get("data")) or {})

    async def delete_collection(self, *, collection_id: str) -> dict[str, Any]:
        return await self._request_dict("DELETE", f"/collections/{collection_id}")

    async def export_collection(self, *, collection_id: str) -> dict[str, Any]:
        return await self._request_dict("POST", f"/collections/{collection_id}/export", json={})

    async def add_clip_to_collection(
        self, *, collection_id: str, content_id: str
    ) -> dict[str, Any]:
        """content_id format: `{projectId}.{clipId}`."""
        return await self._request_dict(
            "POST",
            "/collection-contents",
            json={"collectionId": collection_id, "contentId": content_id},
        )

    async def remove_clip_from_collection(
        self, *, collection_id: str, content_id: str
    ) -> dict[str, Any]:
        return await self._request_dict(
            "POST",
            "/collection-contents/delete-collection-contents",
            json={
                "q": "findByCollectionIdAndContentId",
                "collectionId": collection_id,
                "contentId": content_id,
            },
        )

    # ------------------------------------------------------------------
    # Censor jobs
    # ------------------------------------------------------------------

    async def create_censor_job(
        self, *, project_id: str, clip_id: str, beep_sound: bool = False
    ) -> OpusJob:
        data = await self._request(
            "POST",
            "/censor-jobs",
            json={
                "projectId": project_id,
                "clipId": _strip_project_prefix(clip_id, project_id),
                "options": {"beepSound": beep_sound},
            },
        )
        return OpusJob(
            job_id=str((data or {}).get("jobId") or ""),
            status="QUEUED",
            metadata={"raw": data or {}},
        )

    async def get_censor_job(self, *, job_id: str) -> OpusJob:
        data = await self._request("GET", f"/censor-jobs/{job_id}")
        return OpusJob(
            job_id=job_id,
            status=str((data or {}).get("status") or ""),
            error=(data or {}).get("error"),
            metadata={"raw": data or {}},
        )

    # ------------------------------------------------------------------
    # Social — accounts, copy generation, publish, schedule, cancel
    # ------------------------------------------------------------------

    async def list_social_accounts(self) -> list[OpusSocialAccount]:
        data = await self._request("GET", "/social-accounts", params={"q": "mine"})
        rows: list[dict[str, Any]] = (data or {}).get("data") or []
        return [
            OpusSocialAccount(
                post_account_id=str(r.get("postAccountId") or ""),
                sub_account_id=r.get("subAccountId"),
                platform=str(r.get("platform") or ""),
                name=r.get("extUserName"),
                profile_url=r.get("extUserProfileLink"),
                metadata={
                    "ext_user_id": r.get("extUserId"),
                    "picture": r.get("extUserPictureLink"),
                },
            )
            for r in rows
        ]

    async def create_social_copy_job(
        self,
        *,
        project_id: str,
        clip_id: str,
        post_account_id: str,
        sub_account_id: str | None = None,
        prompt: str | None = None,
        force_regenerate: bool = False,
    ) -> OpusJob:
        body: dict[str, Any] = {
            "projectId": project_id,
            "clipId": _strip_project_prefix(clip_id, project_id),
            "postAccountId": post_account_id,
        }
        if sub_account_id:
            body["subAccountId"] = sub_account_id
        if prompt:
            body["prompt"] = prompt
        if force_regenerate:
            body["forceRegenerate"] = True

        data = await self._request("POST", "/social-copy-jobs", json=body)
        job_id = str(((data or {}).get("data") or {}).get("jobId") or "")
        return OpusJob(job_id=job_id, status="PENDING", metadata={"raw": data or {}})

    async def get_social_copy_job(self, *, job_id: str) -> OpusJob:
        data = await self._request("GET", f"/social-copy-jobs/{job_id}")
        body = (data or {}).get("data") if isinstance(data, dict) else None
        body = body or data or {}
        return OpusJob(
            job_id=job_id,
            status=str(body.get("status") or ""),
            result=body if body.get("status") == "COMPLETED" else None,
            error=body.get("error"),
            metadata={"raw": data or {}},
        )

    async def publish_post(
        self,
        *,
        project_id: str,
        clip_id: str,
        post_account_id: str,
        title: str,
        description: str | None = None,
        privacy: str | None = None,
        media_type: str | None = None,
        sub_account_id: str | None = None,
    ) -> dict[str, Any]:
        post_detail: dict[str, Any] = {"title": title}
        if media_type:
            post_detail["mediaType"] = media_type
        custom: dict[str, Any] = {}
        if description:
            custom["description"] = description
        if privacy:
            custom["privacy"] = privacy
        if custom:
            post_detail["custom"] = custom

        body: dict[str, Any] = {
            "projectId": project_id,
            "clipId": _strip_project_prefix(clip_id, project_id),
            "postAccountId": post_account_id,
            "postDetail": post_detail,
        }
        if sub_account_id:
            body["subAccountId"] = sub_account_id

        return await self._request_dict("POST", "/post-tasks", json=body)

    async def schedule_post(
        self,
        *,
        project_id: str,
        clip_id: str,
        post_account_id: str,
        title: str,
        publish_at: str,  # ISO 8601 UTC, e.g. "2026-03-25T14:00:00Z"
        description: str | None = None,
        privacy: str | None = None,
        media_type: str | None = None,
        sub_account_id: str | None = None,
    ) -> str:
        """Schedule a future post. Returns the scheduleId."""
        post_detail: dict[str, Any] = {"title": title}
        if media_type:
            post_detail["mediaType"] = media_type
        custom: dict[str, Any] = {}
        if description:
            custom["description"] = description
        if privacy:
            custom["privacy"] = privacy
        if custom:
            post_detail["custom"] = custom

        body: dict[str, Any] = {
            "projectId": project_id,
            "clipId": _strip_project_prefix(clip_id, project_id),
            "postAccountId": post_account_id,
            "publishAt": publish_at,
            "postDetail": post_detail,
        }
        if sub_account_id:
            body["subAccountId"] = sub_account_id

        data = await self._request("POST", "/publish-schedules", json=body)
        return str(((data or {}).get("data") or {}).get("scheduleId") or "")

    async def cancel_scheduled_post(self, *, schedule_id: str) -> None:
        await self._request("DELETE", f"/publish-schedules/{schedule_id}", expect_json=False)

    # ------------------------------------------------------------------
    # PublishTool — bridge between ClipPayload and /clip-projects
    # ------------------------------------------------------------------

    async def create_and_post_clip(self, payload: ClipPayload) -> PublishResult:
        """PublishTool protocol method.

        Submits a clip-project request to OpusClip. The video URL is taken
        from `payload.metadata['video_url']` if present, otherwise falls back
        to `payload.video_id`.

        Two modes:
        * ``payload.full_file`` → create a project from the entire source video and
          let Opus auto-curate clips across the whole file (no curation range).
        * otherwise → the agent's chosen [start_seconds, end_seconds] becomes
          OpusClip's curation ``range`` and clip duration.
        """
        meta = payload.metadata or {}
        video_url = str(meta.get("video_url") or payload.video_id or "")
        if not video_url:
            raise ValueError(
                "OpusClipTool.create_and_post_clip: no video_url in metadata or fallback video_id"
            )

        if payload.full_file:
            project = await self.create_project(
                video_url=video_url,
                clip_durations=list(payload.clip_durations) or [60],
                title=str(meta.get("title") or "") or None,
                brand_template_id=str(meta.get("brand_template_id") or "") or None,
                model=str(meta.get("model") or "") or None,
                aspect_ratio=str(meta.get("aspect_ratio") or "portrait"),
            )
        else:
            clip_length = max(int(round(payload.end_seconds - payload.start_seconds)), 1)
            project = await self.create_project(
                video_url=video_url,
                clip_durations=[clip_length],
                range_start_sec=payload.start_seconds or None,
                range_end_sec=payload.end_seconds or None,
                title=str(meta.get("title") or "") or None,
                brand_template_id=str(meta.get("brand_template_id") or "") or None,
                model=str(meta.get("model") or "") or None,
                custom_prompt=payload.hook_quote or None,
                aspect_ratio=str(meta.get("aspect_ratio") or "portrait"),
            )

        logger.info(
            "opusclip_project_created",
            project_id=project.project_id,
            video_url=video_url,
            full_file=payload.full_file,
            range=None if payload.full_file else (payload.start_seconds, payload.end_seconds),
        )

        return PublishResult(
            clip_id=project.project_id,
            status=project.status or "processing",
            url=None,
            metadata={"opusclip_project": project.metadata, "via": "opusclip"},
        )


# ──────────────────────────────────────────────────────────────────────────────
# Parsing helpers
# ──────────────────────────────────────────────────────────────────────────────


def _parse_project(data: Any) -> OpusProject:
    raw = data or {}
    # The API nests project info in slightly different shapes across endpoints;
    # try the common keys before falling back to the raw payload.
    inner = raw.get("data") if isinstance(raw, dict) else None
    if isinstance(inner, dict):
        raw = inner
    project_id = str(raw.get("projectId") or raw.get("id") or raw.get("clipProjectId") or "")
    return OpusProject(
        project_id=project_id,
        status=raw.get("status"),
        title=(raw.get("uploadedVideoAttr") or {}).get("title")
        if isinstance(raw.get("uploadedVideoAttr"), dict)
        else raw.get("title"),
        metadata=raw if isinstance(raw, dict) else {"raw": data},
    )


def _parse_clip(row: dict[str, Any]) -> OpusClip:
    raw_id = str(row.get("id") or "")
    if "." in raw_id:
        project_id, clip_id = raw_id.split(".", 1)
    else:
        project_id = ""
        clip_id = raw_id
    score = row.get("score")
    duration_ms = row.get("durationMs")
    judge = row.get("judgeResult") or {}
    return OpusClip(
        project_id=project_id,
        clip_id=clip_id,
        title=row.get("title"),
        description=row.get("description"),
        hashtags=list(row.get("hashtags") or []),
        score=float(score) if score is not None else None,
        duration_ms=int(duration_ms) if duration_ms is not None else None,
        preview_url=row.get("uriForPreview"),
        export_url=row.get("uriForExport"),
        thumbnail_url=row.get("uriForThumbnail"),
        metadata={
            "rank": row.get("rank"),
            "is_bonus": row.get("isBonusClip"),
            "judge": judge,
            "transcript": row.get("text"),
        },
    )


def _parse_collection(row: dict[str, Any]) -> OpusCollection:
    return OpusCollection(
        collection_id=str(row.get("collectionId") or ""),
        name=row.get("collectionName"),
        metadata={k: row[k] for k in ("createdAt", "updatedAt") if row.get(k) is not None},
    )


def _strip_project_prefix(clip_id: str, project_id: str) -> str:
    """OpusClip's social/edit endpoints want the bare clip suffix.

    Composite IDs come back as `{projectId}.{clipId}` from /exportable-clips.
    The bash CLI strips the prefix before sending; we do the same.
    """
    if project_id and clip_id.startswith(f"{project_id}."):
        return clip_id[len(project_id) + 1 :]
    if "." in clip_id:
        return clip_id.split(".", 1)[1]
    return clip_id
