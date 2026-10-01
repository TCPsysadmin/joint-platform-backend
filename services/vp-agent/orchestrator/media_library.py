from __future__ import annotations

import datetime
from collections import Counter
from typing import Any, cast
from uuid import UUID

from postgrest import CountMethod
from supabase import AsyncClient

from orchestrator.supabase_json import as_dict, as_dict_list

_VIDEO_COLUMNS = (
    "summary_id, source_video_id, title, source_file, has_timestamps, "
    "duration_seconds, recorded_at, summary_text, topics, speakers, "
    "quality_score, b2_path, thumbnail_url, thumbnail_b2_path, "
    "uploaded_by_user_id, uploaded_by_email, created_at, updated_at"
)
_TRANSCRIPT_STATUS_PAGE_SIZE = 1000


async def list_videos(
    svc: AsyncClient,
    *,
    client_id: UUID,
    limit: int = 50,
    offset: int = 0,
    search: str | None = None,
    sort: str = "updated-desc",
    content_filter: str = "all",
) -> tuple[list[dict[str, Any]], int]:
    """Return one lightweight library-folder record per source video."""
    if content_filter == "all":
        query = _video_query(
            svc,
            client_id=client_id,
            search=search,
            sort=sort,
            exact_count=True,
        )
        response = await query.range(offset, offset + limit - 1).execute()
        rows = response.data if response and isinstance(response.data, list) else []
        videos = [row for row in rows if isinstance(row, dict)]
        total = response.count if response and isinstance(response.count, int) else len(videos)
    else:
        # Transcript availability lives in transcript_segments rather than
        # video_summaries. Scan lightweight database pages on the server so the
        # browser still receives only the requested page and an accurate total.
        candidates: list[dict[str, Any]] = []
        page_offset = 0
        while True:
            query = _video_query(
                svc,
                client_id=client_id,
                search=search,
                sort=sort,
                exact_count=False,
            )
            response = await query.range(
                page_offset, page_offset + _TRANSCRIPT_STATUS_PAGE_SIZE - 1
            ).execute()
            rows = response.data if response and isinstance(response.data, list) else []
            candidates.extend(row for row in rows if isinstance(row, dict))
            if len(rows) < _TRANSCRIPT_STATUS_PAGE_SIZE:
                break
            page_offset += _TRANSCRIPT_STATUS_PAGE_SIZE

        candidate_counts = await _transcript_segment_counts(
            svc,
            client_id=client_id,
            source_video_ids=[str(row.get("source_video_id") or "") for row in candidates],
        )
        filtered = [
            row
            for row in candidates
            if _matches_content_filter(
                row,
                transcript_segment_count=candidate_counts[str(row.get("source_video_id") or "")],
                content_filter=content_filter,
            )
        ]
        total = len(filtered)
        videos = filtered[offset : offset + limit]

    segment_counts = await _transcript_segment_counts(
        svc,
        client_id=client_id,
        source_video_ids=[str(row.get("source_video_id") or "") for row in videos],
    )
    return (
        [
            _folder_payload(
                row,
                transcript_segment_count=segment_counts[str(row.get("source_video_id"))],
            )
            for row in videos
        ],
        total,
    )


def _video_query(
    svc: AsyncClient,
    *,
    client_id: UUID,
    search: str | None,
    sort: str,
    exact_count: bool,
) -> Any:
    table = svc.table("video_summaries")
    query = (
        table.select(_VIDEO_COLUMNS, count=CountMethod.exact)
        if exact_count
        else table.select(_VIDEO_COLUMNS)
    )
    query = query.eq("client_id", str(client_id))
    needle = (search or "").strip()
    if needle:
        # PostgREST's `or` expression treats commas as separators. Quoted
        # patterns keep punctuation in filenames from changing the filter.
        pattern = needle.replace("\\", "\\\\").replace('"', '\\"')
        query = query.or_(
            ",".join(
                f'{column}.ilike."*{pattern}*"'
                for column in ("title", "source_file", "source_video_id", "summary_text")
            )
        )

    if sort == "name-asc":
        return query.order("title", desc=False).order("source_video_id", desc=False)
    if sort == "name-desc":
        return query.order("title", desc=True).order("source_video_id", desc=True)
    return query.order("updated_at", desc=sort != "updated-asc")


def _matches_content_filter(
    row: dict[str, Any], *, transcript_segment_count: int, content_filter: str
) -> bool:
    has_transcript = transcript_segment_count > 0
    has_summary = bool(str(row.get("summary_text") or "").strip())
    if content_filter == "complete":
        return has_transcript and has_summary
    if content_filter == "missing-transcript":
        return not has_transcript
    if content_filter == "missing-summary":
        return not has_summary
    return True


async def get_video(
    svc: AsyncClient,
    *,
    client_id: UUID,
    source_video_id: str,
) -> dict[str, Any] | None:
    """Return a video folder with its summary and ordered transcript segments."""
    video_response = (
        await svc.table("video_summaries")
        .select(_VIDEO_COLUMNS)
        .eq("client_id", str(client_id))
        .eq("source_video_id", source_video_id)
        .maybe_single()
        .execute()
    )
    video = as_dict(video_response.data if video_response else None)
    if video is None:
        return None

    transcript_response = (
        await svc.table("transcript_segments")
        .select(
            "segment_id, chunk_index, start_seconds, end_seconds, speaker, "
            "transcript_text, word_count"
        )
        .eq("client_id", str(client_id))
        .eq("source_video_id", source_video_id)
        .order("chunk_index")
        .execute()
    )
    segments = (
        transcript_response.data
        if transcript_response and isinstance(transcript_response.data, list)
        else []
    )

    payload = _folder_payload(video)
    payload["transcript"] = {
        "available": bool(segments),
        "segment_count": len(segments),
        "segments": segments,
        "text": "\n\n".join(
            str(segment.get("transcript_text") or "").strip()
            for segment in segments
            if isinstance(segment, dict) and segment.get("transcript_text")
        ),
    }
    return payload


async def register_storage(
    svc: AsyncClient,
    *,
    client_id: UUID,
    source_video_id: str,
    title: str | None,
    source_file: str | None,
    b2_path: str,
    thumbnail_b2_path: str | None,
    uploaded_by_user_id: UUID,
    uploaded_by_email: str | None = None,
) -> dict[str, Any]:
    """Idempotently attach archived video objects to one tenant's media row."""
    source_id = source_video_id.strip()
    video_path = _object_path(b2_path, field="b2_path")
    thumbnail_path = (
        _object_path(thumbnail_b2_path, field="thumbnail_b2_path") if thumbnail_b2_path else None
    )
    if not source_id:
        raise ValueError("source_video_id must not be empty")

    payload: dict[str, Any] = {
        "client_id": str(client_id),
        "source_video_id": source_id,
        "b2_path": video_path,
        "uploaded_by_user_id": str(uploaded_by_user_id),
        "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    if uploaded_by_email:
        payload["uploaded_by_email"] = uploaded_by_email
    if title and title.strip():
        payload["title"] = title.strip()
    if source_file and source_file.strip():
        payload["source_file"] = source_file.strip()
    if thumbnail_path:
        payload["thumbnail_b2_path"] = thumbnail_path

    response = (
        await svc.table("video_summaries")
        .upsert(payload, on_conflict="client_id,source_video_id")
        .execute()
    )
    rows = response.data if response and isinstance(response.data, list) else []
    row = next((candidate for candidate in rows if isinstance(candidate, dict)), payload)
    return _folder_payload(row)


async def get_media_storage_records(
    svc: AsyncClient,
    *,
    client_id: UUID,
    source_video_ids: list[str],
) -> list[dict[str, Any]]:
    """Return selected storage metadata in request order, scoped to one tenant."""
    response = (
        await svc.table("video_summaries")
        .select("source_video_id,b2_path,thumbnail_b2_path")
        .eq("client_id", str(client_id))
        .in_("source_video_id", source_video_ids)
        .execute()
    )
    rows = as_dict_list(response.data if response is not None else None)
    manifest_response = (
        await svc.table("ingestion_manifests")
        .select("source_video_id,transcript_drive_file_id,summary_drive_file_id")
        .eq("client_id", str(client_id))
        .in_("source_video_id", source_video_ids)
        .execute()
    )
    manifests = as_dict_list(manifest_response.data if manifest_response is not None else None)
    manifest_by_id = {str(row.get("source_video_id") or ""): row for row in manifests}
    by_id: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_id = str(row.get("source_video_id") or "")
        by_id[source_id] = {**row, **manifest_by_id.get(source_id, {})}
    return [by_id[source_id] for source_id in source_video_ids if source_id in by_id]


async def delete_media_records(
    svc: AsyncClient,
    *,
    client_id: UUID,
    source_video_ids: list[str],
) -> int:
    """Atomically remove selected media, manifests, and transcript segments."""
    response = await svc.rpc(
        "admin_delete_media",
        {
            "p_client_id": str(client_id),
            "p_source_video_ids": source_video_ids,
        },
    ).execute()
    return int(cast(Any, response.data) or 0) if response is not None else 0


async def move_media_records(
    svc: AsyncClient,
    *,
    source_client_id: UUID,
    destination_client_id: UUID,
    source_video_ids: list[str],
    destination_b2_bucket: str,
) -> int:
    """Atomically transfer selected media and indexed text to another tenant."""
    response = await svc.rpc(
        "admin_move_media",
        {
            "p_source_client_id": str(source_client_id),
            "p_destination_client_id": str(destination_client_id),
            "p_source_video_ids": source_video_ids,
            "p_destination_b2_bucket": destination_b2_bucket,
        },
    ).execute()
    return int(cast(Any, response.data) or 0) if response is not None else 0


def _object_path(value: str, *, field: str) -> str:
    path = value.strip().lstrip("/")
    if not path:
        raise ValueError(f"{field} must not be empty")
    if ".." in path.split("/"):
        raise ValueError(f"{field} contains an invalid path segment")
    return path


async def _transcript_segment_counts(
    svc: AsyncClient,
    *,
    client_id: UUID,
    source_video_ids: list[str],
) -> Counter[str]:
    """Count transcript chunks for a page of videos without per-video queries."""
    source_ids = sorted({source_id for source_id in source_video_ids if source_id})
    counts: Counter[str] = Counter()
    if not source_ids:
        return counts

    offset = 0
    while True:
        response = (
            await svc.table("transcript_segments")
            .select("segment_id,source_video_id")
            .eq("client_id", str(client_id))
            .in_("source_video_id", source_ids)
            .order("segment_id")
            .range(offset, offset + _TRANSCRIPT_STATUS_PAGE_SIZE - 1)
            .execute()
        )
        rows = response.data if response and isinstance(response.data, list) else []
        for row in rows:
            if isinstance(row, dict) and row.get("source_video_id"):
                counts[str(row["source_video_id"])] += 1
        if len(rows) < _TRANSCRIPT_STATUS_PAGE_SIZE:
            break
        offset += _TRANSCRIPT_STATUS_PAGE_SIZE
    return counts


def _folder_payload(
    row: dict[str, Any], *, transcript_segment_count: int | None = None
) -> dict[str, Any]:
    source_video_id = str(row.get("source_video_id") or "")
    payload = {
        "id": source_video_id,
        "kind": "video_folder",
        "name": row.get("title") or row.get("source_file") or source_video_id,
        "thumbnail": {
            "url": row.get("thumbnail_url"),
            "b2_path": row.get("thumbnail_b2_path"),
        },
        "video": {
            "source_video_id": source_video_id,
            "source_file": row.get("source_file"),
            "b2_path": row.get("b2_path"),
            "duration_seconds": row.get("duration_seconds"),
            "recorded_at": row.get("recorded_at"),
            "has_timestamps": row.get("has_timestamps"),
        },
        "summary": {
            "text": row.get("summary_text"),
            "topics": row.get("topics") or [],
            "speakers": row.get("speakers") or [],
            "quality_score": row.get("quality_score"),
        },
        "uploaded_by": {
            "user_id": row.get("uploaded_by_user_id"),
            "email": row.get("uploaded_by_email"),
        },
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }
    if transcript_segment_count is not None:
        payload["transcript"] = {
            "available": transcript_segment_count > 0,
            "segment_count": transcript_segment_count,
        }
    return payload
