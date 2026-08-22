from __future__ import annotations

import datetime
from collections import Counter
from typing import Any
from uuid import UUID

from supabase import AsyncClient

from orchestrator.supabase_json import as_dict

_VIDEO_COLUMNS = (
    "summary_id, source_video_id, title, source_file, has_timestamps, "
    "duration_seconds, recorded_at, summary_text, topics, speakers, "
    "quality_score, b2_path, thumbnail_url, thumbnail_b2_path, created_at, updated_at"
)
_TRANSCRIPT_STATUS_PAGE_SIZE = 1000


async def list_videos(
    svc: AsyncClient,
    *,
    client_id: UUID,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Return one lightweight library-folder record per source video."""
    response = (
        await svc.table("video_summaries")
        .select(_VIDEO_COLUMNS)
        .eq("client_id", str(client_id))
        .order("updated_at", desc=True)
        .range(offset, offset + limit - 1)
        .execute()
    )
    rows = response.data if response and isinstance(response.data, list) else []
    videos = [row for row in rows if isinstance(row, dict)]
    segment_counts = await _transcript_segment_counts(
        svc,
        client_id=client_id,
        source_video_ids=[str(row.get("source_video_id") or "") for row in videos],
    )
    return [
        _folder_payload(
            row, transcript_segment_count=segment_counts[str(row.get("source_video_id"))]
        )
        for row in videos
    ]


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
        "updated_at": datetime.datetime.now(datetime.UTC).isoformat(),
    }
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
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    }
    if transcript_segment_count is not None:
        payload["transcript"] = {
            "available": transcript_segment_count > 0,
            "segment_count": transcript_segment_count,
        }
    return payload
