from __future__ import annotations

from typing import Any
from uuid import UUID

from supabase import AsyncClient

from orchestrator.supabase_json import as_dict

_VIDEO_COLUMNS = (
    "summary_id, source_video_id, title, source_file, has_timestamps, "
    "duration_seconds, recorded_at, summary_text, topics, speakers, "
    "quality_score, b2_path, thumbnail_url, thumbnail_b2_path, created_at, updated_at"
)


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
    return [_folder_payload(row) for row in rows if isinstance(row, dict)]


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
        "segment_count": len(segments),
        "segments": segments,
        "text": "\n\n".join(
            str(segment.get("transcript_text") or "").strip()
            for segment in segments
            if isinstance(segment, dict) and segment.get("transcript_text")
        ),
    }
    return payload


def _folder_payload(row: dict[str, Any]) -> dict[str, Any]:
    source_video_id = str(row.get("source_video_id") or "")
    return {
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
