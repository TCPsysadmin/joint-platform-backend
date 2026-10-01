from __future__ import annotations

from typing import Any

# Max transcript chars in SSE previews (full text stays in graph state / checkpointer).
_TEXT_PREVIEW_CHARS = 200


def _segment_preview(seg: dict[str, Any]) -> dict[str, Any]:
    text = str(seg.get("text") or "")
    preview = text if len(text) <= _TEXT_PREVIEW_CHARS else text[:_TEXT_PREVIEW_CHARS] + "…"
    return {
        "segment_id": seg.get("segment_id"),
        "video_id": seg.get("video_id"),
        "start_seconds": seg.get("start_seconds"),
        "end_seconds": seg.get("end_seconds"),
        "score": seg.get("score"),
        "text_preview": preview,
    }


def compact_partial_state(_node_name: str, node_update: dict[str, Any]) -> dict[str, Any]:
    """Shrink node updates before SSE so clients don't receive megabyte payloads."""
    out: dict[str, Any] = {}
    for key, value in node_update.items():
        if key == "messages":
            continue
        if key == "retrieved_segments" and isinstance(value, list):
            segments = [s for s in value if isinstance(s, dict)]
            out["retrieved_segments_summary"] = {
                "hit_count": len(segments),
                "top_segments": [_segment_preview(s) for s in segments[:5]],
            }
            continue
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            out[key] = value
    return out
