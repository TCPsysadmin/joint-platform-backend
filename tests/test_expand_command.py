from __future__ import annotations

import pytest

from orchestrator.nodes import expand_clip


def test_timestamp_conversion_accepts_numbers_and_rejects_unknown_values() -> None:
    assert expand_clip._as_float("12.5", 0.0) == 12.5
    assert expand_clip._as_float(object(), 7.0) == 7.0


@pytest.mark.asyncio
async def test_expand_clip_extends_both_sides_and_clamps_start() -> None:
    state = {
        "session_id": "session-expand",
        "user_query": "Expand the previously recommended clip by 15 seconds before and after.",
        "expand_seconds": 15,
        "candidate_recommendation": {
            "video_id": "video-1",
            "hook_quote": "A useful quote",
            "start_seconds": 10.0,
            "end_seconds": 30.0,
            "has_timestamps": True,
            "rationale": "Strong opening.",
        },
        "critique_result": {"weighted_score": 0.9, "verdict": "approved"},
    }

    result = await expand_clip.run(state, {})  # type: ignore[arg-type]
    expanded = result["candidate_recommendation"]

    assert expanded["start_seconds"] == 0.0
    assert expanded["end_seconds"] == 45.0
    assert result["candidate_clips"] == [expanded]
    assert "Expanded by 15 seconds" in expanded["rationale"]
