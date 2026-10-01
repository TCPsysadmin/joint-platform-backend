from orchestrator.sse_sanitize import compact_partial_state


def test_retrieved_segments_replaced_with_summary() -> None:
    long_text = "x" * 500
    update = {
        "retrieved_segments": [
            {
                "segment_id": "a",
                "video_id": "v1",
                "text": long_text,
                "start_seconds": 0.0,
                "end_seconds": 1.0,
                "score": 0.9,
            }
        ],
        "previous_segment_ids": ["a"],
    }
    compact = compact_partial_state("retrieve", update)
    assert "retrieved_segments" not in compact
    summary = compact["retrieved_segments_summary"]
    assert summary["hit_count"] == 1
    assert len(summary["top_segments"][0]["text_preview"]) <= 201
    assert compact["previous_segment_ids"] == ["a"]
