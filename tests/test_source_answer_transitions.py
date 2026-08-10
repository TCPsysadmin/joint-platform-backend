"""A session can leave source-answer mode for a clip follow-up and come back.

`route_intent` clears `source_task` when a follow-up points at a clip already on
screen (see `tests/test_dive_deeper_ingest.py`), so "dive deeper on clip 2" isn't
misread as another source summary. This file covers the rest of that story: the
`source_answer` node itself degrading gracefully when the usual resolved-source
fields are missing, and a session actually transitioning back into source-answer
mode once the creator asks a source question again.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from orchestrator.nodes import route_intent, source_answer
from orchestrator.runtime import RuntimeContext

# ── source_answer node: missing provenance fields ────────────────────────────


@pytest.mark.asyncio
async def test_source_answer_degrades_gracefully_with_no_metadata_or_video_id(
    make_runtime: Any,
) -> None:
    """A follow-up can land in source_answer with only transcript text in hand
    and no resolved source_video_id/source_metadata — e.g. the B2-only
    pseudo-source path, or a legacy checkpoint. The node must still answer
    rather than crashing on the missing fields."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(content="It's a day-in-the-life video about trying new things.")
    )
    runtime: RuntimeContext = make_runtime(llm)

    patch = await source_answer.run(
        {  # type: ignore[arg-type]
            "session_id": "no-metadata",
            "user_query": "what is this about",
            "source_reference": "some-file.mp4",
            "source_video_id": None,
            "source_metadata": None,
            "source_file_content": "Transcript text about trying new things every day.",
            "retrieved_segments": [],
            "session_documents": [],
        },
        {"configurable": {"runtime": runtime, "thread_id": "no-metadata"}},  # type: ignore[arg-type]
    )

    assert patch["final_recommendation"] == "It's a day-in-the-life video about trying new things."
    assert patch["awaiting_confirmation"] is False


@pytest.mark.asyncio
async def test_source_answer_with_no_content_at_all_still_replies_without_calling_the_llm(
    make_runtime: Any,
) -> None:
    """No transcript, no retrieved segments, no metadata: the node's own
    no-content guard must fire before ever reaching the LLM."""
    llm = AsyncMock()
    runtime: RuntimeContext = make_runtime(llm)

    patch = await source_answer.run(
        {  # type: ignore[arg-type]
            "session_id": "empty",
            "user_query": "what is this about",
            "source_reference": None,
            "source_video_id": None,
            "source_metadata": None,
            "source_file_content": None,
            "retrieved_segments": [],
            "session_documents": [],
        },
        {"configurable": {"runtime": runtime, "thread_id": "empty"}},  # type: ignore[arg-type]
    )

    assert "do not have enough transcript" in patch["final_recommendation"]
    assert patch["awaiting_confirmation"] is False
    llm.ainvoke.assert_not_awaited()


# ── the round trip: source_answer → clip follow-up → source_answer ──────────


@pytest.mark.asyncio
async def test_route_intent_transitions_from_source_answer_to_clip_follow_up_and_back(
    make_runtime: Any,
) -> None:
    """The full round trip: a session answering "what is this about" moves into
    a clip follow-up ("dive deeper on clip 2"), which clears source_task so it
    isn't misread as another source summary — and then moves back out when the
    creator names a source and asks about it again."""
    state: dict[str, Any] = {
        "session_id": "round-trip",
        "candidate_clips": [{"video_id": "vid-one"}, {"video_id": "vid-two"}],
        "source_task": "source_answer",
        "retrieved_segments": [{"segment_id": "seg-1", "video_id": "vid-one"}],
    }

    # Turn 1: "dive deeper on clip 2" — a clip reference must clear the
    # lingering source_answer task and force reuse.
    llm_a = AsyncMock()
    llm_a.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=json.dumps(
                {
                    "intent": "follow_up",
                    "user_query": "dive deeper on clip 2",
                    "reuse_context": True,
                }
            )
        )
    )
    runtime_a: RuntimeContext = make_runtime(llm_a)
    state["messages"] = [HumanMessage(content="dive deeper on clip 2")]
    patch_a = await route_intent.run(
        state,  # type: ignore[arg-type]
        {"configurable": {"runtime": runtime_a, "thread_id": "round-trip"}},  # type: ignore[arg-type]
    )
    assert patch_a["intent"] == "follow_up"
    assert patch_a["follow_up_reuse"] is True
    assert patch_a["source_task"] is None
    state.update(patch_a)

    # Turn 2: back to a source question about a named file — source_task is
    # restored to "source_answer" (the deterministic detection in route_intent
    # fires again because this message both names a source and asks about it).
    llm_b = AsyncMock()
    llm_b.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=json.dumps(
                {
                    "intent": "follow_up",
                    "user_query": "what is vid-two.mp4 about",
                    "source_reference": "vid-two.mp4",
                    "reuse_context": False,
                }
            )
        )
    )
    runtime_b: RuntimeContext = make_runtime(llm_b)
    state["messages"] = [HumanMessage(content="what is vid-two.mp4 about")]
    patch_b = await route_intent.run(
        state,  # type: ignore[arg-type]
        {"configurable": {"runtime": runtime_b, "thread_id": "round-trip"}},  # type: ignore[arg-type]
    )

    assert patch_b["intent"] == "follow_up"
    assert patch_b["source_task"] == "source_answer"
    assert patch_b["source_reference"] == "vid-two.mp4"
