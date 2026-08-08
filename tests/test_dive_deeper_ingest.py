"""A follow-up that points at a clip must be able to pull that clip's source file.

Before this, "dive deeper on clip 2" took the reuse shortcut straight to
`analyze`, so if the full transcript had never been ingested the model answered
from retrieved snippets alone — and if a *different* video's transcript happened
to be in state, it answered from the wrong file.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.context import referenced_clip_position
from orchestrator.graph import build_graph
from orchestrator.nodes import fetch_source
from orchestrator.runtime import RuntimeContext
from orchestrator.tools.protocols import B2FetchedFile

CLIPS: list[dict[str, Any]] = [
    {"video_id": "vid-one", "segment_id": "seg-1", "hook_quote": "First hook"},
    {"video_id": "vid-two", "segment_id": "seg-2", "hook_quote": "Second hook"},
]

CRITIQUE_OK = json.dumps(
    {
        "dimension_scores": [],
        "weighted_score": 0.9,
        "overall_rationale": "Strong.",
        "improvement_suggestions": [],
    }
)


def _state(**overrides: Any) -> Any:
    base: dict[str, Any] = {
        "session_id": "dive-deeper",
        "intent": "follow_up",
        "iteration_count": 0,
        "user_query": "dive deeper on clip 2",
        "candidate_clips": CLIPS,
        "retrieved_segments": [{"segment_id": "seg-1", "video_id": "vid-one"}],
        "session_documents": [],
    }
    base.update(overrides)
    return base


def _config(runtime: RuntimeContext) -> Any:
    return {"configurable": {"runtime": runtime, "thread_id": "dive-deeper"}}


# ── which clip the message points at ─────────────────────────────────────────


def test_referenced_clip_position_reads_indexes_and_ordinals() -> None:
    assert referenced_clip_position("dive deeper on clip 2") == 2
    assert referenced_clip_position("expand on option #3") == 3
    assert referenced_clip_position("why did you pick number 1") == 1
    assert referenced_clip_position("tell me more about the second clip") == 2
    assert referenced_clip_position("what about the fifth one") == 5
    # Ordinal word directly, without a following noun/"one" ("the second option").
    assert referenced_clip_position("the second one") == 2
    # A digit reference with no natural upper bound — resolving whether it is
    # in range against the actual candidate list is the caller's job, not this
    # function's; it just reads what was typed.
    assert referenced_clip_position("dive deeper on clip 10") == 10

    # A reference with no position must not guess one.
    assert referenced_clip_position("what about that one") is None
    assert referenced_clip_position("tell me more about the last clip") is None
    assert referenced_clip_position("find me something about pricing") is None
    assert referenced_clip_position("") is None


# ── fetch_source targeting ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_source_ingests_the_referenced_clips_source_not_the_top_segment(
    make_runtime: Any, mock_file_tool: AsyncMock
) -> None:
    mock_file_tool.fetch_source_bundle = AsyncMock(
        return_value=[B2FetchedFile(file_name="t.txt", content="clip two transcript", is_text=True)]
    )
    runtime: RuntimeContext = make_runtime(AsyncMock())

    patch = await fetch_source.run(_state(), _config(runtime))

    mock_file_tool.fetch_source_bundle.assert_awaited_once_with("vid-two")
    assert patch["source_file_content"] == "clip two transcript"
    assert patch["source_content_video_id"] == "vid-two"


@pytest.mark.asyncio
async def test_fetch_source_falls_back_to_the_top_segment_without_a_clip_reference(
    make_runtime: Any, mock_file_tool: AsyncMock
) -> None:
    mock_file_tool.fetch_source_bundle = AsyncMock(return_value=[])
    runtime: RuntimeContext = make_runtime(AsyncMock())

    await fetch_source.run(_state(user_query="make it punchier"), _config(runtime))

    mock_file_tool.fetch_source_bundle.assert_awaited_once_with("vid-one")


@pytest.mark.asyncio
async def test_fetch_source_falls_back_when_the_referenced_position_is_out_of_range(
    make_runtime: Any, mock_file_tool: AsyncMock
) -> None:
    """ "clip 10" against a two-clip list names a position that doesn't exist.
    That must not raise or silently ingest nothing — it degrades to the same
    top-segment default used when no clip is named at all."""
    mock_file_tool.fetch_source_bundle = AsyncMock(return_value=[])
    runtime: RuntimeContext = make_runtime(AsyncMock())

    await fetch_source.run(_state(user_query="dive deeper on clip 10"), _config(runtime))

    mock_file_tool.fetch_source_bundle.assert_awaited_once_with("vid-one")


@pytest.mark.asyncio
async def test_an_explicitly_resolved_source_still_wins(
    make_runtime: Any, mock_file_tool: AsyncMock
) -> None:
    mock_file_tool.fetch_source_bundle = AsyncMock(return_value=[])
    runtime: RuntimeContext = make_runtime(AsyncMock())

    await fetch_source.run(_state(source_video_id="vid-explicit"), _config(runtime))

    mock_file_tool.fetch_source_bundle.assert_awaited_once_with("vid-explicit")


# ── provenance: don't answer clip 2 from clip 1's transcript ─────────────────


@pytest.mark.asyncio
async def test_content_from_another_source_is_replaced(
    make_runtime: Any, mock_file_tool: AsyncMock
) -> None:
    mock_file_tool.fetch_source_bundle = AsyncMock(
        return_value=[B2FetchedFile(file_name="t.txt", content="clip two", is_text=True)]
    )
    runtime: RuntimeContext = make_runtime(AsyncMock())

    patch = await fetch_source.run(
        _state(source_file_content="clip one", source_content_video_id="vid-one"),
        _config(runtime),
    )

    mock_file_tool.fetch_source_bundle.assert_awaited_once_with("vid-two")
    assert patch["source_file_content"] == "clip two"


@pytest.mark.asyncio
async def test_matching_content_short_circuits_without_a_b2_call(
    make_runtime: Any, mock_file_tool: AsyncMock
) -> None:
    runtime: RuntimeContext = make_runtime(AsyncMock())

    patch = await fetch_source.run(
        _state(source_file_content="already here", source_content_video_id="vid-two"),
        _config(runtime),
    )

    assert patch == {}
    mock_file_tool.fetch_source_bundle.assert_not_awaited()


@pytest.mark.asyncio
async def test_content_of_unknown_provenance_is_kept(
    make_runtime: Any, mock_file_tool: AsyncMock
) -> None:
    """Checkpoints written before `source_content_video_id` existed, and the
    pseudo-transcript `retrieve` builds for a B2-only source, must not be refetched."""
    runtime: RuntimeContext = make_runtime(AsyncMock())

    patch = await fetch_source.run(
        _state(source_file_content="legacy content", source_content_video_id=None),
        _config(runtime),
    )

    assert patch == {}
    mock_file_tool.fetch_source_bundle.assert_not_awaited()


# ── combining and capping the bundle ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_sidecars_are_labelled_and_failures_are_dropped(
    make_runtime: Any, mock_file_tool: AsyncMock
) -> None:
    mock_file_tool.fetch_source_bundle = AsyncMock(
        return_value=[
            B2FetchedFile(file_name="raw/talk.mp4", is_text=False),
            B2FetchedFile(file_name="tx/talk.txt", content="the transcript", is_text=True),
            B2FetchedFile(file_name="tx/talk.srt", content="the subtitles", is_text=True),
            B2FetchedFile(file_name="tx/talk.vtt", error="503 Server Error"),
        ]
    )
    runtime: RuntimeContext = make_runtime(AsyncMock())

    patch = await fetch_source.run(_state(), _config(runtime))

    content = patch["source_file_content"]
    assert "## talk.txt" in content and "the transcript" in content
    assert "## talk.srt" in content and "the subtitles" in content
    assert "talk.vtt" not in content and "talk.mp4" not in content


@pytest.mark.asyncio
async def test_a_bundle_with_no_text_yields_no_content(
    make_runtime: Any, mock_file_tool: AsyncMock
) -> None:
    mock_file_tool.fetch_source_bundle = AsyncMock(
        return_value=[B2FetchedFile(file_name="raw/talk.mp4", is_text=False)]
    )
    runtime: RuntimeContext = make_runtime(AsyncMock())

    patch = await fetch_source.run(_state(), _config(runtime))
    assert patch == {"source_file_content": None, "source_content_video_id": None}


@pytest.mark.asyncio
async def test_oversized_transcripts_are_truncated_with_a_marker(
    make_runtime: Any, mock_file_tool: AsyncMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fetch_source.settings, "source_file_content_max_chars", 50)
    mock_file_tool.fetch_source_bundle = AsyncMock(
        return_value=[B2FetchedFile(file_name="t.txt", content="x" * 5000, is_text=True)]
    )
    runtime: RuntimeContext = make_runtime(AsyncMock())

    patch = await fetch_source.run(_state(), _config(runtime))

    content = patch["source_file_content"]
    assert content.startswith("x" * 50)
    assert content.endswith(fetch_source._TRUNCATION_MARKER)


# ── routing guards ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_a_clip_reference_clears_a_lingering_source_answer_task(
    make_runtime: Any,
) -> None:
    """The reuse path now enters at fetch_source, whose router honours source_task.
    A session that once asked "what is this about?" must not answer "dive deeper on
    clip 2" as another source summary."""
    from orchestrator.nodes import route_intent

    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
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
    runtime: RuntimeContext = make_runtime(llm)

    patch = await route_intent.run(
        {  # type: ignore[arg-type]
            "session_id": "lingering-task",
            "messages": [HumanMessage(content="dive deeper on clip 2")],
            "candidate_clips": CLIPS,
            "source_task": "source_answer",
            "retrieved_segments": [{"segment_id": "seg-1", "video_id": "vid-one"}],
        },
        {"configurable": {"runtime": runtime, "thread_id": "lingering-task"}},  # type: ignore[arg-type]
    )

    assert patch["follow_up_reuse"] is True
    assert patch["source_task"] is None


@pytest.mark.asyncio
async def test_a_reuse_follow_up_without_a_clip_reference_keeps_source_task(
    make_runtime: Any,
) -> None:
    from orchestrator.nodes import route_intent

    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
        return_value=AIMessage(
            content=json.dumps(
                {"intent": "follow_up", "user_query": "and what else", "reuse_context": True}
            )
        )
    )
    runtime: RuntimeContext = make_runtime(llm)

    patch = await route_intent.run(
        {  # type: ignore[arg-type]
            "session_id": "kept-task",
            "messages": [HumanMessage(content="and what else does it cover")],
            "candidate_clips": [],
            "source_task": "source_answer",
            "retrieved_segments": [{"segment_id": "seg-1", "video_id": "vid-one"}],
        },
        {"configurable": {"runtime": runtime, "thread_id": "kept-task"}},  # type: ignore[arg-type]
    )

    assert "source_task" not in patch, "an untouched source_task must survive the turn"


# ── graph wiring ─────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_the_reuse_shortcut_passes_through_fetch_source_but_not_retrieval(
    make_runtime: Any, mock_search_tool: AsyncMock, mock_file_tool: AsyncMock
) -> None:
    mock_file_tool.fetch_source_bundle = AsyncMock(
        return_value=[B2FetchedFile(file_name="t.txt", content="ingested now", is_text=True)]
    )
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content=r)
            for r in (
                json.dumps({"intent": "new_request", "user_query": "hook moments"}),
                json.dumps({"clips": CLIPS}),
                CRITIQUE_OK,
                "## Your Clip Options",
                json.dumps(
                    {
                        "intent": "follow_up",
                        "user_query": "dive deeper on clip 2",
                        "reuse_context": True,
                    }
                ),
                json.dumps({"clips": [CLIPS[1]]}),
                CRITIQUE_OK,
                "## Deeper on Option 2",
            )
        ]
    )
    runtime: RuntimeContext = make_runtime(llm)
    graph = build_graph(checkpointer=MemorySaver())
    session_id = "reuse-ingest"
    config: dict[str, Any] = {"configurable": {"runtime": runtime, "thread_id": session_id}}

    def _turn(message: str) -> dict[str, Any]:
        return {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "session_documents": [],
        }

    async for _ in graph.astream(_turn("Find hook clips"), config=config, stream_mode="updates"):
        pass
    await graph.aupdate_state(
        config,
        {"confirmation_response": "rejected", "awaiting_confirmation": False},
        as_node="post_stub",
    )
    async for _ in graph.astream(
        _turn("dive deeper on clip 2"), config=config, stream_mode="updates"
    ):
        pass

    state = await graph.aget_state(config)
    assert state.values["follow_up_reuse"] is True
    # Retrieval is still skipped on the reuse path...
    mock_search_tool.search_transcripts.assert_awaited_once()
    # ...but clip 2's source file is now ingested and visible downstream.
    mock_file_tool.fetch_source_bundle.assert_awaited_with("vid-two")
    assert state.values["source_file_content"] == "ingested now"
    assert state.values["source_content_video_id"] == "vid-two"


def _last_analyze_payload(llm: AsyncMock) -> dict[str, Any]:
    """The JSON payload handed to the LLM on the most recent analyze-shaped call."""
    for call in reversed(llm.ainvoke.await_args_list):
        messages = call.args[0] if call.args else call.kwargs.get("input")
        if not messages or len(messages) < 2:
            continue
        try:
            payload = json.loads(str(messages[-1].content))
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and "max_clips" in payload:
            return payload
    raise AssertionError("no analyze-shaped call found")


@pytest.mark.asyncio
async def test_a_reuse_turn_gives_analyze_both_the_prior_clips_and_the_freshly_ingested_source(
    make_runtime: Any, mock_search_tool: AsyncMock, mock_file_tool: AsyncMock
) -> None:
    """The end-to-end "dive deeper" path: turn 1 produces candidate_clips, turn 2
    ("dive deeper on clip 2") takes the reuse shortcut, fetch_source ingests clip
    2's source file since it was never pulled before, and analyze sees BOTH the
    prior turn's clips (to resolve "clip 2") and the freshly ingested transcript
    (to answer from the right file) in the same call."""
    mock_file_tool.fetch_source_bundle = AsyncMock(
        return_value=[
            B2FetchedFile(file_name="t.txt", content="clip two's full transcript", is_text=True)
        ]
    )
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(
        side_effect=[
            AIMessage(content=r)
            for r in (
                json.dumps({"intent": "new_request", "user_query": "hook moments"}),
                json.dumps({"clips": CLIPS}),
                CRITIQUE_OK,
                "## Your Clip Options",
                json.dumps(
                    {
                        "intent": "follow_up",
                        "user_query": "dive deeper on clip 2",
                        "reuse_context": True,
                    }
                ),
                json.dumps({"clips": [CLIPS[1]]}),
                CRITIQUE_OK,
                "## Deeper on Option 2",
            )
        ]
    )
    runtime: RuntimeContext = make_runtime(llm)
    graph = build_graph(checkpointer=MemorySaver())
    session_id = "reuse-ingest-combined"
    config: dict[str, Any] = {"configurable": {"runtime": runtime, "thread_id": session_id}}

    def _turn(message: str) -> dict[str, Any]:
        return {
            "messages": [HumanMessage(content=message)],
            "session_id": session_id,
            "session_documents": [],
        }

    async for _ in graph.astream(_turn("Find hook clips"), config=config, stream_mode="updates"):
        pass
    await graph.aupdate_state(
        config,
        {"confirmation_response": "rejected", "awaiting_confirmation": False},
        as_node="post_stub",
    )
    async for _ in graph.astream(
        _turn("dive deeper on clip 2"), config=config, stream_mode="updates"
    ):
        pass

    payload = _last_analyze_payload(llm)
    assert payload.get("source_file_content") == "clip two's full transcript"
    previous_clips = payload.get("previous_clips")
    assert previous_clips, "analyze must still see what clip 2 referred to"
    assert previous_clips[1]["hook_quote"] == "Second hook"


# ── truncation keeps the summary, not just the transcript's opening ──────────


def _fetched(name: str, text: str) -> B2FetchedFile:
    return B2FetchedFile(file_name=name, content=text, is_text=True)


def test_truncation_shrinks_the_transcript_and_keeps_the_summary_whole() -> None:
    """Truncating the joined string dropped the summary off the end, leaving the
    model a source's opening minutes with no idea what the rest covered."""
    bundle = [
        _fetched("TCP001 - FLOOR RCM_transcript.txt", "T" * 5_000),
        _fetched("TCP001 - FLOOR RCM_summary.txt", "S" * 200),
    ]

    combined = fetch_source._combine_bundle(bundle, max_chars=2_000)

    assert combined is not None
    # max_chars caps the source text; the marker is appended on top of it.
    assert len(combined) <= 2_000 + len(fetch_source._TRUNCATION_MARKER)
    assert "S" * 200 in combined, "the whole summary must survive"
    assert combined.count("T") < 5_000, "the transcript is the part that gives way"
    assert combined.index("_transcript.txt") < combined.index("_summary.txt")


def test_a_bundle_that_fits_is_not_truncated() -> None:
    bundle = [
        _fetched("a_transcript.txt", "T" * 100),
        _fetched("a_summary.txt", "S" * 50),
    ]
    combined = fetch_source._combine_bundle(bundle, max_chars=10_000)
    assert combined is not None and "truncated" not in combined


def test_a_single_part_is_returned_unlabelled() -> None:
    combined = fetch_source._combine_bundle([_fetched("solo_transcript.txt", "body")], max_chars=0)
    assert combined == "body"
