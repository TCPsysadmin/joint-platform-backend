"""Follow-up turns must keep the previous turn's clip context in view.

Reproduces the reported bug: after a recommendation, "dive deeper on clip 2"
reached `analyze` with no memory of what clip 2 was.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.confirm_intent import classify_confirmation_reply
from orchestrator.context import build_prior_context_summary, references_prior_clip
from orchestrator.graph import build_graph
from orchestrator.runtime import RuntimeContext

TURN_ONE_CLIPS = {
    "clips": [
        {
            "video_id": "vid-abc",
            "segment_id": "seg-001",
            "start_seconds": 42.5,
            "end_seconds": 53.1,
            "has_timestamps": True,
            "hook_quote": "The biggest mistake most creators make is they try to appeal to everyone",
            "rationale": "Strong contrarian hook.",
        },
        {
            "video_id": "vid-abc",
            "segment_id": "seg-002",
            "start_seconds": 88.0,
            "end_seconds": 99.0,
            "has_timestamps": True,
            "hook_quote": "Consistency beats intensity every single time",
            "rationale": "Punchy, memorable, on-doctrine.",
        },
    ]
}

CRITIQUE_OK = json.dumps(
    {
        "dimension_scores": [],
        "weighted_score": 0.9,
        "overall_rationale": "Strong candidate.",
        "improvement_suggestions": [],
    }
)


def _base_input(session_id: str, message: str) -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content=message)],
        "session_id": session_id,
        "session_documents": [],
    }


def _seeded_llm(*responses: str) -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=[AIMessage(content=r) for r in responses])
    return llm


def _analyze_payloads(llm: AsyncMock) -> list[dict[str, Any]]:
    """Every JSON payload handed to the LLM that looks like an analyze call."""
    payloads: list[dict[str, Any]] = []
    for call in llm.ainvoke.await_args_list:
        messages = call.args[0] if call.args else call.kwargs.get("input")
        if not messages or len(messages) < 2:
            continue
        try:
            payload = json.loads(str(messages[-1].content))
        except (ValueError, TypeError):
            continue
        if isinstance(payload, dict) and "max_clips" in payload:
            payloads.append(payload)
    return payloads


@pytest.mark.asyncio
async def test_follow_up_reuse_gives_analyze_the_previous_clips(
    make_runtime: Any,
    mock_search_tool: AsyncMock,
) -> None:
    llm = _seeded_llm(
        # ── turn 1 ────────────────────────────────────────────────────────────
        json.dumps({"intent": "new_request", "user_query": "hook moments about consistency"}),
        json.dumps(TURN_ONE_CLIPS),
        CRITIQUE_OK,
        "## Your Clip Options\n\n### Option 1 ...\n\n### Option 2 ...",
        # ── turn 2: route_intent, analyze, critique, recommend ────────────────
        json.dumps(
            {"intent": "follow_up", "user_query": "dive deeper on clip 2", "reuse_context": True}
        ),
        json.dumps({"clips": [TURN_ONE_CLIPS["clips"][1]]}),
        CRITIQUE_OK,
        "## Deeper on Option 2",
    )
    runtime: RuntimeContext = make_runtime(llm)
    graph = build_graph(checkpointer=MemorySaver())
    session_id = "follow-up-session"
    config: dict[str, Any] = {"configurable": {"runtime": runtime, "thread_id": session_id}}

    async for _ in graph.astream(
        _base_input(session_id, "Find hook clips about consistency"),
        config=config,
        stream_mode="updates",
    ):
        pass

    state = await graph.aget_state(config)
    assert "post_stub" in state.next
    assert len(state.values["candidate_clips"]) == 2

    # The confirmation gate is cleared without publishing (the "other" branch in
    # main._run_chat_turn), then the follow-up runs as a normal turn.
    await graph.aupdate_state(
        config,
        {"confirmation_response": "rejected", "awaiting_confirmation": False},
        as_node="post_stub",
    )

    async for _ in graph.astream(
        _base_input(session_id, "dive deeper on clip 2"),
        config=config,
        stream_mode="updates",
    ):
        pass

    payloads = _analyze_payloads(llm)
    assert len(payloads) == 2, "expected one analyze call per turn"
    follow_up_payload = payloads[1]

    previous = follow_up_payload.get("previous_clips")
    assert previous, "analyze must see the clips from the previous turn"
    assert previous[1]["hook_quote"] == "Consistency beats intensity every single time"
    assert previous[1]["segment_id"] == "seg-002"
    assert follow_up_payload.get("previous_recommendation")

    # Reuse path must not re-run retrieval.
    mock_search_tool.search_transcripts.assert_awaited_once()


@pytest.mark.asyncio
async def test_clip_reference_forces_follow_up_reuse_when_classifier_says_new_request(
    make_runtime: Any,
    mock_search_tool: AsyncMock,
) -> None:
    llm = _seeded_llm(
        json.dumps({"intent": "new_request", "user_query": "hook moments about consistency"}),
        json.dumps(TURN_ONE_CLIPS),
        CRITIQUE_OK,
        "## Your Clip Options",
        # Turn 2: the classifier gets it wrong and says new_request.
        json.dumps({"intent": "new_request", "user_query": "the second clip"}),
        json.dumps({"clips": [TURN_ONE_CLIPS["clips"][1]]}),
        CRITIQUE_OK,
        "## Deeper on Option 2",
    )
    runtime: RuntimeContext = make_runtime(llm)
    graph = build_graph(checkpointer=MemorySaver())
    session_id = "misclassified-session"
    config: dict[str, Any] = {"configurable": {"runtime": runtime, "thread_id": session_id}}

    async for _ in graph.astream(
        _base_input(session_id, "Find hook clips about consistency"),
        config=config,
        stream_mode="updates",
    ):
        pass
    await graph.aupdate_state(
        config,
        {"confirmation_response": "rejected", "awaiting_confirmation": False},
        as_node="post_stub",
    )
    async for _ in graph.astream(
        _base_input(session_id, "tell me more about the second clip"),
        config=config,
        stream_mode="updates",
    ):
        pass

    state = await graph.aget_state(config)
    assert state.values["intent"] == "follow_up"
    assert state.values["follow_up_reuse"] is True
    # Reused context → no second retrieval, and doctrine survived.
    mock_search_tool.search_transcripts.assert_awaited_once()
    assert state.values["brand_doctrine"] is not None


@pytest.mark.asyncio
async def test_affirmative_prefixed_refinement_is_not_an_approval() -> None:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"decision": "approve"}'))

    for message in (
        "yes, let's dive deeper on clip 2",
        "sure, tell me more about the second one",
        "ok but why does that one work?",
    ):
        assert await classify_confirmation_reply(llm, message) == "other", message

    # Decided in code, so the classifier is never consulted for these.
    llm.ainvoke.assert_not_awaited()

    # Approvals that merely point at the pending clip must NOT be swallowed by the
    # override — they still reach the classifier and still approve.
    for message in (
        "yes, create the opus project",
        "make this clip",
        "yes, do that one",
        "post that clip",
    ):
        assert await classify_confirmation_reply(llm, message) == "approve", message


@pytest.mark.asyncio
async def test_confirm_gate_plain_yes_approves_but_yes_dive_deeper_does_not() -> None:
    """The two ends of the confirm gate the deterministic override exists for:
    a bare "yes" is a real approval (and must reach the classifier, since it has
    no clip reference to decide on in code), while "yes, dive deeper on clip 2"
    is a refinement wearing an approval's clothes and must never publish."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(return_value=AIMessage(content='{"decision": "approve"}'))

    assert await classify_confirmation_reply(llm, "yes") == "approve"
    llm.ainvoke.assert_awaited_once()

    llm.ainvoke.reset_mock()
    assert await classify_confirmation_reply(llm, "yes, dive deeper on clip 2") == "other"
    llm.ainvoke.assert_not_awaited()


def test_references_prior_clip_matches_ordinals_and_indexes() -> None:
    for message in (
        "dive deeper on clip 2",
        "tell me more about the second clip",
        "expand on option #3",
        "what about that one",
        "why did you pick number 1",
    ):
        assert references_prior_clip(message), message

    for message in (
        "find me something about pricing",
        "make it shorter",
        "thanks!",
        "show me clips from the Q3 interview",
    ):
        assert not references_prior_clip(message), message


@pytest.mark.asyncio
async def test_analyze_omits_previous_clips_inside_the_refine_loop(
    make_runtime: Any,
) -> None:
    """After critique increments iteration_count, candidate_clips is this turn's
    rejected draft — not the previous turn's answer — so it must not be relabelled
    `previous_clips`, which would pin refinement to what critique threw out."""
    from orchestrator.nodes import analyze

    llm = _seeded_llm(json.dumps({"clips": [TURN_ONE_CLIPS["clips"][0]]}))
    runtime: RuntimeContext = make_runtime(llm)
    config: dict[str, Any] = {"configurable": {"runtime": runtime, "thread_id": "refine-loop"}}

    await analyze.run(
        {  # type: ignore[arg-type]
            "session_id": "refine-loop",
            "intent": "follow_up",
            # critique already ran once this turn, so these are this turn's draft.
            "iteration_count": 1,
            "user_query": "make it punchier",
            "retrieved_segments": [{"segment_id": "seg-001", "video_id": "vid-abc", "text": "hi"}],
            "candidate_clips": TURN_ONE_CLIPS["clips"],
            "final_recommendation": "## Your Clip Options",
            "session_documents": [],
        },
        config,  # type: ignore[arg-type]
    )

    payload = _analyze_payloads(llm)[0]
    assert "previous_clips" not in payload
    assert "previous_recommendation" not in payload


def test_prior_context_summary_lists_clips_and_documents() -> None:
    summary = build_prior_context_summary(
        {
            "candidate_clips": [
                {"hook_quote": "First hook"},
                {"hook_quote": "Second hook"},
            ],
            "final_recommendation": "## Your Clip Options",
            "retrieved_segments": [{"segment_id": "seg-1"}],
            "source_metadata": {"title": "Consistency Interview"},
            "session_documents": [{"filename": "brief.md"}],
        }  # type: ignore[arg-type]
    )
    assert summary is not None
    assert "2 clip option" in summary
    assert "Second hook" in summary
    assert "Consistency Interview" in summary
    assert "brief.md" in summary
    assert build_prior_context_summary({}) is None  # type: ignore[arg-type]
