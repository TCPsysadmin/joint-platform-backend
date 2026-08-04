from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.graph import build_graph
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState
from orchestrator.tools.protocols import B2FileEntry, SourceVideo, TranscriptHit


def _base_input(session_id: str = "smoke-test-session") -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="Find me a great hook clip about consistency")],
        "session_id": session_id,
        "intent": None,
        "follow_up_reuse": False,
        "user_query": None,
        "source_reference": None,
        "source_task": None,
        "source_video_id": None,
        "source_metadata": None,
        "source_resolution_error": None,
        "session_documents": [],
        "refined_query": None,
        "iteration_count": 0,
        "previous_segment_ids": [],
        "retrieved_segments": [],
        "brand_doctrine": None,
        "source_file_content": None,
        "candidate_clips": [],
        "candidate_recommendation": None,
        "critique_result": None,
        "final_recommendation": None,
        "awaiting_confirmation": False,
        "confirmation_response": None,
    }


def _make_llm_mock(*responses: str) -> AsyncMock:
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=[AIMessage(content=r) for r in responses])
    return llm


def test_graph_compiles() -> None:
    graph = build_graph()
    assert graph is not None


def test_graph_compiles_with_memory_checkpointer() -> None:
    graph = build_graph(checkpointer=MemorySaver())
    assert graph is not None


@pytest.mark.asyncio
async def test_graph_chitchat_path(make_runtime: Any, mock_llm_chitchat: Any) -> None:
    runtime: RuntimeContext = make_runtime(mock_llm_chitchat)
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    config: dict[str, Any] = {
        "configurable": {
            "runtime": runtime,
            "thread_id": "chitchat-session",
        }
    }

    final_state: AgentState | None = None
    async for chunk in graph.astream(
        _base_input("chitchat-session"), config=config, stream_mode="values"
    ):
        final_state = chunk

    assert final_state is not None
    assert final_state["intent"] == "chitchat"
    assert len(final_state["messages"]) >= 2


@pytest.mark.asyncio
async def test_graph_new_request_pauses_before_post(
    make_runtime: Any,
    mock_llm_full_flow: Any,
    mock_search_tool: AsyncMock,
) -> None:
    runtime: RuntimeContext = make_runtime(mock_llm_full_flow)
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    session_id = "full-flow-session"
    config: dict[str, Any] = {
        "configurable": {
            "runtime": runtime,
            "thread_id": session_id,
        }
    }

    # Consume all stream events until interrupt.
    async for _ in graph.astream(_base_input(session_id), config=config, stream_mode="updates"):
        pass

    # Graph should be paused before post_stub.
    state = await graph.aget_state(config)
    assert "post_stub" in state.next
    assert state.values.get("final_recommendation") is not None
    assert state.values.get("awaiting_confirmation") is True
    mock_search_tool.search_transcripts.assert_awaited_once()
    mock_search_tool.list_transcript_segments_for_video.assert_not_called()


@pytest.mark.asyncio
async def test_graph_confirm_approved_resumes(make_runtime: Any, mock_llm_full_flow: Any) -> None:
    runtime: RuntimeContext = make_runtime(mock_llm_full_flow)
    checkpointer = MemorySaver()
    graph = build_graph(checkpointer=checkpointer)

    session_id = "confirm-session"
    config: dict[str, Any] = {
        "configurable": {
            "runtime": runtime,
            "thread_id": session_id,
        }
    }

    async for _ in graph.astream(_base_input(session_id), config=config, stream_mode="updates"):
        pass

    # Approve and resume.
    await graph.aupdate_state(
        config,
        {"confirmation_response": "approved", "awaiting_confirmation": False},
    )

    final_state: AgentState | None = None
    async for chunk in graph.astream(None, config=config, stream_mode="values"):
        final_state = chunk

    assert final_state is not None
    assert final_state["awaiting_confirmation"] is False
    assert final_state["confirmation_response"] == "approved"


@pytest.mark.asyncio
async def test_graph_specific_source_uses_source_segments_only(
    make_runtime: Any,
    mock_search_tool: AsyncMock,
) -> None:
    mock_search_tool.resolve_source_video.return_value = SourceVideo(
        source_video_id="vid-abc",
        title="Q3 Interview",
        source_file="Q3 Interview.mp4",
    )

    llm = _make_llm_mock(
        json.dumps(
            {
                "intent": "new_request",
                "user_query": "best consistency hook",
                "source_reference": "Q3 Interview.mp4",
                "reuse_context": False,
            }
        ),
        json.dumps(
            {
                "clips": [
                    {
                        "video_id": "vid-abc",
                        "segment_id": "seg-001",
                        "start_seconds": 42.5,
                        "end_seconds": 53.1,
                        "has_timestamps": True,
                        "hook_quote": "The biggest mistake most creators make is they try to appeal to everyone",
                        "rationale": "Strong source-specific hook.",
                    }
                ]
            }
        ),
        json.dumps(
            {
                "dimension_scores": [],
                "weighted_score": 0.9,
                "overall_rationale": "Strong candidate.",
                "improvement_suggestions": [],
            }
        ),
        "### Option 1 - Consistency hook",
    )
    runtime: RuntimeContext = make_runtime(llm)
    graph = build_graph(checkpointer=MemorySaver())
    config: dict[str, Any] = {
        "configurable": {"runtime": runtime, "thread_id": "specific-source-session"}
    }

    async for _ in graph.astream(
        _base_input("specific-source-session"),
        config=config,
        stream_mode="updates",
    ):
        pass

    state = await graph.aget_state(config)
    assert "post_stub" in state.next
    assert state.values["source_reference"] == "Q3 Interview.mp4"
    assert state.values["source_video_id"] == "vid-abc"
    assert {s["video_id"] for s in state.values["retrieved_segments"]} == {"vid-abc"}
    mock_search_tool.resolve_source_video.assert_awaited_once_with("Q3 Interview.mp4")
    mock_search_tool.list_transcript_segments_for_video.assert_awaited_once_with(
        source_video_id="vid-abc"
    )
    mock_search_tool.search_transcripts.assert_not_called()


@pytest.mark.asyncio
async def test_graph_specific_source_not_found_does_not_fallback_to_global_search(
    make_runtime: Any,
    mock_search_tool: AsyncMock,
) -> None:
    mock_search_tool.resolve_source_video.return_value = None

    llm = _make_llm_mock(
        json.dumps(
            {
                "intent": "new_request",
                "user_query": "best pricing hook",
                "source_reference": "missing-file.mp4",
                "reuse_context": False,
            }
        )
    )
    runtime: RuntimeContext = make_runtime(llm)
    graph = build_graph(checkpointer=MemorySaver())
    config: dict[str, Any] = {
        "configurable": {"runtime": runtime, "thread_id": "missing-source-session"}
    }

    final_state: AgentState | None = None
    async for chunk in graph.astream(
        _base_input("missing-source-session"),
        config=config,
        stream_mode="values",
    ):
        final_state = chunk

    assert final_state is not None
    assert final_state["source_resolution_error"] is not None
    assert "couldn't find" in final_state["final_recommendation"].lower()
    assert final_state["awaiting_confirmation"] is False
    mock_search_tool.resolve_source_video.assert_awaited_once_with("missing-file.mp4")
    mock_search_tool.search_transcripts.assert_not_called()


@pytest.mark.asyncio
async def test_graph_specific_source_uses_b2_path_when_direct_db_lookup_misses(
    make_runtime: Any,
    mock_search_tool: AsyncMock,
    mock_file_tool: AsyncMock,
) -> None:
    mock_search_tool.resolve_source_video = AsyncMock(
        side_effect=[
            None,
            SourceVideo(
                source_video_id="vid-tcp003",
                title="TCP003 Meetings",
                source_file="TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr. Kieschnick.mp3",
            ),
        ]
    )
    mock_search_tool.list_transcript_segments_for_video.return_value = [
        TranscriptHit(
            segment_id="seg-tcp003",
            video_id="vid-tcp003",
            text="The group discusses interpersonal conflict and how to repair trust.",
            start_seconds=10.0,
            end_seconds=35.0,
            has_timestamps=True,
            score=1.0,
        )
    ]
    mock_file_tool.find_file_by_name.return_value = B2FileEntry(
        file_name=(
            "TCP003_MEETINGS/TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr. Kieschnick.mp3"
        ),
        file_id="b2-file-1",
        action="upload",
        content_length=123,
        content_type="audio/mpeg",
    )

    llm = _make_llm_mock(
        json.dumps(
            {
                "intent": "new_request",
                "user_query": "interesting parts",
                "source_reference": (
                    "TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr. Kieschnick"
                ),
                "source_task": "clip_recommendation",
                "reuse_context": False,
            }
        ),
        json.dumps(
            {
                "clips": [
                    {
                        "video_id": "vid-tcp003",
                        "segment_id": "seg-tcp003",
                        "start_seconds": 10.0,
                        "end_seconds": 35.0,
                        "has_timestamps": True,
                        "hook_quote": (
                            "The group discusses interpersonal conflict and how to repair trust."
                        ),
                        "rationale": "A useful section about repairing trust.",
                    }
                ]
            }
        ),
        json.dumps(
            {
                "dimension_scores": [],
                "weighted_score": 0.82,
                "overall_rationale": "Relevant source-specific moment.",
                "improvement_suggestions": [],
            }
        ),
        "### Option 1 - Repairing trust",
    )
    runtime: RuntimeContext = make_runtime(llm)
    graph = build_graph(checkpointer=MemorySaver())
    config: dict[str, Any] = {
        "configurable": {"runtime": runtime, "thread_id": "b2-source-session"}
    }

    async for _ in graph.astream(
        _base_input("b2-source-session"),
        config=config,
        stream_mode="updates",
    ):
        pass

    state = await graph.aget_state(config)
    assert "post_stub" in state.next
    assert state.values["source_video_id"] == "vid-tcp003"
    assert {s["video_id"] for s in state.values["retrieved_segments"]} == {"vid-tcp003"}
    assert mock_search_tool.resolve_source_video.await_count == 2
    mock_file_tool.find_file_by_name.assert_awaited_once()
    mock_search_tool.search_transcripts.assert_not_called()


@pytest.mark.asyncio
async def test_graph_source_about_question_overrides_chitchat_and_answers_from_source(
    make_runtime: Any,
    mock_search_tool: AsyncMock,
    mock_file_tool: AsyncMock,
) -> None:
    mock_search_tool.resolve_source_video.return_value = SourceVideo(
        source_video_id="tcp-001-ditl",
        title="TCP001_DITL_20250502 - JUST TRY",
        source_file="TCP001_DITL_20250502 - JUST TRY.mp3",
        has_timestamps=False,
    )
    mock_file_tool.fetch_source_file.return_value = (
        "The speaker talks through a day-in-the-life experiment focused on "
        "trying before overthinking, recording the messy attempts, and learning "
        "from the process."
    )

    llm = _make_llm_mock(
        json.dumps(
            {
                "intent": "chitchat",
                "user_query": None,
                "source_reference": None,
                "reuse_context": False,
            }
        ),
        "It is about a day-in-the-life experiment where the core theme is to just try.",
    )
    runtime: RuntimeContext = make_runtime(llm)
    graph = build_graph(checkpointer=MemorySaver())
    config: dict[str, Any] = {
        "configurable": {"runtime": runtime, "thread_id": "source-about-session"}
    }
    input_state = _base_input("source-about-session")
    input_state["messages"] = [
        HumanMessage(content="The Video is TCP001_DITL_20250502 - JUST TRY what is it about")
    ]

    final_state: AgentState | None = None
    async for chunk in graph.astream(input_state, config=config, stream_mode="values"):
        final_state = chunk

    assert final_state is not None
    assert final_state["intent"] == "new_request"
    assert final_state["source_task"] == "source_answer"
    assert final_state["source_reference"] == "TCP001_DITL_20250502 - JUST TRY"
    assert final_state["source_video_id"] == "tcp-001-ditl"
    assert "day-in-the-life" in final_state["final_recommendation"].lower()
    assert final_state["awaiting_confirmation"] is False
    mock_search_tool.resolve_source_video.assert_awaited_once_with(
        "TCP001_DITL_20250502 - JUST TRY"
    )
    mock_search_tool.list_transcript_segments_for_video.assert_awaited_once_with(
        source_video_id="tcp-001-ditl"
    )
    mock_search_tool.search_transcripts.assert_not_called()
    mock_search_tool.match_assets.assert_not_called()
