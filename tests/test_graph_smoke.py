from __future__ import annotations

from typing import Any

import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.graph import build_graph
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState


def _base_input(session_id: str = "smoke-test-session") -> dict[str, Any]:
    return {
        "messages": [HumanMessage(content="Find me a great hook clip about consistency")],
        "session_id": session_id,
        "intent": None,
        "user_query": None,
        "refined_query": None,
        "iteration_count": 0,
        "previous_segment_ids": [],
        "retrieved_segments": [],
        "brand_doctrine": None,
        "candidate_recommendation": None,
        "critique_result": None,
        "final_recommendation": None,
        "awaiting_confirmation": False,
        "confirmation_response": None,
    }


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
    make_runtime: Any, mock_llm_full_flow: Any
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
