from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orchestrator.config import settings
from orchestrator.nodes import (
    analyze,
    chat_response,
    critique,
    fetch_doctrine,
    post_stub,
    recommend,
    refine,
    retrieve,
    route_intent,
)
from orchestrator.state import AgentState


def _intent_router(state: AgentState) -> str:
    intent = state.get("intent")
    if intent == "chitchat":
        return "chitchat"
    if intent == "follow_up":
        return "follow_up"
    if settings.require_brand_doctrine:
        return "fetch_doctrine"
    return "retrieve"


def _retrieve_router(state: AgentState) -> str:
    """After retrieve: skip analyze if we've already force-approved."""
    critique_result = state.get("critique_result") or {}
    if critique_result.get("forced") and critique_result.get("verdict") == "approved":
        return "recommend"
    return "analyze"


def _critique_router(state: AgentState) -> str:
    critique_result = state.get("critique_result") or {}
    verdict = str(critique_result.get("verdict", "retry"))
    if verdict == "approved":
        return "recommend"
    if verdict == "needs_refinement":
        return "refine"
    return "retrieve"


def build_graph(
    checkpointer: BaseCheckpointSaver | None = None,
) -> CompiledStateGraph:
    builder: StateGraph = StateGraph(AgentState)

    builder.add_node("route_intent", route_intent.run)
    builder.add_node("chat_response", chat_response.run)
    builder.add_node("fetch_doctrine", fetch_doctrine.run)
    builder.add_node("retrieve", retrieve.run)
    builder.add_node("analyze", analyze.run)
    builder.add_node("critique", critique.run)
    builder.add_node("refine", refine.run)
    builder.add_node("recommend", recommend.run)
    builder.add_node("post_stub", post_stub.run)

    builder.add_edge(START, "route_intent")

    builder.add_conditional_edges(
        "route_intent",
        _intent_router,
        {
            "chitchat": "chat_response",
            "follow_up": "recommend",
            "fetch_doctrine": "fetch_doctrine",
            "retrieve": "retrieve",
        },
    )

    builder.add_edge("chat_response", END)
    builder.add_edge("fetch_doctrine", "retrieve")

    builder.add_conditional_edges(
        "retrieve",
        _retrieve_router,
        {"analyze": "analyze", "recommend": "recommend"},
    )

    builder.add_edge("analyze", "critique")

    builder.add_conditional_edges(
        "critique",
        _critique_router,
        {"recommend": "recommend", "refine": "refine", "retrieve": "retrieve"},
    )

    builder.add_edge("refine", "retrieve")
    builder.add_edge("recommend", "post_stub")
    builder.add_edge("post_stub", END)

    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["post_stub"],
    )
