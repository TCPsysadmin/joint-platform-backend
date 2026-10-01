from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from orchestrator.config import settings
from orchestrator.nodes import (
    analyze,
    chat_response,
    critique,
    expand_clip,
    fetch_doctrine,
    fetch_source,
    post_stub,
    recommend,
    refine,
    retrieve,
    route_intent,
    source_answer,
    source_not_found,
)
from orchestrator.state import AgentState


def _intent_router(state: AgentState) -> str:
    intent = state.get("intent")
    if state.get("command") == "expand" and state.get("candidate_recommendation"):
        return "expand"
    if intent == "chitchat":
        return "chitchat"
    # A follow-up that can reuse the existing chunks/transcript skips retrieval and
    # re-analyzes directly. Otherwise it falls through to the same retrieval path as
    # a new request (route_intent has already reset the loop state for it).
    # Guard: only take the shortcut when there is genuinely something to reuse —
    # analyze has no segments to work from otherwise and would fail the turn.
    has_reusable_context = bool(state.get("retrieved_segments") or state.get("source_file_content"))
    if intent == "follow_up" and state.get("follow_up_reuse") and has_reusable_context:
        return "reuse"
    if settings.require_brand_doctrine and state.get("brand_doctrine") is None:
        return "fetch_doctrine"
    return "retrieve"


def _retrieve_router(state: AgentState) -> str:
    """After retrieve: skip fetch_source+analyze if we've already force-approved."""
    if state.get("source_resolution_error"):
        return "source_not_found"
    critique_result = state.get("critique_result") or {}
    if critique_result.get("forced") and critique_result.get("verdict") == "approved":
        return "recommend"
    return "fetch_source"


def _fetch_source_router(state: AgentState) -> str:
    if state.get("source_resolution_error"):
        return "source_not_found"
    if state.get("source_task") == "source_answer":
        return "source_answer"
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
    checkpointer: BaseCheckpointSaver | None = None,  # type: ignore[type-arg]
) -> CompiledStateGraph:  # type: ignore[type-arg]
    builder: StateGraph = StateGraph(AgentState)  # type: ignore[type-arg]

    builder.add_node("route_intent", route_intent.run)
    builder.add_node("chat_response", chat_response.run)
    builder.add_node("fetch_doctrine", fetch_doctrine.run)
    builder.add_node("retrieve", retrieve.run)
    builder.add_node("fetch_source", fetch_source.run)
    builder.add_node("analyze", analyze.run)
    builder.add_node("critique", critique.run)
    builder.add_node("expand_clip", expand_clip.run)
    builder.add_node("refine", refine.run)
    builder.add_node("recommend", recommend.run)
    builder.add_node("post_stub", post_stub.run)
    builder.add_node("source_answer", source_answer.run)
    builder.add_node("source_not_found", source_not_found.run)

    builder.add_edge(START, "route_intent")

    builder.add_conditional_edges(
        "route_intent",
        _intent_router,
        {
            "chitchat": "chat_response",
            "expand": "expand_clip",
            # Reuse enters at fetch_source, not analyze: retrieval is still
            # skipped, but "dive deeper on clip 2" gets a chance to ingest that
            # clip's source file if it isn't in context yet. fetch_source
            # short-circuits when the right transcript is already there, and
            # _fetch_source_router lands on analyze exactly as before.
            "reuse": "fetch_source",
            "fetch_doctrine": "fetch_doctrine",
            "retrieve": "retrieve",
        },
    )

    builder.add_edge("chat_response", END)
    builder.add_edge("expand_clip", "recommend")
    builder.add_edge("fetch_doctrine", "retrieve")

    builder.add_conditional_edges(
        "retrieve",
        _retrieve_router,
        {
            "fetch_source": "fetch_source",
            "recommend": "recommend",
            "source_not_found": "source_not_found",
        },
    )
    builder.add_edge("source_not_found", END)

    builder.add_conditional_edges(
        "fetch_source",
        _fetch_source_router,
        {
            "source_not_found": "source_not_found",
            "source_answer": "source_answer",
            "analyze": "analyze",
        },
    )
    builder.add_edge("source_answer", END)
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
