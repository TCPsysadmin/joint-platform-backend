from __future__ import annotations

import dataclasses
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from orchestrator.config import settings
from orchestrator.errors import RetrievalError
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)

_FORCED_NOTE = "Best available match — further refinement did not surface new options."


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]

    query = state.get("refined_query") or state.get("user_query") or ""
    iteration_count = state.get("iteration_count", 0)
    prior_critique = state.get("critique_result")

    # Cap check: if we've already done max iterations, force-approve with existing critique.
    if iteration_count >= settings.max_critique_iterations and prior_critique is not None:
        logger.info(
            "retrieve_forced_approve_max_iter",
            iteration_count=iteration_count,
            session_id=state.get("session_id"),
        )
        return {
            "critique_result": {
                **prior_critique,
                "verdict": "approved",
                "forced": True,
                "note": _FORCED_NOTE,
            }
        }

    try:
        embedding = await runtime.embedder.embed(query)
    except Exception as exc:
        raise RetrievalError(f"Embedding failed: {exc}") from exc

    try:
        hits = await runtime.search_tool.search_transcripts(
            query_text=query,
            query_embedding=embedding,
            match_count=40,
        )
    except Exception as exc:
        raise RetrievalError(f"Transcript search failed: {exc}") from exc

    segments = [dataclasses.asdict(h) for h in hits]
    top5_ids = [h.segment_id for h in hits[:5]]
    prev_ids = state.get("previous_segment_ids") or []

    logger.info(
        "retrieve_completed",
        hit_count=len(hits),
        top5_ids=top5_ids,
        session_id=state.get("session_id"),
    )

    # Staleness check: same top-5 and we already have a critique → force-approve.
    if top5_ids == prev_ids[:5] and prior_critique is not None:
        logger.info("retrieve_forced_approve_stale", session_id=state.get("session_id"))
        return {
            "retrieved_segments": segments,
            "previous_segment_ids": top5_ids,
            "critique_result": {
                **prior_critique,
                "verdict": "approved",
                "forced": True,
                "note": _FORCED_NOTE,
            },
        }

    return {
        "retrieved_segments": segments,
        "previous_segment_ids": top5_ids,
    }
