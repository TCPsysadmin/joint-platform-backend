from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.errors import LLMError
from orchestrator.prompts import load_prompt
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]
    prompt = load_prompt("critique.md")

    candidate: dict[str, Any] = dict(state.get("candidate_recommendation") or {})
    doctrine: dict[str, Any] = dict(state.get("brand_doctrine") or {})
    has_timestamps: bool = bool(candidate.get("has_timestamps", False))

    rubric: list[dict[str, Any]] = list(doctrine.get("rubric") or [])

    # Exclude dimensions that require timestamps when the candidate has none.
    if not has_timestamps:
        rubric = [d for d in rubric if not d.get("requires_timestamps", False)]

    # Renormalize weights over remaining dimensions.
    total_weight: float = sum(float(d.get("weight", 1.0)) for d in rubric)
    if total_weight > 0:
        rubric = [{**d, "weight": float(d.get("weight", 1.0)) / total_weight} for d in rubric]

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(
            content=json.dumps(
                {
                    "candidate": candidate,
                    "rubric": rubric,
                    "has_timestamps": has_timestamps,
                },
                default=str,
            )
        ),
    ]

    response = await runtime.llm.ainvoke(messages)
    raw = str(response.content).strip()

    try:
        result: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"critique returned non-JSON: {raw!r}") from exc

    weighted_score: float = float(result.get("weighted_score", 0.0))
    min_a_tier: float = float(doctrine.get("minimum_a_tier_score", 0.75))
    auto_reject: float = float(doctrine.get("auto_reject_below", 0.40))

    if weighted_score >= min_a_tier:
        verdict = "approved"
    elif weighted_score < auto_reject:
        verdict = "retry"
    else:
        verdict = "needs_refinement"

    new_iteration = state.get("iteration_count", 0) + 1

    logger.info(
        "critique_completed",
        verdict=verdict,
        weighted_score=weighted_score,
        iteration_count=new_iteration,
        session_id=state.get("session_id"),
    )

    return {
        "critique_result": {**result, "verdict": verdict, "forced": False},
        "iteration_count": new_iteration,
    }
