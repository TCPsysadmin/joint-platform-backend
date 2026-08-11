from __future__ import annotations

from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


def _as_float(value: object, default: float) -> float:
    """Convert timestamp-shaped state safely without trusting untyped graph data."""
    if isinstance(value, (int, float, str)):
        try:
            return float(value)
        except ValueError:
            pass
    return default


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    """Extend the primary recommendation on both sides of its current range."""
    _ = config
    current = dict(state.get("candidate_recommendation") or {})
    if not current:
        return {"command": None}

    increment = _latest_increment(state)
    total = int(state.get("expand_seconds") or increment)

    if current.get("has_timestamps"):
        start = _as_float(current.get("start_seconds"), 0.0)
        end = _as_float(current.get("end_seconds"), start)
        current["start_seconds"] = max(0.0, start - increment)
        current["end_seconds"] = end + increment

    rationale = str(current.get("rationale") or "").strip()
    expansion_note = f"Expanded by {increment} seconds before and after."
    current["rationale"] = f"{rationale} {expansion_note}".strip()

    logger.info(
        "clip_expanded",
        increment_seconds=increment,
        total_seconds=total,
        session_id=state.get("session_id"),
    )
    return {
        "command": None,
        "candidate_clips": [current],
        "candidate_recommendation": current,
        "critique_result": {
            **(state.get("critique_result") or {}),
            "verdict": "approved",
            "forced": False,
            "note": expansion_note,
        },
    }


def _latest_increment(state: AgentState) -> int:
    query = str(state.get("user_query") or "")
    for token in query.split():
        if token.isdigit():
            return max(1, min(int(token), 300))
    return 15
