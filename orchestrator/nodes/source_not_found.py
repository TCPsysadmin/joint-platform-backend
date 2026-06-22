from __future__ import annotations

from typing import Any

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    _ = config
    message = state.get("source_resolution_error") or (
        "I couldn't find that source video or file. Try the exact title or filename."
    )

    logger.info(
        "source_not_found_response_generated",
        source_reference=state.get("source_reference"),
        session_id=state.get("session_id"),
    )

    return {
        "messages": [AIMessage(content=message)],
        "final_recommendation": message,
        "awaiting_confirmation": False,
    }
