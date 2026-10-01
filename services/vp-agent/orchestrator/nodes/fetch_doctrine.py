from __future__ import annotations

import dataclasses
from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from orchestrator.errors import ToolError
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]

    try:
        doctrine = await runtime.doctrine_tool.get_active_doctrine()
    except Exception as exc:
        raise ToolError(f"Failed to load brand doctrine: {exc}") from exc

    logger.info(
        "doctrine_fetched",
        doctrine_name=doctrine.name,
        session_id=state.get("session_id"),
    )

    return {"brand_doctrine": dataclasses.asdict(doctrine)}
