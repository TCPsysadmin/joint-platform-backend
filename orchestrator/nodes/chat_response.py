from __future__ import annotations

from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]

    response = await runtime.llm.ainvoke(state["messages"])

    logger.info("chat_response_generated", session_id=state.get("session_id"))

    return {"messages": [response]}
