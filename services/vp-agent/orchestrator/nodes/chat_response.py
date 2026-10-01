from __future__ import annotations

from typing import Any

import structlog
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.config import settings
from orchestrator.context import build_context_window
from orchestrator.runtime import RuntimeContext
from orchestrator.session_documents import build_document_context
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]

    windowed = build_context_window(state["messages"], settings.context_window_messages)
    document_context = build_document_context(list(state.get("session_documents") or []))
    messages = (
        [SystemMessage(content=document_context), *windowed] if document_context else windowed
    )
    response = await runtime.llm.ainvoke(messages)

    logger.info("chat_response_generated", session_id=state.get("session_id"))

    return {"messages": [response]}
