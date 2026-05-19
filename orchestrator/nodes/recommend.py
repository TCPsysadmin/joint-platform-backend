from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.errors import LLMError
from orchestrator.prompts import load_prompt
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]
    prompt = load_prompt("recommend.md")

    candidate = state.get("candidate_recommendation") or {}
    critique = state.get("critique_result") or {}
    doctrine = state.get("brand_doctrine") or {}

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(
            content=json.dumps(
                {
                    "candidate": candidate,
                    "critique": critique,
                    "doctrine": doctrine,
                },
                default=str,
            )
        ),
    ]

    response = await runtime.llm.ainvoke(messages)
    final_recommendation = str(response.content).strip()

    if not final_recommendation:
        raise LLMError("recommend returned empty content")

    logger.info("recommendation_formatted", session_id=state.get("session_id"))

    return {
        # Add to messages so the recommendation appears in session history
        # and is visible when GET /sessions/{id}/messages is called.
        "messages": [AIMessage(content=final_recommendation)],
        "final_recommendation": final_recommendation,
        "awaiting_confirmation": True,
    }
