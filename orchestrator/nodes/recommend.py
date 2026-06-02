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

    clips = state.get("candidate_clips") or []
    if not clips and state.get("candidate_recommendation"):
        clips = [state["candidate_recommendation"]]  # type: ignore[list-item]
    critique = state.get("critique_result") or {}
    doctrine = state.get("brand_doctrine") or {}
    transcript = state.get("source_file_content")
    user_query = state.get("refined_query") or state.get("user_query") or ""

    payload: dict[str, Any] = {
        "user_query": user_query,
        # Ranked clip candidates, best-first. The critique below scores the top clip.
        "clips": clips,
        "critique": critique,
        "doctrine": doctrine,
    }
    # Ground the final recommendation in the actual transcript pulled from B2, not
    # just the terse candidate JSON — this is what makes the answer in-depth.
    if transcript:
        payload["source_file_content"] = transcript

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=json.dumps(payload, default=str)),
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
