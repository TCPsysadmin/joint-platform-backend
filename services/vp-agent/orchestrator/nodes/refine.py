from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.llm_json import parse_llm_json
from orchestrator.prompts import load_prompt
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]
    prompt = load_prompt("refine.md")

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(
            content=json.dumps(
                {
                    "original_query": state.get("user_query") or "",
                    "refined_query": state.get("refined_query"),
                    "critique_result": state.get("critique_result"),
                    "candidate_recommendation": state.get("candidate_recommendation"),
                },
                default=str,
            )
        ),
    ]

    response = await runtime.llm.ainvoke(messages)
    result: dict[str, Any] = parse_llm_json(str(response.content), source="refine")

    refined_query = str(result.get("refined_query") or "")

    logger.info(
        "query_refined",
        refined_query=refined_query,
        session_id=state.get("session_id"),
    )

    return {"refined_query": refined_query}
