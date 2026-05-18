from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.errors import LLMError
from orchestrator.prompts import load_prompt
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]
    prompt = load_prompt("route_intent.md")

    messages = [SystemMessage(content=prompt), *state["messages"]]

    response = await runtime.llm.ainvoke(messages)
    raw = str(response.content).strip()

    try:
        data: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"route_intent returned non-JSON: {raw!r}") from exc

    intent = data.get("intent")
    if intent not in ("new_request", "follow_up", "chitchat"):
        raise LLMError(f"route_intent returned unknown intent: {intent!r}")

    logger.info("intent_routed", intent=intent, session_id=state.get("session_id"))

    base: dict[str, Any] = {
        "intent": intent,
        "user_query": data.get("user_query"),
    }

    if intent == "new_request":
        base.update(
            {
                "refined_query": None,
                "iteration_count": 0,
                "previous_segment_ids": [],
                "retrieved_segments": [],
                "brand_doctrine": None,
                "candidate_recommendation": None,
                "critique_result": None,
                "final_recommendation": None,
                "awaiting_confirmation": False,
                "confirmation_response": None,
            }
        )

    return base
