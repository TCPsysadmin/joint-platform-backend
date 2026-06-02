from __future__ import annotations

from typing import Any

import structlog
from langchain_core.messages import SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.config import settings
from orchestrator.context import build_context_window
from orchestrator.errors import LLMError
from orchestrator.llm_json import parse_llm_json
from orchestrator.prompts import load_prompt
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]
    prompt = load_prompt("route_intent.md")

    windowed = build_context_window(state["messages"], settings.context_window_messages)
    messages = [SystemMessage(content=prompt), *windowed]

    response = await runtime.llm.ainvoke(messages)
    data: dict[str, Any] = parse_llm_json(str(response.content), source="route_intent")

    intent = data.get("intent")
    if intent not in ("new_request", "follow_up", "chitchat"):
        raise LLMError(f"route_intent returned unknown intent: {intent!r}")

    # The model decides — in this same call — whether a follow-up can reuse the
    # existing chunks/transcript or needs a fresh retrieval. Cheap: no extra LLM call.
    reuse_context = bool(data.get("reuse_context", False)) if intent == "follow_up" else False

    logger.info(
        "intent_routed",
        intent=intent,
        reuse_context=reuse_context,
        session_id=state.get("session_id"),
    )

    base: dict[str, Any] = {
        "intent": intent,
        "follow_up_reuse": reuse_context,
        "user_query": data.get("user_query"),
    }

    if intent == "new_request":
        # A brand-new request wipes all prior pipeline state.
        base.update(
            {
                "refined_query": None,
                "iteration_count": 0,
                "previous_segment_ids": [],
                "retrieved_segments": [],
                "brand_doctrine": None,
                "source_file_content": None,
                "candidate_recommendation": None,
                "critique_result": None,
                "final_recommendation": None,
                "awaiting_confirmation": False,
                "confirmation_response": None,
            }
        )
    elif intent == "follow_up" and not reuse_context:
        # Fresh retrieval for the follow-up: reset the retrieval/critique loop so
        # staleness and iteration-cap checks don't misfire on the new query.
        # Brand doctrine is preserved — it doesn't change within a session.
        base.update(
            {
                "refined_query": None,
                "iteration_count": 0,
                "previous_segment_ids": [],
                "retrieved_segments": [],
                "source_file_content": None,
                "candidate_recommendation": None,
                "critique_result": None,
                "final_recommendation": None,
                "awaiting_confirmation": False,
                "confirmation_response": None,
            }
        )
    elif intent == "follow_up" and reuse_context:
        # Reuse existing chunks + transcript, but re-reason from scratch against the
        # new instruction: clear the critique loop and force a fresh candidate.
        # refined_query is cleared so analyze keys off the new user_query.
        base.update(
            {
                "refined_query": None,
                "iteration_count": 0,
                "candidate_recommendation": None,
                "critique_result": None,
                "final_recommendation": None,
                "awaiting_confirmation": False,
                "confirmation_response": None,
            }
        )

    return base
