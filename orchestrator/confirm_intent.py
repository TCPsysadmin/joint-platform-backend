from __future__ import annotations

from typing import Literal

import structlog
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from orchestrator.llm_json import parse_llm_json
from orchestrator.prompts import load_prompt

logger = structlog.get_logger(__name__)

ConfirmDecision = Literal["approve", "reject", "other"]


async def classify_confirmation_reply(llm: BaseChatModel, message: str) -> ConfirmDecision:
    """Classify a chat message sent while a recommendation awaits confirmation.

    Returns "approve" (act on the pending clip / create the Opus project),
    "reject" (decline without a new direction), or "other" (a new request or
    refinement that should be handled as a normal turn).

    Falls back to "other" on any classification/parse failure — the safe default,
    since it never creates a project the user didn't clearly ask for.
    """
    prompt = load_prompt("confirm_intent.md")
    try:
        response = await llm.ainvoke([SystemMessage(content=prompt), HumanMessage(content=message)])
        data = parse_llm_json(str(response.content), source="confirm_intent")
        decision = data.get("decision")
    except Exception as exc:  # noqa: BLE001 — never block the turn on classifier failure
        logger.warning("confirm_intent_classify_failed", error=str(exc))
        return "other"

    if decision not in ("approve", "reject", "other"):
        logger.warning("confirm_intent_unknown_decision", decision=decision)
        return "other"
    return decision  # type: ignore[return-value]
