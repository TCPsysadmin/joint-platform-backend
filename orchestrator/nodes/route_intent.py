from __future__ import annotations

import re
from typing import Any

import structlog
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.config import settings
from orchestrator.context import build_context_window
from orchestrator.errors import LLMError
from orchestrator.llm_json import parse_llm_json
from orchestrator.prompts import load_prompt
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)

_URL_RE = re.compile(r"https?://[^\s)]+", re.IGNORECASE)
_FILE_RE = re.compile(
    r"(?P<ref>[\w./:%+\- ]+?\."
    r"(?:mp4|mov|m4v|mp3|wav|m4a|aac|flac|pdf|docx|txt|md|csv|json|srt|vtt))\b",
    re.IGNORECASE,
)
_NAMED_SOURCE_RE = re.compile(
    r"\b(?:the\s+)?(?:video|file|source|recording|upload)\s+"
    r"(?:is|called|named)\s+(?P<ref>.+?)(?=\s+(?:what\b|what's\b|summari[sz]e\b|"
    r"tell\b|can\b|please\b)|[?.!]*$)",
    re.IGNORECASE | re.DOTALL,
)
_CODED_TITLE_RE = re.compile(
    r"\b(?P<ref>[A-Z0-9]{2,}(?:[_-][A-Z0-9]+)*_[0-9]{8}\s*-\s*.+?)"
    r"(?=\s+(?:what\b|what's\b|summari[sz]e\b|tell\b)|[?.!]*$)",
    re.IGNORECASE | re.DOTALL,
)
_SOURCE_ANSWER_RE = re.compile(
    r"\b("
    r"what\s+(?:is|is\s+this|is\s+it|is\s+that).{0,30}about|"
    r"what'?s\s+(?:this|it|that).{0,30}about|"
    r"summari[sz]e|"
    r"tell\s+me\s+about|"
    r"what\s+happens\s+in|"
    r"what\s+does\s+(?:this|it|that)\s+(?:cover|say)"
    r")\b",
    re.IGNORECASE | re.DOTALL,
)


def _clean_source_reference(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    return text[:300]


def _latest_message_text(messages: list[BaseMessage]) -> str:
    if not messages:
        return ""
    content = messages[-1].content
    if isinstance(content, str):
        return content
    return str(content)


def _strip_reference_noise(text: str) -> str:
    cleaned = text.strip().strip("`'\"“”‘’.,!? \n\r\t")
    cleaned = re.sub(
        r"(?is)^.*\b(?:from|use|using|file|video|source|recording|upload|called|named|is)\s+",
        "",
        cleaned,
    )
    return cleaned.strip().strip("`'\"“”‘’.,!? \n\r\t")


def _detect_source_reference(text: str) -> str | None:
    url_match = _URL_RE.search(text)
    if url_match:
        return _strip_reference_noise(url_match.group(0))

    named_match = _NAMED_SOURCE_RE.search(text)
    if named_match:
        return _strip_reference_noise(named_match.group("ref"))

    file_match = _FILE_RE.search(text)
    if file_match:
        return _strip_reference_noise(file_match.group("ref"))

    coded_title_match = _CODED_TITLE_RE.search(text)
    if coded_title_match:
        return _strip_reference_noise(coded_title_match.group("ref"))

    return None


def _is_source_answer_request(text: str) -> bool:
    return bool(_SOURCE_ANSWER_RE.search(text))


def _clean_user_query(value: object, *, fallback: str) -> str:
    text = str(value or "").strip()
    if not text or text.lower() == "null":
        text = fallback.strip()
    return text[:200]


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

    latest_text = _latest_message_text(state["messages"])
    detected_source_reference = _detect_source_reference(latest_text)

    # The model decides — in this same call — whether a follow-up can reuse the
    # existing chunks/transcript or needs a fresh retrieval. Cheap: no extra LLM call.
    reuse_context = bool(data.get("reuse_context", False)) if intent == "follow_up" else False
    source_reference = _clean_source_reference(data.get("source_reference"))
    if detected_source_reference and not source_reference:
        source_reference = detected_source_reference

    source_task: str | None = data.get("source_task")
    if source_task not in ("clip_recommendation", "source_answer"):
        source_task = None
    if source_reference and _is_source_answer_request(latest_text):
        source_task = "source_answer"
    elif source_reference and source_task is None:
        source_task = "clip_recommendation"

    if source_reference and intent == "chitchat":
        intent = "new_request"
        reuse_context = False

    logger.info(
        "intent_routed",
        intent=intent,
        reuse_context=reuse_context,
        source_reference=source_reference,
        source_task=source_task,
        session_id=state.get("session_id"),
    )

    base: dict[str, Any] = {
        "intent": intent,
        "follow_up_reuse": reuse_context,
        "user_query": _clean_user_query(data.get("user_query"), fallback=latest_text)
        if intent != "chitchat"
        else None,
    }

    if intent == "new_request" or (intent == "follow_up" and not reuse_context) or source_reference:
        base.update(
            {
                "source_reference": source_reference,
                "source_task": source_task,
                "source_video_id": None,
                "source_metadata": None,
                "source_resolution_error": None,
            }
        )

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
                "candidate_clips": [],
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
                "candidate_clips": [],
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
                "candidate_clips": [],
                "candidate_recommendation": None,
                "critique_result": None,
                "final_recommendation": None,
                "awaiting_confirmation": False,
                "confirmation_response": None,
            }
        )

    return base
