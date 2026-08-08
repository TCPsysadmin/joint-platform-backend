from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.errors import LLMError
from orchestrator.prompts import load_prompt
from orchestrator.runtime import RuntimeContext
from orchestrator.session_documents import build_document_context
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)

_PREVIOUS_RECOMMENDATION_CHARS = 4000


def _build_source_lookup(state: AgentState) -> dict[str, dict[str, str]]:
    """Map each video_id to its human-readable source title/file.

    Pulls from `retrieved_segments[].metadata` (which carries source_title and
    source_file per segment — this is what covers the multi-video general-search
    case) and from `source_metadata` (the single resolved source for
    source-specific requests). First non-empty value for a field wins, so a real
    name is never clobbered by a blank.
    """
    lookup: dict[str, dict[str, str]] = {}

    def _record(video_id: str, title: object, source_file: object) -> None:
        if not video_id:
            return
        entry = lookup.setdefault(video_id, {})
        if title and "source_title" not in entry:
            entry["source_title"] = str(title)
        if source_file and "source_file" not in entry:
            entry["source_file"] = str(source_file)

    for seg in state.get("retrieved_segments") or []:
        meta = seg.get("metadata")
        meta_dict = meta if isinstance(meta, dict) else {}
        _record(
            str(seg.get("video_id") or ""),
            meta_dict.get("source_title"),
            meta_dict.get("source_file"),
        )

    sm = state.get("source_metadata") or {}
    _record(str(sm.get("source_video_id") or ""), sm.get("title"), sm.get("source_file"))

    return lookup


def _with_source(clip: dict[str, object], lookup: dict[str, dict[str, str]]) -> dict[str, Any]:
    """Tag a clip with the human-readable source title/file of its video."""
    enriched: dict[str, Any] = dict(clip)
    src = lookup.get(str(clip.get("video_id") or ""), {})
    if src.get("source_title"):
        enriched["source_title"] = src["source_title"]
    if src.get("source_file"):
        enriched["source_file"] = src["source_file"]
    return enriched


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]
    prompt = load_prompt("recommend.md")

    raw_clips = state.get("candidate_clips") or []
    if not raw_clips and state.get("candidate_recommendation"):
        raw_clips = [state["candidate_recommendation"]]  # type: ignore[list-item]
    # Tag each clip with the source video it's pulled from so the final answer can
    # tell the creator which file every option will be clipped from.
    source_lookup = _build_source_lookup(state)
    clips = [_with_source(clip, source_lookup) for clip in raw_clips]
    critique = state.get("critique_result") or {}
    doctrine = state.get("brand_doctrine") or {}
    transcript = state.get("source_file_content")
    user_query = state.get("refined_query") or state.get("user_query") or ""
    document_context = build_document_context(list(state.get("session_documents") or []))

    payload: dict[str, Any] = {
        "user_query": user_query,
        # Ranked clip candidates, best-first, each tagged with its source video.
        # The critique below scores the top clip.
        "clips": clips,
        "critique": critique,
        "doctrine": doctrine,
    }
    # The resolved source (when the request targeted one specific video) — lets the
    # answer name the single source file up front instead of repeating it per clip.
    if state.get("source_metadata"):
        payload["source_metadata"] = state.get("source_metadata")
    # Ground the final recommendation in the actual transcript pulled from B2, not
    # just the terse candidate JSON — this is what makes the answer in-depth.
    if transcript:
        payload["source_file_content"] = transcript
    if document_context:
        payload["session_document_context"] = document_context
    # On a follow-up, `final_recommendation` still holds the previous turn's answer
    # (this node overwrites it below). Passing it through keeps option numbering and
    # phrasing continuous instead of presenting the follow-up as a cold new answer.
    previous_recommendation = state.get("final_recommendation")
    if state.get("intent") == "follow_up" and previous_recommendation:
        payload["previous_recommendation"] = str(previous_recommendation)[
            :_PREVIOUS_RECOMMENDATION_CHARS
        ]

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
