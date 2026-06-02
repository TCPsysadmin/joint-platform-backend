from __future__ import annotations

import dataclasses
import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.config import settings
from orchestrator.errors import LLMError, RetrievalError
from orchestrator.llm_json import parse_llm_json
from orchestrator.prompts import load_prompt
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]
    prompt = load_prompt("analyze.md")

    query = state.get("refined_query") or state.get("user_query") or ""
    segments = state.get("retrieved_segments") or []
    doctrine = state.get("brand_doctrine") or {}
    source_file_content = state.get("source_file_content")

    payload: dict[str, object] = {
        "user_query": query,
        "max_clips": settings.clip_candidate_count,
        "segments": segments,
        "brand_doctrine": doctrine,
    }
    if source_file_content:
        payload["source_file_content"] = source_file_content

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(content=json.dumps(payload, default=str)),
    ]

    response = await runtime.llm.ainvoke(messages)
    parsed: dict[str, Any] = parse_llm_json(str(response.content), source="analyze")

    clips = parsed.get("clips")
    if not isinstance(clips, list) or not clips:
        raise LLMError("analyze returned no clips")
    clips = [c for c in clips if isinstance(c, dict)][: settings.clip_candidate_count]
    if not clips:
        raise LLMError("analyze returned no valid clip objects")

    # Ensure every clip carries a broll_suggestions key (system-owned).
    for clip in clips:
        clip.setdefault("broll_suggestions", [])

    primary = clips[0]

    # Embed the primary hook quote (not the user query) for b-roll matching. We only
    # match b-roll for the top clip — the others are presented as ranked alternatives
    # and don't need their own asset lookups (keeps this to one match_assets call).
    hook_quote = str(primary.get("hook_quote") or query)
    try:
        hook_embedding = await runtime.embedder.embed(hook_quote)
    except Exception as exc:
        raise RetrievalError(f"Hook embedding failed: {exc}") from exc

    try:
        assets = await runtime.search_tool.match_assets(
            query_embedding=hook_embedding,
            asset_types=["broll_prompt", "music", "overlay"],
            match_count=8,
        )
    except Exception as exc:
        raise RetrievalError(f"Asset matching failed: {exc}") from exc

    primary["broll_suggestions"] = [dataclasses.asdict(a) for a in assets]

    logger.info(
        "analyze_completed",
        clip_count=len(clips),
        primary_video_id=primary.get("video_id"),
        has_timestamps=primary.get("has_timestamps"),
        broll_count=len(assets),
        session_id=state.get("session_id"),
    )

    return {
        "candidate_clips": clips,
        # candidate_recommendation = the primary clip; critique + post_stub key off it.
        "candidate_recommendation": primary,
    }
