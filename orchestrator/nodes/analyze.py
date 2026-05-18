from __future__ import annotations

import dataclasses
import json
from typing import Any

import structlog
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.errors import LLMError, RetrievalError
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

    messages = [
        SystemMessage(content=prompt),
        HumanMessage(
            content=json.dumps(
                {
                    "user_query": query,
                    "segments": segments,
                    "brand_doctrine": doctrine,
                },
                default=str,
            )
        ),
    ]

    response = await runtime.llm.ainvoke(messages)
    raw = str(response.content).strip()

    try:
        candidate: dict[str, Any] = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError(f"analyze returned non-JSON: {raw!r}") from exc

    # Embed the hook quote (not the user query) for b-roll matching.
    hook_quote = str(candidate.get("hook_quote") or query)
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

    candidate["broll_suggestions"] = [dataclasses.asdict(a) for a in assets]

    logger.info(
        "analyze_completed",
        video_id=candidate.get("video_id"),
        has_timestamps=candidate.get("has_timestamps"),
        broll_count=len(assets),
        session_id=state.get("session_id"),
    )

    return {"candidate_recommendation": candidate}
