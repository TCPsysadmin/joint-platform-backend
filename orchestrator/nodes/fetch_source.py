from __future__ import annotations

from typing import Any

import structlog
from langchain_core.runnables import RunnableConfig

from orchestrator.errors import RetrievalError
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]

    if state.get("source_file_content"):
        return {}

    segments = state.get("retrieved_segments") or []
    source_video_id = str(state.get("source_video_id") or "")

    if not segments and not source_video_id:
        return {"source_file_content": None}

    # For source-specific requests, use the resolved source video. Otherwise use
    # the top-ranked segment's video as before.
    if not source_video_id:
        source_video_id = str(segments[0].get("video_id") or "")
    if not source_video_id:
        logger.warning("fetch_source_no_video_id", session_id=state.get("session_id"))
        return {"source_file_content": None}

    try:
        content = await runtime.file_tool.fetch_source_file(source_video_id)
    except Exception as exc:
        raise RetrievalError(f"B2 file download failed for {source_video_id}: {exc}") from exc

    logger.info(
        "fetch_source_completed",
        source_video_id=source_video_id,
        has_content=content is not None,
        char_count=len(content) if content else 0,
        session_id=state.get("session_id"),
    )

    return {"source_file_content": content}
