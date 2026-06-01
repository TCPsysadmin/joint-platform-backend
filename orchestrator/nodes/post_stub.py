from __future__ import annotations

from typing import Any

import structlog
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.config import settings
from orchestrator.errors import ToolError
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState
from orchestrator.tools.protocols import ClipPayload

logger = structlog.get_logger(__name__)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]

    if state.get("confirmation_response") != "approved":
        logger.info("post_skipped_not_approved", session_id=state.get("session_id"))
        return {
            "awaiting_confirmation": False,
            "messages": [
                AIMessage(
                    content="Got it — clip not posted. Let me know if you'd like to try a different clip."
                )
            ],
        }

    candidate: dict[str, Any] = dict(state.get("candidate_recommendation") or {})
    video_id = str(candidate.get("video_id") or "")

    # OpusClip pulls the source video directly from B2 via a signed URL, so we
    # never proxy the bytes. Resolve it now; fall back to video_id only if the
    # publisher can make sense of that on its own (e.g. a stub).
    metadata: dict[str, Any] = {}
    try:
        signed_url = await runtime.file_tool.get_download_url(
            video_id, valid_duration_seconds=settings.b2_download_url_ttl_seconds
        )
    except Exception as exc:
        raise ToolError(f"Resolving source video URL failed for {video_id}: {exc}") from exc
    if signed_url:
        metadata["video_url"] = signed_url
    else:
        logger.warning("post_no_signed_url", video_id=video_id, session_id=state.get("session_id"))

    payload = ClipPayload(
        video_id=video_id,
        start_seconds=float(candidate.get("start_seconds") or 0.0),
        end_seconds=float(candidate.get("end_seconds") or 0.0),
        hook_quote=str(candidate.get("hook_quote") or ""),
        broll_suggestions=list(candidate.get("broll_suggestions") or []),
        metadata=metadata,
    )

    try:
        result = await runtime.publish_tool.create_and_post_clip(payload)
    except Exception as exc:
        raise ToolError(f"Publishing clip failed: {exc}") from exc

    logger.info(
        "clip_posted",
        clip_id=result.clip_id,
        status=result.status,
        session_id=state.get("session_id"),
    )

    reply = f"Clip posted! **ID:** `{result.clip_id}` | **Status:** {result.status}"
    if result.url:
        reply += f"\n\n[View clip]({result.url})"

    return {
        "awaiting_confirmation": False,
        "messages": [AIMessage(content=reply)],
    }
