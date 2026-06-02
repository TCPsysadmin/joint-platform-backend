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
        # Not approved → clear the gate silently. Any conversational acknowledgement
        # is handled by the next turn (route_intent/chat_response), so we don't
        # append a canned "not posted" message here.
        logger.info("post_skipped_not_approved", session_id=state.get("session_id"))
        return {"awaiting_confirmation": False}

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

    # Create an OpusClip project from the entire source file and let Opus curate.
    payload = ClipPayload(
        video_id=video_id,
        start_seconds=0.0,
        end_seconds=0.0,
        hook_quote=str(candidate.get("hook_quote") or ""),
        broll_suggestions=list(candidate.get("broll_suggestions") or []),
        metadata=metadata,
        full_file=True,
        clip_durations=[settings.opus_default_clip_seconds],
    )

    try:
        result = await runtime.publish_tool.create_and_post_clip(payload)
    except Exception as exc:
        raise ToolError(f"Creating OpusClip project failed: {exc}") from exc

    logger.info(
        "opus_project_created",
        project_id=result.clip_id,
        status=result.status,
        session_id=state.get("session_id"),
    )

    reply = (
        f"Created your OpusClip project from the source video. "
        f"**Project ID:** `{result.clip_id}` | **Status:** {result.status}\n\n"
        f"Opus is now curating clips across the full file — they'll appear in your "
        f"OpusClip dashboard shortly."
    )
    if result.url:
        reply += f"\n\n[Open in OpusClip]({result.url})"

    return {
        "awaiting_confirmation": False,
        "messages": [AIMessage(content=reply)],
    }
