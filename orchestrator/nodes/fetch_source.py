from __future__ import annotations

import posixpath
from typing import Any

import structlog
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.config import settings
from orchestrator.context import referenced_clip_position
from orchestrator.errors import RetrievalError
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState
from orchestrator.tools.protocols import B2FetchedFile

logger = structlog.get_logger(__name__)

_TRUNCATION_MARKER = "\n\n[... source file truncated — ask about a specific part for more ...]"


def _latest_message_text(messages: list[BaseMessage] | None) -> str:
    if not messages:
        return ""
    content = messages[-1].content
    return content if isinstance(content, str) else str(content)


def _target_source_video_id(state: AgentState) -> str:
    """Which source's file this turn should have in context.

    An explicitly resolved source wins. Otherwise, on a follow-up that points at
    a clip already on screen ("dive deeper on clip 2"), it is *that clip's*
    video — not the top-ranked segment's, which is what the old default would
    have picked and is usually a different file. Everything else keeps the
    previous behaviour: the top-ranked retrieved segment.
    """
    explicit = str(state.get("source_video_id") or "")
    if explicit:
        return explicit

    clips = [c for c in (state.get("candidate_clips") or []) if isinstance(c, dict)]
    if clips and state.get("intent") == "follow_up" and state.get("iteration_count", 0) == 0:
        text = str(state.get("user_query") or "") or _latest_message_text(state.get("messages"))
        position = referenced_clip_position(text)
        if position is not None and 1 <= position <= len(clips):
            referenced = str(clips[position - 1].get("video_id") or "")
            if referenced:
                return referenced

    segments = [s for s in (state.get("retrieved_segments") or []) if isinstance(s, dict)]
    if segments:
        return str(segments[0].get("video_id") or "")
    if clips:
        return str(clips[0].get("video_id") or "")
    return ""


def _combine_bundle(results: list[B2FetchedFile], *, max_chars: int) -> str | None:
    """Join the text that came back, labelling parts only when there are several."""
    parts = [(posixpath.basename(r.file_name), r.content or "") for r in results if r.content]
    if not parts:
        return None

    labelled = len(parts) > 1

    def render(texts: list[str]) -> str:
        if not labelled:
            return texts[0]
        return "\n\n".join(
            f"## {name}\n{text}" for (name, _), text in zip(parts, texts, strict=True)
        )

    texts = [text for _, text in parts]
    combined = render(texts)
    if max_chars <= 0 or len(combined) <= max_chars:
        return combined

    # Over budget: shrink only the longest part — the transcript — and keep the
    # shorter ones whole. Truncating the join instead would drop the summary
    # entirely, leaving the model with a source's opening minutes and no idea
    # what the rest of it covers.
    longest = max(range(len(texts)), key=lambda i: len(texts[i]))
    overflow = len(combined) - max_chars
    keep = max(0, len(texts[longest]) - overflow)
    texts[longest] = texts[longest][:keep] + _TRUNCATION_MARKER
    return render(texts)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]

    source_video_id = _target_source_video_id(state)

    existing = state.get("source_file_content")
    if existing:
        ingested_for = state.get("source_content_video_id")
        # None = unknown provenance (a checkpoint written before this field, or
        # content retrieve resolved itself): keep it rather than re-fetching.
        if not source_video_id or ingested_for is None or ingested_for == source_video_id:
            return {}
        logger.info(
            "fetch_source_switching_source",
            previous_source_video_id=ingested_for,
            source_video_id=source_video_id,
            session_id=state.get("session_id"),
        )

    if not source_video_id:
        if state.get("retrieved_segments"):
            logger.warning("fetch_source_no_video_id", session_id=state.get("session_id"))
        return {"source_file_content": None, "source_content_video_id": None}

    # One bounded, concurrent batch: the source file plus any transcript sidecars
    # sharing its name. A sidecar that 404s comes back as a failed entry rather
    # than taking the whole fetch down.
    try:
        results = await runtime.file_tool.fetch_source_bundle(source_video_id)
    except Exception as exc:
        raise RetrievalError(f"B2 file download failed for {source_video_id}: {exc}") from exc

    content = _combine_bundle(results, max_chars=settings.source_file_content_max_chars)

    logger.info(
        "fetch_source_completed",
        source_video_id=source_video_id,
        file_count=len(results),
        text_file_count=sum(1 for r in results if r.content),
        failed_file_count=sum(1 for r in results if r.error),
        has_content=content is not None,
        char_count=len(content) if content else 0,
        session_id=state.get("session_id"),
    )

    return {
        "source_file_content": content,
        "source_content_video_id": source_video_id if content else None,
    }
