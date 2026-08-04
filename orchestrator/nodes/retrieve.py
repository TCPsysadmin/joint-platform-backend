from __future__ import annotations

import dataclasses
import posixpath
import re
from typing import Any
from urllib.parse import unquote, urlparse

import structlog
from langchain_core.runnables import RunnableConfig

from orchestrator.config import settings
from orchestrator.doctrine_defaults import default_brand_doctrine_dict
from orchestrator.errors import RetrievalError
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState
from orchestrator.tools.protocols import B2FileEntry, SourceVideo

logger = structlog.get_logger(__name__)

_FORCED_NOTE = "Best available match — further refinement did not surface new options."
_CODED_SOURCE_RE = re.compile(r"\b[A-Z0-9]{2,}(?:[_-][A-Z0-9]+)*_[0-9]{8}\b", re.IGNORECASE)
_SOURCE_FILE_EXTENSIONS = {
    ".aac",
    ".csv",
    ".docx",
    ".flac",
    ".json",
    ".log",
    ".m4a",
    ".m4v",
    ".md",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".srt",
    ".text",
    ".txt",
    ".vtt",
    ".wav",
}


def _without_extension(value: str) -> str:
    root, extension = posixpath.splitext(value)
    if extension.lower() not in _SOURCE_FILE_EXTENSIONS:
        return value
    return root


def _source_reference_candidates(value: str) -> list[str]:
    candidates: list[str] = []

    def add(candidate: str | None) -> None:
        text = (candidate or "").strip().strip("/")
        if text and text not in candidates:
            candidates.append(text)

    decoded = unquote(value)
    add(value)
    add(decoded)

    parsed = urlparse(decoded)
    if parsed.scheme and parsed.netloc:
        path = unquote(parsed.path).strip("/")
        add(path)
        path_parts = path.split("/")
        if len(path_parts) >= 3 and path_parts[0] == "file":
            add("/".join(path_parts[2:]))
        add(posixpath.basename(path))

    for candidate in list(candidates):
        basename = posixpath.basename(candidate)
        add(basename)
        add(_without_extension(basename))
        add(_without_extension(candidate))
        coded_match = _CODED_SOURCE_RE.search(candidate)
        if coded_match:
            add(coded_match.group(0))

    return candidates


def _normalized_source_key(value: str) -> str:
    decoded = unquote(value).lower()
    basename = posixpath.basename(decoded.replace("\\", "/"))
    without_extension = _without_extension(basename)
    return re.sub(r"[^a-z0-9]+", "", without_extension)


def _find_session_document(
    documents: list[dict[str, object]],
    source_reference: str,
) -> dict[str, object] | None:
    generic_reference = re.sub(r"[^a-z]+", "", source_reference.lower())
    if len(documents) == 1 and re.fullmatch(
        r"(?:the|my|this)?(?:attached|uploaded)?(?:file|document|attachment|upload)",
        generic_reference,
    ):
        return documents[0]

    reference_keys = {
        key
        for candidate in _source_reference_candidates(source_reference)
        if (key := _normalized_source_key(candidate))
    }
    for document in documents:
        filename = str(document.get("filename") or "")
        document_key = _normalized_source_key(filename)
        if not document_key:
            continue
        if document_key in reference_keys or any(
            len(document_key) >= 6 and document_key in key for key in reference_keys
        ):
            return document
    return None


async def _find_b2_source_file(
    runtime: RuntimeContext,
    source_reference: str,
) -> B2FileEntry | None:
    for candidate in _source_reference_candidates(source_reference):
        try:
            entry = await runtime.file_tool.find_file_by_name(file_name=candidate)
        except Exception as exc:
            raise RetrievalError(f"B2 file lookup failed for {candidate!r}: {exc}") from exc
        if entry is not None:
            logger.info(
                "source_reference_b2_file_found",
                source_reference=source_reference,
                b2_path=entry.file_name,
            )
            return entry
    return None


async def _resolve_source_video(
    runtime: RuntimeContext,
    source_reference: str,
) -> tuple[SourceVideo | None, B2FileEntry | None, str | None]:
    source = await runtime.search_tool.resolve_source_video(source_reference)
    if source is not None:
        return source, None, None

    b2_entry = await _find_b2_source_file(runtime, source_reference)
    if b2_entry is None:
        return None, None, None

    for candidate in _source_reference_candidates(b2_entry.file_name):
        source = await runtime.search_tool.resolve_source_video(candidate)
        if source is not None:
            logger.info(
                "source_reference_resolved_from_b2_path",
                source_reference=source_reference,
                b2_path=b2_entry.file_name,
                source_video_id=source.source_video_id,
            )
            return source, b2_entry, None

    try:
        source_file_content = await runtime.file_tool.fetch_file_by_name(
            file_name=b2_entry.file_name
        )
    except Exception as exc:
        raise RetrievalError(f"B2 file download failed for {b2_entry.file_name!r}: {exc}") from exc

    return None, b2_entry, source_file_content


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]

    doctrine_patch: dict[str, Any] = {}
    if not settings.require_brand_doctrine and state.get("brand_doctrine") is None:
        doctrine_patch["brand_doctrine"] = default_brand_doctrine_dict(
            runtime.client_id,
            permissive=True,
        )

    query = state.get("refined_query") or state.get("user_query") or ""
    iteration_count = state.get("iteration_count", 0)
    prior_critique = state.get("critique_result")
    source_reference = state.get("source_reference")

    # Cap check: if we've already done max iterations, force-approve with existing critique.
    if iteration_count >= settings.max_critique_iterations and prior_critique is not None:
        logger.info(
            "retrieve_forced_approve_max_iter",
            iteration_count=iteration_count,
            session_id=state.get("session_id"),
        )
        return {
            **doctrine_patch,
            "critique_result": {
                **prior_critique,
                "verdict": "approved",
                "forced": True,
                "note": _FORCED_NOTE,
            },
        }

    if source_reference:
        source_video_id = state.get("source_video_id")
        source_patch: dict[str, Any] = {}

        if not source_video_id:
            session_document = _find_session_document(
                list(state.get("session_documents") or []),
                source_reference,
            )
            if session_document is not None:
                content = str(
                    session_document.get("content_text")
                    or session_document.get("summary")
                    or ""
                ).strip()
                filename = str(session_document.get("filename") or source_reference)
                document_id = str(session_document.get("doc_id") or filename)
                attachment_id = f"attachment:{document_id}"
                segment = {
                    "segment_id": f"{attachment_id}:full",
                    "video_id": attachment_id,
                    "text": content,
                    "start_seconds": 0.0,
                    "end_seconds": 0.0,
                    "has_timestamps": False,
                    "score": 1.0,
                    "metadata": {
                        "source_file": filename,
                        "session_attachment": True,
                    },
                }
                logger.info(
                    "source_reference_resolved_from_session_document",
                    source_reference=source_reference,
                    filename=filename,
                    session_id=state.get("session_id"),
                )
                return {
                    **doctrine_patch,
                    "source_video_id": attachment_id,
                    "source_metadata": {
                        "source_video_id": attachment_id,
                        "title": _without_extension(filename),
                        "source_file": filename,
                        "has_timestamps": False,
                        "metadata": {"session_attachment": True},
                    },
                    "source_resolution_error": None,
                    "source_file_content": content,
                    "retrieved_segments": [segment],
                    "previous_segment_ids": [segment["segment_id"]],
                }

            try:
                source, b2_entry, b2_source_content = await _resolve_source_video(
                    runtime,
                    source_reference,
                )
            except Exception as exc:
                raise RetrievalError(
                    f"Source lookup failed for {source_reference!r}: {exc}"
                ) from exc

            if source is None:
                if b2_entry is not None and b2_source_content:
                    source_video_id = f"b2:{b2_entry.file_id or b2_entry.file_name}"
                    source_patch = {
                        "source_video_id": source_video_id,
                        "source_metadata": {
                            "source_video_id": source_video_id,
                            "title": _without_extension(posixpath.basename(b2_entry.file_name)),
                            "source_file": posixpath.basename(b2_entry.file_name),
                            "has_timestamps": False,
                            "metadata": {
                                "b2_path": b2_entry.file_name,
                                "b2_fallback": True,
                            },
                        },
                        "source_resolution_error": None,
                    }
                    segment = {
                        "segment_id": f"{source_video_id}:full",
                        "video_id": source_video_id,
                        "text": b2_source_content,
                        "start_seconds": 0.0,
                        "end_seconds": 0.0,
                        "has_timestamps": False,
                        "score": 1.0,
                        "metadata": {
                            "source_file": posixpath.basename(b2_entry.file_name),
                            "b2_path": b2_entry.file_name,
                            "b2_fallback": True,
                        },
                    }
                    logger.info(
                        "retrieve_source_from_b2_text_completed",
                        source_reference=source_reference,
                        b2_path=b2_entry.file_name,
                        session_id=state.get("session_id"),
                    )
                    return {
                        **doctrine_patch,
                        **source_patch,
                        "source_file_content": b2_source_content,
                        "retrieved_segments": [segment],
                        "previous_segment_ids": [segment["segment_id"]],
                    }

                if b2_entry is not None:
                    message = (
                        f"I found a matching Backblaze B2 file for '{source_reference}' "
                        f"at '{b2_entry.file_name}', but I couldn't find indexed transcript "
                        "chunks for it. Re-run transcript indexing or check that the "
                        "transcript metadata uses the same source title/file."
                    )
                else:
                    message = (
                        f"I couldn't find a source video or file matching "
                        f"'{source_reference}'. Try the exact title or filename."
                    )
                logger.info(
                    "source_reference_not_found",
                    source_reference=source_reference,
                    b2_path=b2_entry.file_name if b2_entry is not None else None,
                    session_id=state.get("session_id"),
                )
                return {
                    **doctrine_patch,
                    "source_video_id": None,
                    "source_metadata": None,
                    "source_resolution_error": message,
                    "retrieved_segments": [],
                    "previous_segment_ids": [],
                }

            source_video_id = source.source_video_id
            source_patch = {
                "source_video_id": source.source_video_id,
                "source_metadata": dataclasses.asdict(source),
                "source_resolution_error": None,
            }

        try:
            hits = await runtime.search_tool.list_transcript_segments_for_video(
                source_video_id=source_video_id
            )
        except Exception as exc:
            raise RetrievalError(
                f"Transcript lookup failed for source {source_video_id!r}: {exc}"
            ) from exc

        segments = [dataclasses.asdict(h) for h in hits]
        top5_ids = [h.segment_id for h in hits[:5]]

        logger.info(
            "retrieve_source_completed",
            source_reference=source_reference,
            source_video_id=source_video_id,
            hit_count=len(hits),
            session_id=state.get("session_id"),
        )

        return {
            **doctrine_patch,
            **source_patch,
            "retrieved_segments": segments,
            "previous_segment_ids": top5_ids,
        }

    try:
        embedding = await runtime.embedder.embed(query)
    except Exception as exc:
        raise RetrievalError(f"Embedding failed: {exc}") from exc

    try:
        hits = await runtime.search_tool.search_transcripts(
            query_text=query,
            query_embedding=embedding,
            match_count=settings.retrieve_match_count,
        )
    except Exception as exc:
        raise RetrievalError(f"Transcript search failed: {exc}") from exc

    segments = [dataclasses.asdict(h) for h in hits]
    top5_ids = [h.segment_id for h in hits[:5]]
    prev_ids = state.get("previous_segment_ids") or []

    logger.info(
        "retrieve_completed",
        hit_count=len(hits),
        top5_ids=top5_ids,
        session_id=state.get("session_id"),
    )

    # Staleness check: same top-5 and we already have a critique → force-approve.
    if top5_ids == prev_ids[:5] and prior_critique is not None:
        logger.info("retrieve_forced_approve_stale", session_id=state.get("session_id"))
        return {
            **doctrine_patch,
            "retrieved_segments": segments,
            "previous_segment_ids": top5_ids,
            "critique_result": {
                **prior_critique,
                "verdict": "approved",
                "forced": True,
                "note": _FORCED_NOTE,
            },
        }

    return {
        **doctrine_patch,
        "retrieved_segments": segments,
        "previous_segment_ids": top5_ids,
    }
