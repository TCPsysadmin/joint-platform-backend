from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from orchestrator.errors import LLMError
from orchestrator.prompts import load_prompt
from orchestrator.runtime import RuntimeContext
from orchestrator.state import AgentState

logger = structlog.get_logger(__name__)


def _segments_as_context(segments: list[dict[str, object]]) -> str:
    lines: list[str] = []
    for segment in segments:
        text = str(segment.get("text") or "").strip()
        if not text:
            continue
        if segment.get("has_timestamps"):
            start = segment.get("start_seconds")
            end = segment.get("end_seconds")
            lines.append(f"[{start}-{end}] {text}")
        else:
            lines.append(text)
    return "\n\n".join(lines)


async def run(state: AgentState, config: RunnableConfig) -> dict[str, Any]:
    runtime: RuntimeContext = config["configurable"]["runtime"]
    segments = list(state.get("retrieved_segments") or [])
    source_file_content = state.get("source_file_content") or _segments_as_context(segments)

    if not source_file_content.strip():
        source_reference = state.get("source_reference") or "that source"
        message = (
            f"I found `{source_reference}`, but I do not have enough transcript "
            "or file content available to tell what it is about."
        )
        logger.info(
            "source_answer_no_content",
            source_reference=state.get("source_reference"),
            source_video_id=state.get("source_video_id"),
            session_id=state.get("session_id"),
        )
        return {
            "messages": [AIMessage(content=message)],
            "final_recommendation": message,
            "awaiting_confirmation": False,
        }

    payload: dict[str, object] = {
        "user_query": state.get("refined_query") or state.get("user_query") or "",
        "source_reference": state.get("source_reference"),
        "source_video_id": state.get("source_video_id"),
        "source_metadata": state.get("source_metadata") or {},
        "source_file_content": source_file_content,
        "segments": segments,
    }

    messages = [
        SystemMessage(content=load_prompt("source_answer.md")),
        HumanMessage(content=json.dumps(payload, default=str)),
    ]

    response = await runtime.llm.ainvoke(messages)
    answer = str(response.content).strip()
    if not answer:
        raise LLMError("source_answer returned empty content")

    logger.info(
        "source_answer_generated",
        source_reference=state.get("source_reference"),
        source_video_id=state.get("source_video_id"),
        session_id=state.get("session_id"),
    )

    return {
        "messages": [AIMessage(content=answer)],
        "final_recommendation": answer,
        "awaiting_confirmation": False,
    }
