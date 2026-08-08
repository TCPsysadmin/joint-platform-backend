from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    session_id: str
    intent: Literal["new_request", "follow_up", "chitchat"] | None
    # For follow_up only: True if the existing retrieved chunks + transcript can
    # answer the follow-up (cheap re-analysis), False if a fresh retrieval is needed.
    follow_up_reuse: bool
    user_query: str | None
    # Optional user-specified video/file reference. When present, retrieval is
    # constrained to the resolved source instead of searching across all chunks.
    source_reference: str | None
    # Source-specific task mode. "source_answer" means answer a question about
    # the resolved source itself instead of recommending clips from it.
    source_task: Literal["clip_recommendation", "source_answer"] | None
    source_video_id: str | None
    source_metadata: dict[str, object] | None
    source_resolution_error: str | None
    session_documents: list[dict[str, object]]
    refined_query: str | None
    iteration_count: int
    previous_segment_ids: list[str]
    retrieved_segments: list[dict[str, object]]
    brand_doctrine: dict[str, object] | None
    source_file_content: str | None
    # Which source `source_file_content` was ingested for. Lets fetch_source tell
    # "already have this transcript" from "have a different video's transcript",
    # so "dive deeper on clip 2" can pull clip 2's source when it is not in yet.
    # None means unknown provenance (legacy checkpoint) — treated as a match.
    source_content_video_id: str | None
    # Ranked list of clip candidates surfaced by analyze (best-first). The primary
    # (candidate_clips[0]) is mirrored into candidate_recommendation for critique/post.
    candidate_clips: list[dict[str, object]]
    candidate_recommendation: dict[str, object] | None
    critique_result: dict[str, object] | None
    final_recommendation: str | None
    awaiting_confirmation: bool
    confirmation_response: Literal["approved", "rejected"] | None
