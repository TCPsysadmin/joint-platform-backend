from __future__ import annotations

from langchain_core.messages import HumanMessage

from orchestrator.state import AgentState


def test_agent_state_minimal_construction() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="hello")],
        "session_id": "sess-001",
        "intent": None,
        "follow_up_reuse": False,
        "command": None,
        "expand_seconds": 0,
        "user_query": None,
        "source_reference": None,
        "source_task": None,
        "source_video_id": None,
        "source_metadata": None,
        "source_resolution_error": None,
        "session_documents": [],
        "refined_query": None,
        "iteration_count": 0,
        "previous_segment_ids": [],
        "retrieved_segments": [],
        "brand_doctrine": None,
        "candidate_clips": [],
        "candidate_recommendation": None,
        "critique_result": None,
        "final_recommendation": None,
        "awaiting_confirmation": False,
        "confirmation_response": None,
    }

    assert state["session_id"] == "sess-001"
    assert state["iteration_count"] == 0
    assert state["awaiting_confirmation"] is False
    assert state["messages"][0].content == "hello"


def test_agent_state_intent_literals() -> None:
    for intent in ("new_request", "follow_up", "chitchat"):
        state: AgentState = {
            "messages": [],
            "session_id": "s",
            "intent": intent,  # type: ignore[typeddict-item]
            "follow_up_reuse": False,
            "command": None,
            "expand_seconds": 0,
            "user_query": "test",
            "source_reference": None,
            "source_task": None,
            "source_video_id": None,
            "source_metadata": None,
            "source_resolution_error": None,
            "session_documents": [],
            "refined_query": None,
            "iteration_count": 0,
            "previous_segment_ids": [],
            "retrieved_segments": [],
            "brand_doctrine": None,
            "candidate_clips": [],
            "candidate_recommendation": None,
            "critique_result": None,
            "final_recommendation": None,
            "awaiting_confirmation": False,
            "confirmation_response": None,
        }
        assert state["intent"] == intent


def test_agent_state_all_fields_json_serialisable() -> None:
    import json

    state: AgentState = {
        "messages": [],
        "session_id": "sess-002",
        "intent": "new_request",
        "follow_up_reuse": False,
        "command": None,
        "expand_seconds": 0,
        "user_query": "best hook",
        "source_reference": "founder-story.mp4",
        "source_task": "clip_recommendation",
        "source_video_id": "vid-001",
        "source_metadata": {"title": "Founder Story", "source_file": "founder-story.mp4"},
        "source_resolution_error": None,
        "session_documents": [
            {
                "doc_id": "doc-1",
                "filename": "brief.md",
                "content_text": "Focus on customer proof.",
            }
        ],
        "refined_query": "emotional hook moments",
        "iteration_count": 2,
        "previous_segment_ids": ["seg-1", "seg-2"],
        "retrieved_segments": [{"segment_id": "seg-1", "text": "hello"}],
        "brand_doctrine": {"name": "Test", "rubric": []},
        "candidate_clips": [{"video_id": "v1", "hook_quote": "hi"}],
        "candidate_recommendation": {"video_id": "v1", "hook_quote": "hi"},
        "critique_result": {"weighted_score": 0.8, "verdict": "approved"},
        "final_recommendation": "## Recommendation\n...",
        "awaiting_confirmation": True,
        "confirmation_response": "approved",
    }

    serialisable = {k: v for k, v in state.items() if k != "messages"}
    dumped = json.dumps(serialisable)
    reloaded = json.loads(dumped)
    assert reloaded["session_id"] == "sess-002"
    assert reloaded["iteration_count"] == 2
    assert reloaded["source_video_id"] == "vid-001"
    assert reloaded["source_task"] == "clip_recommendation"
    assert reloaded["session_documents"][0]["filename"] == "brief.md"
