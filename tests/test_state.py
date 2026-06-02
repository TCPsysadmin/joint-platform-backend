from __future__ import annotations

from langchain_core.messages import HumanMessage

from orchestrator.state import AgentState


def test_agent_state_minimal_construction() -> None:
    state: AgentState = {
        "messages": [HumanMessage(content="hello")],
        "session_id": "sess-001",
        "intent": None,
        "follow_up_reuse": False,
        "user_query": None,
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
            "user_query": "test",
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
        "user_query": "best hook",
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
