from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock
from uuid import UUID

import pytest
from langchain_core.messages import AIMessage

from orchestrator.runtime import RuntimeContext
from orchestrator.tools.protocols import (
    AssetHit,
    Doctrine,
    PublishResult,
    TranscriptHit,
)

_FIXTURES = Path(__file__).parent / "fixtures"

FAKE_CLIENT_ID = UUID("00000000-0000-0000-0000-000000000001")
FAKE_USER_ID = UUID("00000000-0000-0000-0000-000000000002")

SAMPLE_SEGMENTS: list[dict[str, Any]] = json.loads((_FIXTURES / "sample_segments.json").read_text())


@pytest.fixture
def sample_hits() -> list[TranscriptHit]:
    return [
        TranscriptHit(
            segment_id=s["segment_id"],
            video_id=s["video_id"],
            text=s["text"],
            start_seconds=s["start_seconds"],
            end_seconds=s["end_seconds"],
            has_timestamps=s["has_timestamps"],
            score=s["score"],
        )
        for s in SAMPLE_SEGMENTS
    ]


@pytest.fixture
def mock_search_tool(sample_hits: list[TranscriptHit]) -> AsyncMock:
    tool = AsyncMock()
    tool.search_transcripts = AsyncMock(return_value=sample_hits)
    tool.match_assets = AsyncMock(
        return_value=[
            AssetHit(
                asset_id="asset-001",
                asset_type="broll_prompt",
                description="Creator looking into camera, confident expression",
                score=0.91,
            )
        ]
    )
    return tool


@pytest.fixture
def mock_doctrine_tool() -> AsyncMock:
    tool = AsyncMock()
    tool.get_active_doctrine = AsyncMock(
        return_value=Doctrine(
            client_id=FAKE_CLIENT_ID,
            name="Test Brand",
            rubric=[
                {
                    "dimension": "hook_strength",
                    "weight": 0.5,
                    "guidance": "Hook must grab in 3 seconds.",
                    "requires_timestamps": False,
                },
                {
                    "dimension": "brand_alignment",
                    "weight": 0.5,
                    "guidance": "Must reflect brand values.",
                    "requires_timestamps": False,
                },
            ],
            minimum_a_tier_score=0.75,
            auto_reject_below=0.40,
        )
    )
    return tool


@pytest.fixture
def mock_publish_tool() -> AsyncMock:
    tool = AsyncMock()
    tool.create_and_post_clip = AsyncMock(
        return_value=PublishResult(
            clip_id="stub-test-clip",
            status="stub_posted",
            url=None,
        )
    )
    return tool


@pytest.fixture
def mock_embedder() -> AsyncMock:
    embedder = AsyncMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 1536)
    return embedder


@pytest.fixture
def mock_file_tool() -> AsyncMock:
    tool = AsyncMock()
    tool.fetch_source_file = AsyncMock(return_value=None)
    tool.get_download_url = AsyncMock(return_value=None)
    tool.list_buckets = AsyncMock(return_value=[])
    tool.list_file_names = AsyncMock(return_value=([], None))
    tool.list_file_versions = AsyncMock(return_value=([], None, None))
    tool.list_keys = AsyncMock(return_value=([], None))
    tool.find_file_by_name = AsyncMock(return_value=None)
    return tool


def _make_llm_mock(*responses: str) -> AsyncMock:
    """Create an LLM mock that returns each response string in order."""
    llm = AsyncMock()
    llm.ainvoke = AsyncMock(side_effect=[AIMessage(content=r) for r in responses])
    return llm


@pytest.fixture
def mock_llm_chitchat() -> AsyncMock:
    return _make_llm_mock(
        '{"intent": "chitchat", "user_query": null}',
        "Hey! I'm here to help you find great clips. What are you looking for?",
    )


@pytest.fixture
def mock_llm_full_flow() -> AsyncMock:
    """LLM mock wired for: route_intent → analyze → critique (approved) → recommend."""
    return _make_llm_mock(
        # route_intent
        '{"intent": "new_request", "user_query": "best hook moments about consistency"}',
        # analyze — returns a ranked list of clip candidates
        json.dumps(
            {
                "clips": [
                    {
                        "video_id": "vid-abc",
                        "segment_id": "seg-001",
                        "start_seconds": 42.5,
                        "end_seconds": 53.1,
                        "has_timestamps": True,
                        "hook_quote": (
                            "The biggest mistake most creators make is they try to "
                            "appeal to everyone"
                        ),
                        "rationale": "Strong contrarian hook that challenges conventional wisdom.",
                    },
                    {
                        "video_id": "vid-abc",
                        "segment_id": "seg-002",
                        "start_seconds": 88.0,
                        "end_seconds": 99.0,
                        "has_timestamps": True,
                        "hook_quote": "Consistency beats intensity every single time",
                        "rationale": "Punchy, memorable, on-doctrine.",
                    },
                ]
            }
        ),
        # critique
        json.dumps(
            {
                "dimension_scores": [
                    {
                        "dimension": "hook_strength",
                        "score": 0.9,
                        "rationale": "Excellent pattern interrupt.",
                    },
                    {
                        "dimension": "brand_alignment",
                        "score": 0.85,
                        "rationale": "Aligned with brand voice.",
                    },
                ],
                "weighted_score": 0.875,
                "overall_rationale": "Strong candidate.",
                "improvement_suggestions": [],
            }
        ),
        # recommend
        "## Your Clip Recommendation\n\n> The biggest mistake most creators make...\n\n**Score: 87.5%**",
    )


@pytest.fixture
def make_runtime(
    mock_search_tool: AsyncMock,
    mock_doctrine_tool: AsyncMock,
    mock_publish_tool: AsyncMock,
    mock_file_tool: AsyncMock,
    mock_embedder: AsyncMock,
) -> Any:
    def _make(llm: AsyncMock) -> RuntimeContext:
        return RuntimeContext(
            client_id=FAKE_CLIENT_ID,
            user_id=FAKE_USER_ID,
            search_tool=mock_search_tool,
            doctrine_tool=mock_doctrine_tool,
            publish_tool=mock_publish_tool,
            file_tool=mock_file_tool,
            llm=llm,
            embedder=mock_embedder,
        )

    return _make
