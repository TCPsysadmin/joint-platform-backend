from __future__ import annotations

import pytest

from orchestrator.tools.local.b2_file import _is_text_file, _normalize_file_match_text
from orchestrator.tools.local.doctrine import SupabaseDoctrineTool
from orchestrator.tools.local.opusclip_stub import OpusClipStub
from orchestrator.tools.local.supabase_search import SupabaseSearchTool
from orchestrator.tools.protocols import (
    ClipPayload,
    DoctrineTool,
    PublishResult,
    PublishTool,
    SearchTool,
)
from tests.conftest import FAKE_CLIENT_ID

# ── Protocol conformance (structural) ─────────────────────────────────────────


def test_supabase_search_tool_satisfies_protocol() -> None:
    from unittest.mock import MagicMock

    tool = SupabaseSearchTool(supabase=MagicMock(), client_id=FAKE_CLIENT_ID)
    assert isinstance(tool, SearchTool)


def test_supabase_doctrine_tool_satisfies_protocol() -> None:
    from unittest.mock import MagicMock

    tool = SupabaseDoctrineTool(supabase=MagicMock(), client_id=FAKE_CLIENT_ID)
    assert isinstance(tool, DoctrineTool)


def test_opusclip_stub_satisfies_protocol() -> None:
    tool = OpusClipStub()
    assert isinstance(tool, PublishTool)


def test_b2_text_detection_skips_binary_audio() -> None:
    assert _is_text_file("TCP001_DITL_20250502 - JUST TRY.mp3", "audio/mpeg") is False
    assert _is_text_file("transcripts/TCP001_DITL_20250502 - JUST TRY.txt", "") is True
    assert _is_text_file("transcripts/source.vtt", "application/octet-stream") is True


def test_b2_filename_matching_normalizes_extension_and_punctuation() -> None:
    assert _normalize_file_match_text(
        "TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr. Kieschnick"
    ) == _normalize_file_match_text(
        "TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr Kieschnick.mp3"
    )


# ── OpusClipStub functional test (no real I/O) ───────────────────────────────


@pytest.mark.asyncio
async def test_opusclip_stub_returns_publish_result() -> None:
    stub = OpusClipStub()
    payload = ClipPayload(
        video_id="vid-123",
        start_seconds=10.0,
        end_seconds=45.0,
        hook_quote="This changes everything.",
        broll_suggestions=[],
    )
    result = await stub.create_and_post_clip(payload)
    assert isinstance(result, PublishResult)
    assert result.status == "stub_posted"
    assert result.clip_id.startswith("stub-")
    assert result.url is None
    assert result.metadata.get("stub") is True


# ── Embedder protocol ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_mock_embedder_returns_embedding(mock_embedder: object) -> None:
    result = await mock_embedder.embed("test text")  # type: ignore[union-attr]
    assert isinstance(result, list)
    assert len(result) == 1536
