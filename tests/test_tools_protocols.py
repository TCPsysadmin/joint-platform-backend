from __future__ import annotations

import pytest

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
