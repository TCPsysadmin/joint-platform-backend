from __future__ import annotations

from typing import Any

import pytest

from orchestrator import session_manager
from orchestrator.tools.local.supabase_search import SupabaseSearchTool
from tests.conftest import FAKE_CLIENT_ID, FAKE_USER_ID


class _Response:
    def __init__(self, data: Any = None) -> None:
        self.data = [] if data is None else data


class _RpcRequest:
    def __init__(self, data: Any = None) -> None:
        self._data = data

    async def execute(self) -> _Response:
        return _Response(self._data)


class _RpcClient:
    def __init__(self, response_data: Any = None) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.response_data = response_data

    def rpc(self, name: str, params: dict[str, Any]) -> _RpcRequest:
        self.calls.append((name, params))
        return _RpcRequest(self.response_data)


@pytest.mark.asyncio
async def test_transcript_search_passes_resolved_client_id_to_rpc() -> None:
    supabase = _RpcClient()
    tool = SupabaseSearchTool(
        supabase=supabase,  # type: ignore[arg-type]
        client_id=FAKE_CLIENT_ID,
    )

    await tool.search_transcripts(
        query_text="pricing",
        query_embedding=[0.0] * 1536,
    )

    assert supabase.calls == [
        (
            "hybrid_search_transcripts",
            {
                "p_client_id": str(FAKE_CLIENT_ID),
                "p_query_text": "pricing",
                "p_query_embedding": [0.0] * 1536,
                "p_match_count": 40,
                "p_min_seconds": 0.0,
            },
        )
    ]


@pytest.mark.asyncio
async def test_asset_search_passes_resolved_client_id_to_rpc() -> None:
    supabase = _RpcClient()
    tool = SupabaseSearchTool(
        supabase=supabase,  # type: ignore[arg-type]
        client_id=FAKE_CLIENT_ID,
    )

    await tool.match_assets(
        query_embedding=[0.0] * 1536,
        asset_types=["broll"],
        match_count=3,
    )

    assert supabase.calls == [
        (
            "match_assets",
            {
                "p_client_id": str(FAKE_CLIENT_ID),
                "p_query_embedding": [0.0] * 1536,
                "p_match_count": 3,
                "p_asset_types": ["broll"],
            },
        )
    ]


@pytest.mark.asyncio
async def test_session_archive_passes_user_and_client_to_rpc() -> None:
    supabase = _RpcClient(response_data=True)

    archived = await session_manager.archive_session(
        supabase,  # type: ignore[arg-type]
        "session-123",
        FAKE_USER_ID,
        FAKE_CLIENT_ID,
    )

    assert archived is True
    assert supabase.calls == [
        (
            "archive_session",
            {
                "p_session_id": "session-123",
                "p_user_id": str(FAKE_USER_ID),
                "p_client_id": str(FAKE_CLIENT_ID),
            },
        )
    ]
