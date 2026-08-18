from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator import main
from tests.conftest import FAKE_CLIENT_ID, FAKE_USER_ID


def _maybe_single_query(data: object) -> MagicMock:
    response = MagicMock(data=data)
    query = MagicMock()
    query.select.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
        return_value=response
    )
    return query


@pytest.mark.asyncio
async def test_onboarding_status_reports_missing_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = MagicMock()
    svc.table.return_value = _maybe_single_query(None)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(svc=svc)),
        headers={"Authorization": "Bearer token"},
    )
    monkeypatch.setattr(main, "verify_token", AsyncMock(return_value=FAKE_USER_ID))

    result = await main.get_onboarding_status(request)  # type: ignore[arg-type]

    assert result == {"has_workspace": False, "user_id": str(FAKE_USER_ID)}


@pytest.mark.asyncio
async def test_onboarding_status_returns_existing_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile_query = _maybe_single_query(
        {
            "client_id": str(FAKE_CLIENT_ID),
            "role": "admin",
            "display_name": "Taylor",
        }
    )
    client_query = _maybe_single_query(
        {"display_name": "Taylor Studio", "slug": "taylor-studio-00000000"}
    )
    svc = MagicMock()
    svc.table.side_effect = [profile_query, client_query]
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(svc=svc)),
        headers={"Authorization": "Bearer token"},
    )
    monkeypatch.setattr(main, "verify_token", AsyncMock(return_value=FAKE_USER_ID))

    result = await main.get_onboarding_status(request)  # type: ignore[arg-type]

    assert result["has_workspace"] is True
    assert result["client_id"] == str(FAKE_CLIENT_ID)
    assert result["display_name"] == "Taylor Studio"
    assert result["role"] == "admin"


@pytest.mark.asyncio
async def test_create_workspace_uses_authenticated_user_and_isolated_slug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rpc_response = MagicMock(data=str(FAKE_CLIENT_ID))
    rpc_query = MagicMock()
    rpc_query.execute = AsyncMock(return_value=rpc_response)

    client_response = MagicMock(
        data={"display_name": "Taylor Studio", "slug": "taylor-studio-00000000"}
    )
    client_query = MagicMock()
    client_query.select.return_value.eq.return_value.single.return_value.execute = AsyncMock(
        return_value=client_response
    )

    svc = MagicMock()
    svc.rpc.return_value = rpc_query
    svc.table.return_value = client_query
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(svc=svc)),
        headers={"Authorization": "Bearer token"},
    )
    monkeypatch.setattr(main, "verify_token", AsyncMock(return_value=FAKE_USER_ID))

    result = await main.create_user_workspace(
        request,  # type: ignore[arg-type]
        main.CreateWorkspaceRequest(display_name="  Taylor Studio  "),
    )

    assert result["client_id"] == str(FAKE_CLIENT_ID)
    assert result["role"] == "admin"
    svc.rpc.assert_called_once_with(
        "provision_user_workspace",
        {
            "p_user_id": str(FAKE_USER_ID),
            "p_slug": "taylor-studio-00000000-0000-0000-0000-000000000002",
            "p_display_name": "Taylor Studio",
        },
    )


def test_workspace_slug_is_stable_and_url_safe() -> None:
    assert main._workspace_slug("  My Cool Workspace!  ", FAKE_USER_ID) == (
        "my-cool-workspace-00000000-0000-0000-0000-000000000002"
    )
