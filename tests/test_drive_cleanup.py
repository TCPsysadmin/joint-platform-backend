from __future__ import annotations

from unittest.mock import AsyncMock

import httpx
import pytest

from orchestrator import drive_cleanup


@pytest.mark.asyncio
async def test_trash_drive_files_calls_drive_api_for_unique_ids(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"id": request.url.path.rsplit("/", 1)[-1], "trashed": True})

    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(drive_cleanup.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(drive_cleanup, "_access_token", AsyncMock(return_value="token"))

    deleted = await drive_cleanup.trash_drive_files(
        ["transcript-1", "summary-1", "transcript-1"]
    )

    assert deleted == 2
    assert len(requests) == 2
    assert [request.method for request in requests] == ["PATCH", "PATCH"]
    assert requests[0].headers["Authorization"] == "Bearer token"
    assert requests[0].url.params["supportsAllDrives"] == "true"
    assert requests[0].content == b'{"trashed":true}'


@pytest.mark.asyncio
async def test_trash_drive_files_rejects_unconfirmed_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"id": "transcript-1", "trashed": False})

    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(drive_cleanup.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(drive_cleanup, "_access_token", AsyncMock(return_value="token"))

    with pytest.raises(drive_cleanup.DriveCleanupError, match="did not confirm"):
        await drive_cleanup.trash_drive_files(["transcript-1", "summary-1"])


@pytest.mark.asyncio
async def test_trash_drive_files_treats_missing_file_as_already_removed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": {"message": "not found"}})

    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(drive_cleanup.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(drive_cleanup, "_access_token", AsyncMock(return_value="token"))

    assert await drive_cleanup.trash_drive_files(["already-gone"]) == 1


def test_credentials_require_valid_json(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(drive_cleanup.settings, "google_drive_service_account_json", "not-json")

    with pytest.raises(drive_cleanup.DriveCleanupError, match="invalid"):
        drive_cleanup._credentials()
