from __future__ import annotations

import httpx
import pytest

from orchestrator import drive_cleanup


@pytest.mark.asyncio
async def test_trash_drive_files_calls_private_webhook_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requests: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"ok": True, "trashed": 2})

    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(drive_cleanup.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(
        drive_cleanup.settings,
        "media_deletion_webhook_url",
        "https://n8n.example.test/webhook/delete-media-drive-files",
    )
    monkeypatch.setattr(drive_cleanup.settings, "media_deletion_webhook_secret", "secret")

    deleted = await drive_cleanup.trash_drive_files(["transcript-1", "summary-1", "transcript-1"])

    assert deleted == 2
    assert len(requests) == 1
    assert requests[0].headers["X-VP-Media-Delete-Secret"] == "secret"
    assert requests[0].content == b'{"file_ids":["transcript-1","summary-1"]}'


@pytest.mark.asyncio
async def test_trash_drive_files_rejects_incomplete_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def handle(_: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True, "trashed": 1})

    real_client = httpx.AsyncClient

    def client_factory(**kwargs):
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(drive_cleanup.httpx, "AsyncClient", client_factory)
    monkeypatch.setattr(
        drive_cleanup.settings,
        "media_deletion_webhook_url",
        "https://n8n.example.test/webhook/delete-media-drive-files",
    )

    with pytest.raises(drive_cleanup.DriveCleanupError, match="incomplete"):
        await drive_cleanup.trash_drive_files(["transcript-1", "summary-1"])
