from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from orchestrator import main, media_library
from tests.conftest import FAKE_CLIENT_ID


def test_folder_payload_groups_video_summary_and_thumbnail() -> None:
    payload = media_library._folder_payload(
        {
            "source_video_id": "vid-123",
            "title": "Founder interview",
            "source_file": "founder.mp4",
            "b2_path": "videos/founder.mp4",
            "thumbnail_url": "https://cdn.example.com/founder.jpg",
            "summary_text": "A conversation about product growth.",
            "topics": ["growth"],
            "speakers": ["Alex"],
        }
    )

    assert payload["kind"] == "video_folder"
    assert payload["name"] == "Founder interview"
    assert payload["thumbnail"]["url"].endswith("founder.jpg")
    assert payload["video"]["source_file"] == "founder.mp4"
    assert payload["summary"]["topics"] == ["growth"]


@pytest.mark.asyncio
async def test_list_media_is_tenant_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = object()
    file_tool = SimpleNamespace(
        get_path_download_urls=AsyncMock(return_value={})
    )
    runtime = SimpleNamespace(client_id=FAKE_CLIENT_ID, file_tool=file_tool)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(svc=svc)),
        headers={"authorization": "Bearer token"},
    )
    list_videos = AsyncMock(return_value=[{"id": "vid-123"}])

    monkeypatch.setattr(main, "resolve_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(main.media_library, "list_videos", list_videos)

    response = await main.list_media(request, limit=25, offset=10)  # type: ignore[arg-type]

    assert response["items"] == [{"id": "vid-123"}]
    assert response["count"] == 1
    list_videos.assert_awaited_once_with(
        svc,
        client_id=FAKE_CLIENT_ID,
        limit=25,
        offset=10,
    )


@pytest.mark.asyncio
async def test_get_media_returns_open_folder(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = object()
    file_tool = SimpleNamespace(
        get_path_download_urls=AsyncMock(
            return_value={"thumbnails/vid-123.webp": "https://download.example/thumb.webp"}
        )
    )
    runtime = SimpleNamespace(client_id=FAKE_CLIENT_ID, file_tool=file_tool)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(svc=svc)),
        headers={"authorization": "Bearer token"},
    )
    get_video = AsyncMock(
        return_value={
            "id": "vid-123",
            "thumbnail": {"url": None, "b2_path": "thumbnails/vid-123.webp"},
            "transcript": {"segment_count": 1, "text": "Hello"},
        }
    )

    monkeypatch.setattr(main, "resolve_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(main.media_library, "get_video", get_video)

    response = await main.get_media("vid-123", request)  # type: ignore[arg-type]

    assert response["transcript"]["segment_count"] == 1
    assert response["thumbnail"]["url"] == "https://download.example/thumb.webp"
    get_video.assert_awaited_once_with(
        svc,
        client_id=FAKE_CLIENT_ID,
        source_video_id="vid-123",
    )
    file_tool.get_path_download_urls.assert_awaited_once()


@pytest.mark.asyncio
async def test_get_media_video_url_is_tenant_scoped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = object()
    file_tool = SimpleNamespace(
        get_download_url=AsyncMock(return_value="https://download.example/video.mp4")
    )
    runtime = SimpleNamespace(client_id=FAKE_CLIENT_ID, file_tool=file_tool)
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(svc=svc)))
    get_video = AsyncMock(
        return_value={
            "id": "vid-123",
            "video": {"source_file": "interview.mp4"},
        }
    )

    monkeypatch.setattr(main, "resolve_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(main.media_library, "get_video", get_video)

    response = await main.get_media_video_url(
        "vid-123",
        request,  # type: ignore[arg-type]
    )

    assert response["url"] == "https://download.example/video.mp4"
    assert response["filename"] == "interview.mp4"
    get_video.assert_awaited_once_with(
        svc,
        client_id=FAKE_CLIENT_ID,
        source_video_id="vid-123",
    )
    file_tool.get_download_url.assert_awaited_once()
