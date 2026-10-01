from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from orchestrator import main, media_library
from tests.conftest import FAKE_CLIENT_ID

DESTINATION_CLIENT_ID = "11111111-2222-4333-8444-555555555555"


def _query_response(data: dict[str, object]) -> MagicMock:
    response = MagicMock()
    response.data = data
    query = MagicMock()
    query.select.return_value.eq.return_value.maybe_single.return_value.execute = AsyncMock(
        return_value=response
    )
    return query


@pytest.mark.asyncio
async def test_ingestion_config_is_resolved_from_authenticated_user(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    query = _query_response(
        {
            "client_id": str(FAKE_CLIENT_ID),
            "display_name": "Test Company",
            "b2_bucket": "vpstorage-test",
            "drive_transcripts_intake_folder_id": "transcripts-in",
            "drive_summaries_intake_folder_id": "summaries-in",
            "drive_transcripts_completed_folder_id": "transcripts-done",
            "drive_summaries_completed_folder_id": "summaries-done",
        }
    )
    svc = MagicMock()
    svc.table.return_value = query
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(svc=svc)),
        headers={"Authorization": "Bearer token"},
    )
    monkeypatch.setattr(main, "verify_token", AsyncMock(return_value="user-id"))
    monkeypatch.setattr(main, "get_client_id", AsyncMock(return_value=FAKE_CLIENT_ID))

    response = await main.get_ingestion_config(request)  # type: ignore[arg-type]

    assert response["client_id"] == str(FAKE_CLIENT_ID)
    assert response["name"] == "Test Company"
    assert response["b2_bucket"] == "vpstorage-test"
    assert response["transcripts_folder_id"] == "transcripts-in"
    query.select.return_value.eq.assert_called_once_with("client_id", str(FAKE_CLIENT_ID))


@pytest.mark.asyncio
async def test_ingestion_destinations_only_returns_authenticated_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    response = MagicMock()
    response.data = [
        {
            "client_id": str(FAKE_CLIENT_ID),
            "display_name": "Test Company",
            "b2_bucket": "vpstorage-test",
            "drive_transcripts_intake_folder_id": "transcripts-in",
            "drive_summaries_intake_folder_id": "summaries-in",
            "drive_transcripts_completed_folder_id": "transcripts-done",
            "drive_summaries_completed_folder_id": "summaries-done",
        },
        {
            "client_id": "other-client",
            "display_name": "Other Company",
            "b2_bucket": "vpstorage-other",
            "drive_transcripts_intake_folder_id": "other-transcripts-in",
            "drive_summaries_intake_folder_id": "other-summaries-in",
            "drive_transcripts_completed_folder_id": "other-transcripts-done",
            "drive_summaries_completed_folder_id": "other-summaries-done",
        },
    ]
    query = MagicMock()
    tenant_query = query.select.return_value.eq.return_value
    tenant_query.eq.return_value.order.return_value.execute = AsyncMock(return_value=response)
    svc = MagicMock()
    svc.table.return_value = query
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(svc=svc)),
        headers={"Authorization": "Bearer token"},
    )
    monkeypatch.setattr(main, "verify_token", AsyncMock(return_value="user-id"))
    monkeypatch.setattr(main, "get_client_id", AsyncMock(return_value=FAKE_CLIENT_ID))

    result = await main.list_ingestion_destinations(request)  # type: ignore[arg-type]

    assert result == {
        "destinations": [
            {
                "client_id": str(FAKE_CLIENT_ID),
                "name": "Test Company",
                "b2_bucket": "vpstorage-test",
                "transcripts_folder_id": "transcripts-in",
                "summaries_folder_id": "summaries-in",
                "transcripts_completed_folder_id": "transcripts-done",
                "summaries_completed_folder_id": "summaries-done",
            }
        ]
    }
    query.select.return_value.eq.assert_called_once_with("client_id", str(FAKE_CLIENT_ID))
    tenant_query.eq.assert_called_once_with("status", "active")


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
        },
        transcript_segment_count=3,
    )

    assert payload["kind"] == "video_folder"
    assert payload["name"] == "Founder interview"
    assert payload["thumbnail"]["url"].endswith("founder.jpg")
    assert payload["video"]["source_file"] == "founder.mp4"
    assert payload["summary"]["topics"] == ["growth"]
    assert payload["transcript"] == {"available": True, "segment_count": 3}


def test_folder_payload_reports_a_missing_transcript() -> None:
    payload = media_library._folder_payload(
        {"source_video_id": "summary-only", "summary_text": "Existing summary"},
        transcript_segment_count=0,
    )

    assert payload["transcript"] == {"available": False, "segment_count": 0}


@pytest.mark.asyncio
async def test_list_media_is_tenant_scoped(monkeypatch: pytest.MonkeyPatch) -> None:
    svc = object()
    file_tool = SimpleNamespace(get_path_download_urls=AsyncMock(return_value={}))
    runtime = SimpleNamespace(client_id=FAKE_CLIENT_ID, file_tool=file_tool)
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(svc=svc)),
        headers={"authorization": "Bearer token"},
    )
    list_videos = AsyncMock(return_value=([{"id": "vid-123"}], 42))

    monkeypatch.setattr(main, "resolve_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(main.media_library, "list_videos", list_videos)

    response = await main.list_media(
        request,
        limit=25,
        offset=10,
        search="founder",
        sort="name-asc",
        content_filter="missing-summary",
    )  # type: ignore[arg-type]

    assert response["items"] == [{"id": "vid-123"}]
    assert response["count"] == 42
    list_videos.assert_awaited_once_with(
        svc,
        client_id=FAKE_CLIENT_ID,
        limit=25,
        offset=10,
        search="founder",
        sort="name-asc",
        content_filter="missing-summary",
    )


def test_media_content_filters_use_transcript_and_summary_availability() -> None:
    complete = {"summary_text": "Summary"}
    summary_only = {"summary_text": "Summary"}
    transcript_only = {"summary_text": None}

    assert media_library._matches_content_filter(
        complete, transcript_segment_count=2, content_filter="complete"
    )
    assert media_library._matches_content_filter(
        summary_only, transcript_segment_count=0, content_filter="missing-transcript"
    )
    assert media_library._matches_content_filter(
        transcript_only, transcript_segment_count=2, content_filter="missing-summary"
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


@pytest.mark.asyncio
async def test_bulk_delete_media_requires_manager_and_cleans_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = object()
    file_tool = SimpleNamespace(delete_paths_from_storage=AsyncMock(return_value=[]))
    runtime = SimpleNamespace(
        client_id=FAKE_CLIENT_ID,
        user_id="user-id",
        file_tool=file_tool,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(svc=svc)))
    records = [
        {
            "source_video_id": "video-1",
            "b2_path": "videos/video-1.mp4",
            "thumbnail_b2_path": "thumbnails/video-1.webp",
            "transcript_drive_file_id": "drive-transcript-1",
            "summary_drive_file_id": "drive-summary-1",
        }
    ]
    get_records = AsyncMock(return_value=records)
    delete_records = AsyncMock(return_value=1)
    trash_drive_files = AsyncMock(return_value=2)
    require_manager = AsyncMock(return_value={"role": "admin"})
    monkeypatch.setattr(main, "resolve_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(main, "_require_workspace_manager", require_manager)
    monkeypatch.setattr(main.media_library, "get_media_storage_records", get_records)
    monkeypatch.setattr(main.media_library, "delete_media_records", delete_records)
    monkeypatch.setattr(main.drive_cleanup, "trash_drive_files", trash_drive_files)

    result = await main.bulk_delete_media(
        main.BulkMediaDeleteRequest(source_video_ids=["video-1"]),
        request,  # type: ignore[arg-type]
    )

    assert result == {"deleted": 1, "source_video_ids": ["video-1"], "warnings": []}
    require_manager.assert_awaited_once()
    delete_records.assert_awaited_once_with(
        svc,
        client_id=FAKE_CLIENT_ID,
        source_video_ids=["video-1"],
    )
    file_tool.delete_paths_from_storage.assert_awaited_once_with(
        ["videos/video-1.mp4", "thumbnails/video-1.webp"]
    )
    trash_drive_files.assert_awaited_once_with(["drive-transcript-1", "drive-summary-1"])


@pytest.mark.asyncio
async def test_bulk_delete_media_keeps_everything_when_drive_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = object()
    file_tool = SimpleNamespace(delete_paths_from_storage=AsyncMock(return_value=[]))
    runtime = SimpleNamespace(
        client_id=FAKE_CLIENT_ID,
        user_id="user-id",
        file_tool=file_tool,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(svc=svc)))
    records = [
        {
            "source_video_id": "video-1",
            "b2_path": "videos/video-1.mp4",
            "transcript_drive_file_id": "drive-transcript-1",
            "summary_drive_file_id": "drive-summary-1",
        }
    ]
    delete_records = AsyncMock(return_value=1)
    monkeypatch.setattr(main, "resolve_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(main, "_require_workspace_manager", AsyncMock(return_value={}))
    monkeypatch.setattr(
        main.media_library,
        "get_media_storage_records",
        AsyncMock(return_value=records),
    )
    monkeypatch.setattr(main.media_library, "delete_media_records", delete_records)
    monkeypatch.setattr(
        main.drive_cleanup,
        "trash_drive_files",
        AsyncMock(side_effect=main.drive_cleanup.DriveCleanupError("n8n unavailable")),
    )

    with pytest.raises(main.HTTPException) as exc_info:
        await main.bulk_delete_media(
            main.BulkMediaDeleteRequest(source_video_ids=["video-1"]),
            request,  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 502
    file_tool.delete_paths_from_storage.assert_not_awaited()
    delete_records.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_delete_media_keeps_database_records_when_storage_cleanup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    svc = object()
    file_tool = SimpleNamespace(
        delete_paths_from_storage=AsyncMock(side_effect=RuntimeError("B2 unavailable"))
    )
    runtime = SimpleNamespace(
        client_id=FAKE_CLIENT_ID,
        user_id="user-id",
        file_tool=file_tool,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(svc=svc)))
    records = [
        {
            "source_video_id": "video-1",
            "b2_path": "videos/video-1.mp4",
            "thumbnail_b2_path": "thumbnails/video-1.webp",
        }
    ]
    delete_records = AsyncMock(return_value=1)
    monkeypatch.setattr(main, "resolve_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(main, "_require_workspace_manager", AsyncMock(return_value={}))
    monkeypatch.setattr(
        main.media_library,
        "get_media_storage_records",
        AsyncMock(return_value=records),
    )
    monkeypatch.setattr(main.media_library, "delete_media_records", delete_records)

    with pytest.raises(main.HTTPException) as exc_info:
        await main.bulk_delete_media(
            main.BulkMediaDeleteRequest(source_video_ids=["video-1"]),
            request,  # type: ignore[arg-type]
        )

    assert exc_info.value.status_code == 502
    assert exc_info.value.detail == (
        "Could not delete the selected videos from storage. Please retry."
    )
    delete_records.assert_not_awaited()


@pytest.mark.asyncio
async def test_bulk_move_media_copies_storage_then_moves_database(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination_response = MagicMock()
    destination_response.data = {
        "client_id": DESTINATION_CLIENT_ID,
        "b2_bucket": "destination-bucket",
        "b2_prefix": "workspace",
        "status": "active",
    }
    query = MagicMock()
    query.select.return_value.eq.return_value.eq.return_value.maybe_single.return_value.execute = (
        AsyncMock(return_value=destination_response)
    )
    svc = MagicMock()
    svc.table.return_value = query
    file_tool = SimpleNamespace(
        copy_path_to_storage=AsyncMock(return_value=True),
        delete_paths_from_storage=AsyncMock(return_value=[]),
    )
    runtime = SimpleNamespace(
        client_id=FAKE_CLIENT_ID,
        user_id="user-id",
        file_tool=file_tool,
    )
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(svc=svc)))
    records = [
        {
            "source_video_id": "video-1",
            "b2_path": "videos/video-1.mp4",
            "thumbnail_b2_path": "thumbnails/video-1.webp",
        }
    ]
    get_records = AsyncMock(side_effect=[records, []])
    move_records = AsyncMock(return_value=1)
    monkeypatch.setattr(main, "resolve_runtime", AsyncMock(return_value=runtime))
    monkeypatch.setattr(
        main,
        "_require_workspace_manager",
        AsyncMock(return_value={"role": "owner"}),
    )
    monkeypatch.setattr(
        main,
        "_workspace_membership",
        AsyncMock(return_value={"role": "member"}),
    )
    monkeypatch.setattr(main.media_library, "get_media_storage_records", get_records)
    monkeypatch.setattr(main.media_library, "move_media_records", move_records)

    result = await main.bulk_move_media(
        main.BulkMediaMoveRequest(
            source_video_ids=["video-1"],
            destination_client_id=DESTINATION_CLIENT_ID,
        ),
        request,  # type: ignore[arg-type]
    )

    assert result["moved"] == 1
    assert result["destination_client_id"] == DESTINATION_CLIENT_ID
    assert file_tool.copy_path_to_storage.await_count == 2
    move_records.assert_awaited_once()
    file_tool.delete_paths_from_storage.assert_awaited_once_with(
        ["videos/video-1.mp4", "thumbnails/video-1.webp"]
    )
