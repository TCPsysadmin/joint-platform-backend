"""The B2 "legend" file index, concurrent fetches, and the download-URL encoding fix.

Every B2 call is served by an `httpx.MockTransport` — nothing here touches the
network. `B2FileTool` builds its own `httpx.AsyncClient` per batch, so the tests
swap the class out inside the `b2_file` module namespace.
"""

from __future__ import annotations

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import httpx
import pytest

from orchestrator.tools.local import b2_file as b2_file_module
from orchestrator.tools.local.b2_file import B2FileTool
from orchestrator.tools.local.b2_legend import (
    B2Legend,
    base_match_key,
    legend_path,
    load_legend,
    normalize_file_match_text,
    save_legend,
    sidecar_role,
)
from orchestrator.tools.protocols import B2FileEntry, FileTool
from tests.conftest import FAKE_CLIENT_ID

BUCKET = "TCP-MASTER"

# The same basename under two folders — the collision the legend must resolve.
VIDEO_PATH = "raw/TCP001_DITL_20250502 - JUST TRY.mp3"
TRANSCRIPT_PATH = "transcripts/TCP001_DITL_20250502 - JUST TRY.txt"
SUBTITLE_PATH = "transcripts/TCP001_DITL_20250502 - JUST TRY.srt"
OLD_DUPLICATE_PATH = "archive/2019/TCP001_DITL_20250502 - JUST TRY.txt"
# Characters httpx would otherwise read as a URL fragment.
AWKWARD_PATH = "raw/notes #1 – café.txt"

# The convention the tenant's bucket actually uses: a transcript is not the video
# with a different extension, it is the video's stem plus a role suffix. Fixtures
# built on same-stem sidecars are why the pairing bug passed 88 tests. Distinct
# stem from the paths above so it does not perturb the collision fixtures.
_REAL_DIR = "TCP004_PORTAL/TCP004_PORTAL_2025 - 10d MINI"
_REAL_STEM = f"{_REAL_DIR}/TCP004_PORTAL_20250601 - 10d MINI - 10"
REAL_VIDEO_PATH = f"{_REAL_STEM}.mp4"
REAL_TRANSCRIPT_PATH = f"{_REAL_STEM}_transcript.txt"
REAL_SUMMARY_PATH = f"{_REAL_STEM}_summary.txt"
# A source whose video was never uploaded — only its transcript exists.
ORPHAN_TRANSCRIPT_PATH = "TCP003_MEETINGS/TCP003_MEETINGS_20260430 - AI MEETING_transcript.txt"
ORPHAN_VIDEO_NAME = "TCP003_MEETINGS_20260430 - AI MEETING.mp4"
# A title that merely contains the word "transcript"; its stem must survive intact.
TRANSCRIPT_ONLY_VIDEO_PATH = (
    "TCP003_MEETINGS/TCP003_MEETING_20260625 - PAW MSS DEBRIEF - TRANSCRIPT ONLY.mp4"
)


def _file_row(name: str, *, ts: int, content_type: str = "text/plain") -> dict[str, Any]:
    return {
        "fileName": name,
        "fileId": f"id-{name}",
        "action": "upload",
        "contentLength": 42,
        "contentType": content_type,
        "uploadTimestamp": ts,
    }


DEFAULT_PAGES: list[dict[str, Any]] = [
    {
        "files": [
            _file_row(OLD_DUPLICATE_PATH, ts=1_000),
            _file_row(AWKWARD_PATH, ts=1_500),
            {"fileName": "raw/", "action": "folder", "contentLength": 0},
        ],
        "nextFileName": "raw/x",
    },
    {
        "files": [
            _file_row(VIDEO_PATH, ts=2_000, content_type="audio/mpeg"),
            _file_row(TRANSCRIPT_PATH, ts=3_000),
            _file_row(SUBTITLE_PATH, ts=3_000, content_type="text/srt"),
            _file_row(REAL_VIDEO_PATH, ts=4_000, content_type="video/mp4"),
            _file_row(REAL_TRANSCRIPT_PATH, ts=4_100),
            _file_row(REAL_SUMMARY_PATH, ts=4_200),
            _file_row(ORPHAN_TRANSCRIPT_PATH, ts=4_300),
            _file_row(TRANSCRIPT_ONLY_VIDEO_PATH, ts=4_400, content_type="video/mp4"),
        ],
        "nextFileName": None,
    },
]


# ── fake Supabase ────────────────────────────────────────────────────────────


class _Response:
    def __init__(self, data: dict[str, Any] | None) -> None:
        self.data = data


class _FakeQuery:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self._rows = rows
        self._eq: dict[str, str] = {}

    def select(self, _fields: str) -> _FakeQuery:
        return self

    def eq(self, field: str, value: str) -> _FakeQuery:
        self._eq[field] = value
        return self

    def maybe_single(self) -> _FakeQuery:
        return self

    async def execute(self) -> _Response:
        for row in self._rows:
            if all(str(row.get(k)) == v for k, v in self._eq.items()):
                return _Response(row)
        return _Response(None)


class _FakeSupabase:
    def __init__(self, **tables: list[dict[str, Any]]) -> None:
        self._tables = tables

    def table(self, name: str) -> _FakeQuery:
        return _FakeQuery(self._tables.get(name, []))


def _supabase(*, source_file: str | None = None, b2_path: str | None = None) -> _FakeSupabase:
    return _FakeSupabase(
        clients_registry=[{"client_id": str(FAKE_CLIENT_ID), "b2_bucket": BUCKET, "b2_prefix": ""}],
        video_summaries=[
            {
                "client_id": str(FAKE_CLIENT_ID),
                "source_video_id": "vid-abc",
                "b2_path": b2_path,
                "source_file": source_file,
            }
        ],
    )


# ── fake B2 ──────────────────────────────────────────────────────────────────


class FakeB2:
    """Serves the handful of B2 endpoints B2FileTool touches."""

    def __init__(
        self,
        *,
        pages: list[dict[str, Any]] | None = None,
        downloads: dict[str, httpx.Response] | None = None,
        expire_first_token: bool = False,
        expire_for_path: str | None = None,
        fail_page_index: int | None = None,
        fail_count: int = 1,
        synthetic_page_count: int | None = None,
        files_per_page: int = 5,
        versions: dict[str, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.pages = pages if pages is not None else DEFAULT_PAGES
        self.downloads = downloads or {}
        self.expire_first_token = expire_first_token
        # Like `expire_first_token`, but scoped to a single path — simulates a
        # token expiring mid-batch, not on every download in it.
        self.expire_for_path = expire_for_path
        # Makes b2_list_file_names 500 for the Nth page (0=first, 1=second, ...)
        # the first `fail_count` times it's requested, then serves normally —
        # a transient mid-sweep failure that must recover on a later call.
        self.fail_page_index = fail_page_index
        self.fail_count = fail_count
        self._page_fail_calls: dict[int, int] = {}
        # When set, b2_list_file_names ignores `pages` and synthesizes an
        # effectively unbounded number of pages, to exercise the sweep's own
        # page cap rather than running out of fixture data first.
        self.synthetic_page_count = synthetic_page_count
        self.files_per_page = files_per_page
        self.versions = versions or {}
        self.requests: list[httpx.Request] = []
        self.auth_calls = 0
        self.list_calls = 0
        self.download_calls = 0
        self.sign_calls = 0
        self.delete_calls: list[dict[str, str]] = []

    @property
    def token(self) -> str:
        return f"token-{self.auth_calls}"

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        url = str(request.url)

        if "b2_authorize_account" in url:
            self.auth_calls += 1
            return httpx.Response(
                200,
                json={
                    "authorizationToken": self.token,
                    "accountId": "acct-1",
                    "apiInfo": {
                        "storageApi": {
                            "apiUrl": "https://api005.backblazeb2.com",
                            "downloadUrl": "https://f005.backblazeb2.com",
                        }
                    },
                },
            )

        if "b2_list_buckets" in url:
            return httpx.Response(
                200,
                json={
                    "buckets": [
                        {
                            "bucketId": "bucket-1",
                            "bucketName": BUCKET,
                            "bucketType": "allPrivate",
                            "accountId": "acct-1",
                        }
                    ]
                },
            )

        if "b2_list_file_names" in url:
            self.list_calls += 1
            start = parse_qs(urlparse(url).query).get("startFileName", [None])[0]

            if self.synthetic_page_count is not None:
                page = 0 if start is None else int(start.split(":", 1)[1])
                files = [
                    _file_row(f"raw/synthetic_{page}_{i}.txt", ts=page * 1000 + i)
                    for i in range(self.files_per_page)
                ]
                next_page = page + 1
                next_file_name = (
                    f"page:{next_page}" if next_page < self.synthetic_page_count else None
                )
                return httpx.Response(200, json={"files": files, "nextFileName": next_file_name})

            index = 0 if start is None else 1
            if self.fail_page_index is not None and index == self.fail_page_index:
                calls = self._page_fail_calls.get(index, 0) + 1
                self._page_fail_calls[index] = calls
                if calls <= self.fail_count:
                    return httpx.Response(500, json={"status": 500, "code": "internal_error"})
            return httpx.Response(200, json=self.pages[min(index, len(self.pages) - 1)])

        if "b2_list_file_versions" in url:
            prefix = parse_qs(urlparse(url).query).get("prefix", [""])[0]
            return httpx.Response(
                200,
                json={"files": self.versions.get(prefix, []), "nextFileName": None},
            )

        if "b2_delete_file_version" in url:
            payload = json.loads(request.content)
            self.delete_calls.append(
                {"fileName": str(payload["fileName"]), "fileId": str(payload["fileId"])}
            )
            return httpx.Response(200, json=payload)

        if "/file/" in url:
            self.download_calls += 1
            if self.expire_first_token and request.headers.get("Authorization") == "token-1":
                return httpx.Response(401, json={"status": 401, "code": "expired_auth_token"})
            path = unquote(urlparse(url).path).split(f"/file/{BUCKET}/", 1)[-1]
            if (
                self.expire_for_path
                and path == self.expire_for_path
                and request.headers.get("Authorization") == "token-1"
            ):
                return httpx.Response(401, json={"status": 401, "code": "expired_auth_token"})
            if path in self.downloads:
                return self.downloads[path]
            return httpx.Response(200, text="default body", headers={"content-type": "text/plain"})

        if "b2_get_download_authorization" in url:
            self.sign_calls += 1
            return httpx.Response(200, json={"authorizationToken": "sign-token"})

        raise AssertionError(f"unexpected B2 call: {url}")


def _install(monkeypatch: pytest.MonkeyPatch, fake: FakeB2) -> None:
    real_client = httpx.AsyncClient

    def _factory(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(fake.handle), **kwargs)

    monkeypatch.setattr(b2_file_module.httpx, "AsyncClient", _factory)


def _tool(
    tmp_path: Path,
    *,
    supabase: Any = None,
    ttl: int = 900,
    min_refresh: int = 60,
    concurrency: int = 6,
) -> B2FileTool:
    return B2FileTool(
        supabase=supabase if supabase is not None else _supabase(),
        client_id=FAKE_CLIENT_ID,
        key_id="key",
        application_key="secret",
        legend_cache_dir=tmp_path,
        legend_ttl_seconds=ttl,
        legend_min_refresh_seconds=min_refresh,
        fetch_concurrency=concurrency,
    )


# ── legend build / lookup / collisions ───────────────────────────────────────


def test_b2_file_tool_still_satisfies_the_file_tool_protocol(tmp_path: Path) -> None:
    assert isinstance(_tool(tmp_path), FileTool)


@pytest.mark.asyncio
async def test_delete_paths_removes_every_version_inside_the_workspace_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = "tenant-a/videos/video-1.mp4"
    thumbnail_path = "tenant-a/thumbnails/video-1.webp"
    fake = FakeB2(
        versions={
            video_path: [
                _file_row(video_path, ts=2_000, content_type="video/mp4"),
                {
                    **_file_row(video_path, ts=1_000, content_type="video/mp4"),
                    "fileId": "old-video-version",
                },
            ],
            thumbnail_path: [
                _file_row(thumbnail_path, ts=2_000, content_type="image/webp")
            ],
        }
    )
    _install(monkeypatch, fake)
    supabase = _FakeSupabase(
        clients_registry=[
            {
                "client_id": str(FAKE_CLIENT_ID),
                "b2_bucket": BUCKET,
                "b2_prefix": "tenant-a",
            }
        ]
    )
    tool = _tool(tmp_path, supabase=supabase)

    deleted = await tool.delete_paths_from_storage(
        ["videos/video-1.mp4", "thumbnails/video-1.webp"]
    )

    assert deleted == ["videos/video-1.mp4", "thumbnails/video-1.webp"]
    assert fake.delete_calls == [
        {"fileName": video_path, "fileId": f"id-{video_path}"},
        {"fileName": video_path, "fileId": "old-video-version"},
        {"fileName": thumbnail_path, "fileId": f"id-{thumbnail_path}"},
    ]


@pytest.mark.asyncio
async def test_delete_paths_rejects_paths_outside_the_workspace(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    with pytest.raises(ValueError, match="invalid path segment"):
        await tool.delete_paths_from_storage(["../other-workspace/video.mp4"])

    assert fake.delete_calls == []


@pytest.mark.asyncio
async def test_legend_build_pages_the_bucket_once_and_answers_later_lookups_from_memory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    entry = await tool.find_file_by_name(file_name="TCP001_DITL_20250502 - JUST TRY.mp3")
    assert entry is not None
    assert entry.file_name == VIDEO_PATH
    assert entry.action == "upload"
    assert fake.list_calls == 2, "both pages of the sweep are followed via nextFileName"

    # A second lookup must not touch B2 at all.
    again = await tool.find_file_by_name(file_name=TRANSCRIPT_PATH)
    assert again is not None and again.file_name == TRANSCRIPT_PATH
    assert fake.list_calls == 2


@pytest.mark.asyncio
async def test_legend_skips_folder_placeholders(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)
    await tool.find_file_by_name(file_name=TRANSCRIPT_PATH)

    legend = tool._legend
    assert legend is not None
    assert all(e.action == "upload" for entries in legend.files.values() for e in entries)
    assert not any(e.file_name.endswith("/") for entries in legend.files.values() for e in entries)


@pytest.mark.asyncio
async def test_legend_collision_prefers_exact_path_then_most_recent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    # Three files share one normalized key (extension is stripped on purpose).
    key = normalize_file_match_text(VIDEO_PATH)
    await tool.find_file_by_name(file_name=VIDEO_PATH)
    legend = tool._legend
    assert legend is not None
    assert {e.file_name for e in legend.files[key]} == {
        VIDEO_PATH,
        TRANSCRIPT_PATH,
        SUBTITLE_PATH,
        OLD_DUPLICATE_PATH,
    }

    # Exact full path wins over its same-named siblings...
    exact = await tool.find_file_by_name(file_name=OLD_DUPLICATE_PATH)
    assert exact is not None and exact.file_name == OLD_DUPLICATE_PATH

    # ...and a bare basename falls through to the newest upload.
    newest = legend.lookup("TCP001_DITL_20250502 - JUST TRY")
    assert newest is not None and newest.upload_timestamp == 3_000


@pytest.mark.asyncio
async def test_legend_finds_text_siblings_of_a_media_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)
    await tool.find_file_by_name(file_name=VIDEO_PATH)

    legend = tool._legend
    assert legend is not None
    siblings = [e.file_name for e in legend.text_siblings(VIDEO_PATH)]
    # The .mp3 itself is excluded; the .txt/.srt twins come back sorted.
    assert siblings == [OLD_DUPLICATE_PATH, SUBTITLE_PATH, TRANSCRIPT_PATH]


@pytest.mark.asyncio
async def test_a_lookup_outside_the_clients_location_bypasses_the_legend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The legend indexes exactly one bucket+prefix. A caller that overrides either
    must fall straight through to the scan rather than answer from the wrong index."""
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    entry = await tool.find_file_by_name(
        file_name=TRANSCRIPT_PATH, bucket_name=BUCKET, prefix="some/other/prefix"
    )

    # Answered by the original scan, whose "first basename match wins" semantics
    # are unchanged — not by the index built for the client's own prefix.
    assert entry is not None
    assert tool._legend is None, "no legend is built for a non-default location"
    assert not legend_path(tmp_path, FAKE_CLIENT_ID).exists()


@pytest.mark.asyncio
async def test_a_bucket_name_override_also_bypasses_the_legend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The prefix-override bypass is covered above; overriding just `bucket_name`
    away from the client's configured bucket must bypass the legend the same way —
    the index was built for one bucket and must never answer for another."""
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    entry = await tool.find_file_by_name(file_name=TRANSCRIPT_PATH, bucket_name="OTHER-BUCKET")

    assert entry is not None
    assert tool._legend is None, "no legend is built for a non-default bucket"
    assert not legend_path(tmp_path, FAKE_CLIENT_ID).exists()


@pytest.mark.asyncio
async def test_legend_sweep_is_capped_at_the_page_limit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pathologically large tenant must not page forever. The sweep stops at
    `_LEGEND_MAX_PAGES`, and the legend built from that truncated sweep is still
    usable for anything indexed before the cap was hit."""
    total_pages = b2_file_module._LEGEND_MAX_PAGES + 50
    fake = FakeB2(synthetic_page_count=total_pages, files_per_page=5)
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    entry = await tool.find_file_by_name(file_name="synthetic_0_0.txt")
    assert entry is not None
    assert entry.file_name == "raw/synthetic_0_0.txt"
    assert fake.list_calls == b2_file_module._LEGEND_MAX_PAGES, "the sweep stops at the page cap"

    legend = tool._legend
    assert legend is not None
    assert legend.entry_count == b2_file_module._LEGEND_MAX_PAGES * 5, (
        "only what was listed before the cap is indexed"
    )

    # A file that only exists on a page past the cap was never listed, so it is
    # correctly reported missing rather than the sweep hanging trying to find it.
    beyond_cap = await tool.find_file_by_name(file_name=f"synthetic_{total_pages - 1}_0.txt")
    assert beyond_cap is None


# ── refresh policy ───────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_lookup_miss_refreshes_once_then_falls_back_to_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, min_refresh=0)

    missing = await tool.find_file_by_name(file_name="nowhere.txt")
    assert missing is None
    # 2 (initial sweep) + 2 (miss-triggered refresh) + 1 (fallback scan page one,
    # which stops at the page cap only after following nextFileName).
    assert fake.list_calls > 4


@pytest.mark.asyncio
async def test_min_refresh_floor_stops_a_missing_file_from_resweeping_every_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, min_refresh=600)

    await tool.find_file_by_name(file_name="nowhere.txt")
    after_first = fake.list_calls
    await tool.find_file_by_name(file_name="also-nowhere.txt")

    # Only the fallback scan re-lists; the legend is not rebuilt a second time.
    rebuilt_pages = fake.list_calls - after_first
    assert rebuilt_pages <= 2, "the miss-triggered rebuild must be rate-limited"


@pytest.mark.asyncio
async def test_expired_ttl_triggers_a_rebuild(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, ttl=900)

    await tool.find_file_by_name(file_name=TRANSCRIPT_PATH)
    assert fake.list_calls == 2

    assert tool._legend is not None
    tool._legend.built_at = datetime.now(UTC) - timedelta(seconds=901)

    await tool.find_file_by_name(file_name=TRANSCRIPT_PATH)
    assert fake.list_calls == 4, "a stale legend is rebuilt before the lookup"


# Bespoke pages for the failed-sweep test below: each page has one file with a
# unique stem, so the paginated fallback scan is forced to reach page 2 rather
# than short-circuiting on page 1's shared-stem collision (see DEFAULT_PAGES).
_FAIL_TEST_PAGES: list[dict[str, Any]] = [
    {"files": [_file_row("raw/alpha.txt", ts=1_000)], "nextFileName": "raw/x"},
    {"files": [_file_row("raw/beta.txt", ts=2_000)], "nextFileName": None},
]


@pytest.mark.asyncio
async def test_a_failed_sweep_persists_nothing_and_falls_back_to_the_scan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Page 2 of the sweep 500s. The build must not persist a half-built legend,
    and the turn must still succeed via the paginated fallback scan. The next
    call — once the transient failure is over — must recover cleanly."""
    fake = FakeB2(pages=_FAIL_TEST_PAGES, fail_page_index=1, fail_count=1)
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    entry = await tool.find_file_by_name(file_name="raw/beta.txt")

    assert entry is not None and entry.file_name == "raw/beta.txt", (
        "the fallback scan still finds the file even though the sweep failed"
    )
    assert tool._legend is None, "a failed build must not leave a partial legend in memory"
    assert not legend_path(tmp_path, FAKE_CLIENT_ID).exists(), "nothing corrupt is persisted"

    # The transient failure is over now (fail_count=1 was already spent), so the
    # very next call must degrade gracefully rather than staying broken forever.
    second = await tool.find_file_by_name(file_name="raw/alpha.txt")
    assert second is not None
    assert tool._legend is not None
    on_disk = json.loads(legend_path(tmp_path, FAKE_CLIENT_ID).read_text(encoding="utf-8"))
    assert on_disk["bucket"] == BUCKET


# ── persistence ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_legend_round_trips_through_disk_across_tool_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)

    await _tool(tmp_path).find_file_by_name(file_name=TRANSCRIPT_PATH)
    assert fake.list_calls == 2

    path = legend_path(tmp_path, FAKE_CLIENT_ID)
    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["bucket"] == BUCKET
    assert on_disk["prefix"] == ""
    datetime.fromisoformat(on_disk["built_at"])  # ISO-8601
    assert any(
        row["file_name"] == TRANSCRIPT_PATH for rows in on_disk["files"].values() for row in rows
    )

    # A fresh tool (i.e. the next request) reads the cache instead of re-listing.
    entry = await _tool(tmp_path).find_file_by_name(file_name=TRANSCRIPT_PATH)
    assert entry is not None and entry.file_name == TRANSCRIPT_PATH
    assert fake.list_calls == 2


@pytest.mark.asyncio
async def test_a_persisted_legend_for_a_different_bucket_is_rebuilt_not_served_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A legend cached on disk under an old bucket/prefix (e.g. the tenant's B2
    location was reconfigured) must never answer a lookup for the *current*
    location — `matches()` failing must trigger a rebuild, not a stale hit."""
    stale = B2Legend.from_entries(
        [B2FileEntry(file_name="old/file.txt", file_id="old-1", action="upload", content_length=1)],
        bucket="OLD-BUCKET",
        prefix="old-prefix",
    )
    path = legend_path(tmp_path, FAKE_CLIENT_ID)
    assert save_legend(path, stale) is True

    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    entry = await tool.find_file_by_name(file_name=TRANSCRIPT_PATH)

    assert entry is not None and entry.file_name == TRANSCRIPT_PATH
    assert fake.list_calls == 2, "the mismatched cache is ignored and a real sweep runs"
    assert tool._legend is not None
    assert tool._legend.bucket == BUCKET and tool._legend.prefix == ""

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk["bucket"] == BUCKET, "the stale entry is overwritten, not merged with"


def test_load_legend_treats_missing_and_corrupt_files_as_no_legend(tmp_path: Path) -> None:
    path = legend_path(tmp_path, FAKE_CLIENT_ID)
    assert load_legend(path) is None

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{not json", encoding="utf-8")
    assert load_legend(path) is None

    path.write_text(json.dumps({"files": {}}), encoding="utf-8")  # no built_at
    assert load_legend(path) is None


def test_save_legend_writes_atomically_and_leaves_no_temp_files(tmp_path: Path) -> None:
    legend = B2Legend.from_entries(
        [B2FileEntry(file_name=TRANSCRIPT_PATH, file_id="f1", action="upload", content_length=5)],
        bucket=BUCKET,
        prefix="",
    )
    path = legend_path(tmp_path, FAKE_CLIENT_ID)
    assert save_legend(path, legend) is True
    assert save_legend(path, legend) is True  # overwrite must not fail

    assert [p.name for p in tmp_path.iterdir()] == [path.name]
    restored = load_legend(path)
    assert restored is not None
    assert restored.lookup(TRANSCRIPT_PATH) is not None
    assert restored.bucket == BUCKET


def test_concurrent_saves_never_leave_a_corrupt_legend_file(tmp_path: Path) -> None:
    """Several sessions for one tenant can race to rebuild the legend. Per
    `save_legend`'s own contract, a write that loses the race may report False
    (logged, never raised) rather than corrupt anything — `os.replace` either
    fully lands or doesn't touch the destination at all. What must hold
    regardless of how many individual writes fail under contention is: no
    half-written temp file is ever left behind, and the destination — once
    anything has been written — is always whole, parseable JSON, never torn."""
    path = legend_path(tmp_path, FAKE_CLIENT_ID)
    legends = [
        B2Legend.from_entries(
            [
                B2FileEntry(
                    file_name=f"writer-{writer}/{i}.txt",
                    file_id=f"id-{writer}-{i}",
                    action="upload",
                    content_length=i,
                )
                for i in range(25)
            ],
            bucket=BUCKET,
            prefix="",
        )
        for writer in range(4)
    ]
    results: list[bool] = []

    def _write_repeatedly(legend: B2Legend) -> None:
        for _ in range(15):
            results.append(save_legend(path, legend))

    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_write_repeatedly, legend) for legend in legends]
        for future in futures:
            future.result()

    assert any(results), "at least one writer must land under this much contention"
    # Whichever writer landed last, the file must be whole, valid JSON — never a
    # half-written temp artifact and never a crash reading a torn file — no
    # matter how many of the raced writes above reported False.
    restored = load_legend(path)
    assert restored is not None
    assert restored.bucket == BUCKET
    assert [p.name for p in tmp_path.iterdir()] == [path.name], "no leaked .tmp files"


# ── concurrent fetch / auth retry / URL encoding ─────────────────────────────


@pytest.mark.asyncio
async def test_fetch_files_returns_one_result_per_path_including_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2(
        downloads={
            TRANSCRIPT_PATH: httpx.Response(
                200, text="the transcript", headers={"content-type": "text/plain"}
            ),
            VIDEO_PATH: httpx.Response(
                200, content=b"\x00\x01", headers={"content-type": "audio/mpeg"}
            ),
            SUBTITLE_PATH: httpx.Response(503, text="nope"),
        }
    )
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, concurrency=4)

    results = await tool.fetch_files(
        paths=[TRANSCRIPT_PATH, VIDEO_PATH, SUBTITLE_PATH], bucket_name=BUCKET
    )

    assert [r.file_name for r in results] == [TRANSCRIPT_PATH, VIDEO_PATH, SUBTITLE_PATH]
    # Three genuinely different outcomes, not three Nones.
    assert results[0].content == "the transcript" and results[0].is_text
    assert results[1].content is None and results[1].is_text is False and results[1].error is None
    assert results[2].content is None and results[2].error is not None
    # The media file still gets a result row, but is ruled out by name — only the
    # transcript and the failing subtitle are actually requested.
    assert fake.download_calls == 2


@pytest.mark.asyncio
async def test_fetch_files_deduplicates_and_ignores_blank_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    results = await tool.fetch_files(paths=[TRANSCRIPT_PATH, TRANSCRIPT_PATH, "  "])
    assert [r.file_name for r in results] == [TRANSCRIPT_PATH]
    assert await tool.fetch_files(paths=[]) == []


@pytest.mark.asyncio
async def test_expired_auth_token_reauthorizes_and_retries_that_call_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2(
        downloads={
            TRANSCRIPT_PATH: httpx.Response(
                200, text="recovered", headers={"content-type": "text/plain"}
            )
        },
        expire_first_token=True,
    )
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    results = await tool.fetch_files(paths=[TRANSCRIPT_PATH], bucket_name=BUCKET)

    assert results[0].content == "recovered"
    assert fake.auth_calls == 2, "the stale token is refreshed exactly once"
    assert fake.download_calls == 2, "only the failed call is retried"


@pytest.mark.asyncio
async def test_expired_auth_in_a_batch_only_retries_the_one_affected_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A late-batch token expiry must not discard the work already done on the
    other files in the same concurrent fetch — only the one call that actually
    hit the stale token is retried."""
    fake = FakeB2(
        downloads={
            TRANSCRIPT_PATH: httpx.Response(
                200, text="transcript body", headers={"content-type": "text/plain"}
            ),
            VIDEO_PATH: httpx.Response(
                200, content=b"\x00", headers={"content-type": "audio/mpeg"}
            ),
            SUBTITLE_PATH: httpx.Response(
                200, text="subtitle body", headers={"content-type": "text/srt"}
            ),
        },
        expire_for_path=SUBTITLE_PATH,
    )
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, concurrency=4)

    results = await tool.fetch_files(
        paths=[TRANSCRIPT_PATH, VIDEO_PATH, SUBTITLE_PATH], bucket_name=BUCKET
    )

    by_name = {r.file_name: r for r in results}
    assert by_name[TRANSCRIPT_PATH].content == "transcript body"
    assert by_name[VIDEO_PATH].is_text is False
    assert by_name[SUBTITLE_PATH].content == "subtitle body"
    assert fake.auth_calls == 2, "re-authorized exactly once for the one stale call"
    assert fake.download_calls == 3, "2 downloaded files (media skipped) + 1 retry"


@pytest.mark.asyncio
async def test_a_401_that_is_not_an_expiry_is_not_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2(
        downloads={
            TRANSCRIPT_PATH: httpx.Response(401, json={"status": 401, "code": "missing_auth_token"})
        }
    )
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    results = await tool.fetch_files(paths=[TRANSCRIPT_PATH], bucket_name=BUCKET)
    assert results[0].error is not None
    assert fake.auth_calls == 1
    assert fake.download_calls == 1


@pytest.mark.asyncio
async def test_fetch_files_never_exceeds_the_configured_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`fetch_files` bounds fan-out with `asyncio.Semaphore(fetch_concurrency)`.
    Drive more downloads than the bound, each briefly slow, and track how many
    are in flight at once to prove the bound is actually enforced."""
    concurrency = 2
    active = 0
    max_active = 0
    lock = asyncio.Lock()

    async def handle(request: httpx.Request) -> httpx.Response:
        nonlocal active, max_active
        url = str(request.url)
        if "b2_authorize_account" in url:
            return httpx.Response(
                200,
                json={
                    "authorizationToken": "token-1",
                    "accountId": "acct-1",
                    "apiInfo": {
                        "storageApi": {
                            "apiUrl": "https://api005.backblazeb2.com",
                            "downloadUrl": "https://f005.backblazeb2.com",
                        }
                    },
                },
            )
        if "/file/" in url:
            async with lock:
                active += 1
                max_active = max(max_active, active)
            await asyncio.sleep(0.02)
            async with lock:
                active -= 1
            return httpx.Response(200, text="body", headers={"content-type": "text/plain"})
        raise AssertionError(f"unexpected B2 call: {url}")

    real_client = httpx.AsyncClient

    def _factory(**kwargs: Any) -> httpx.AsyncClient:
        return real_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(b2_file_module.httpx, "AsyncClient", _factory)
    tool = _tool(tmp_path, concurrency=concurrency)

    paths = [f"raw/file-{i}.txt" for i in range(6)]
    results = await tool.fetch_files(paths=paths, bucket_name=BUCKET)

    assert len(results) == 6
    assert all(r.content == "body" for r in results)
    assert max_active <= concurrency, "the semaphore bound must never be exceeded"
    assert max_active == concurrency, "6 downloads against a bound of 2 must saturate it"


@pytest.mark.asyncio
async def test_download_urls_percent_encode_the_file_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`#` in a filename used to be parsed as a URL fragment, truncating the path."""
    fake = FakeB2(
        downloads={
            AWKWARD_PATH: httpx.Response(
                200, text="awkward", headers={"content-type": "text/plain"}
            )
        }
    )
    _install(monkeypatch, fake)
    tool = _tool(tmp_path)

    results = await tool.fetch_files(paths=[AWKWARD_PATH], bucket_name=BUCKET)
    assert results[0].content == "awkward"

    download = [r for r in fake.requests if "/file/" in str(r.url)][-1]
    assert download.url.fragment == ""
    assert "%23" in str(download.url), "the '#' must be escaped, not treated as a fragment"
    assert (
        download.url.raw_path.decode()
        == f"/file/{BUCKET}/raw/notes%20%231%20%E2%80%93%20caf%C3%A9.txt"
    )


# ── the "dive deeper" bundle ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_source_bundle_pulls_the_text_sidecars_and_never_the_media(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A source resolves to its video, which carries no ingestible text.

    Requesting it anyway downloaded the whole file only to discard it — 19s and
    zero context against the real bucket.
    """
    fake = FakeB2(
        downloads={
            VIDEO_PATH: httpx.Response(
                200, content=b"\x00", headers={"content-type": "audio/mpeg"}
            ),
            TRANSCRIPT_PATH: httpx.Response(
                200, text="full transcript", headers={"content-type": "text/plain"}
            ),
            SUBTITLE_PATH: httpx.Response(
                200, text="1\n00:00 --> 00:02\nhi", headers={"content-type": "text/srt"}
            ),
            OLD_DUPLICATE_PATH: httpx.Response(
                200, text="stale copy", headers={"content-type": "text/plain"}
            ),
        }
    )
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, supabase=_supabase(b2_path=VIDEO_PATH))

    results = await tool.fetch_source_bundle("vid-abc")

    by_name = {r.file_name: r for r in results}
    assert set(by_name) == {TRANSCRIPT_PATH, SUBTITLE_PATH, OLD_DUPLICATE_PATH}
    assert VIDEO_PATH not in by_name, "the media file is not part of a text bundle"
    assert by_name[TRANSCRIPT_PATH].content == "full transcript"
    assert all(VIDEO_PATH not in str(r.url) for r in fake.requests if "/file/" in str(r.url)), (
        "no request may be made for the media file"
    )


@pytest.mark.asyncio
async def test_fetch_source_bundle_is_empty_when_the_source_cannot_be_resolved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, supabase=_supabase())
    assert await tool.fetch_source_bundle("no-such-video") == []


# ── the real bucket's naming convention ──────────────────────────────────────
#
# The tenant's 534 sidecars are all "<stem>_transcript.txt" / "<stem>_summary.txt".
# Against the live bucket the pre-fix code paired 0 of 236 resolved files with a
# transcript and returned an empty bundle after downloading the whole video.


def test_base_match_key_pairs_a_video_with_its_role_suffixed_sidecars() -> None:
    assert base_match_key(REAL_VIDEO_PATH) == base_match_key(REAL_TRANSCRIPT_PATH)
    assert base_match_key(REAL_VIDEO_PATH) == base_match_key(REAL_SUMMARY_PATH)
    # The un-widened key is what used to file them apart.
    assert normalize_file_match_text(REAL_VIDEO_PATH) != normalize_file_match_text(
        REAL_TRANSCRIPT_PATH
    )


def test_only_a_trailing_role_token_is_stripped() -> None:
    """ "PAW MSS DEBRIEF - TRANSCRIPT ONLY" is a title, not a sidecar."""
    assert base_match_key(TRANSCRIPT_ONLY_VIDEO_PATH).endswith("transcript only")
    assert sidecar_role(TRANSCRIPT_ONLY_VIDEO_PATH) is None
    assert sidecar_role(REAL_TRANSCRIPT_PATH) == "transcript"
    assert sidecar_role(REAL_SUMMARY_PATH) == "summary"
    assert sidecar_role(REAL_VIDEO_PATH) is None


@pytest.mark.asyncio
async def test_a_video_finds_its_role_suffixed_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, FakeB2())
    tool = _tool(tmp_path)
    legend = await tool.refresh_legend(bucket=BUCKET, prefix="")
    assert legend is not None

    siblings = [e.file_name for e in legend.text_siblings(REAL_VIDEO_PATH)]
    assert siblings == [REAL_TRANSCRIPT_PATH, REAL_SUMMARY_PATH], (
        "transcript must sort ahead of summary: ingestion truncates at a character "
        "cap and 36% of the real transcripts exceed it"
    )


@pytest.mark.asyncio
async def test_a_source_whose_video_was_never_uploaded_still_finds_its_transcript(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, FakeB2())
    tool = _tool(tmp_path)

    entry = await tool.find_file_by_name(file_name=ORPHAN_VIDEO_NAME)
    assert entry is not None and entry.file_name == ORPHAN_TRANSCRIPT_PATH


@pytest.mark.asyncio
async def test_the_bundle_for_a_real_video_is_transcript_first_and_costs_no_media_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2(
        downloads={
            REAL_TRANSCRIPT_PATH: httpx.Response(
                200, text="the full transcript", headers={"content-type": "text/plain"}
            ),
            REAL_SUMMARY_PATH: httpx.Response(
                200, text="a short summary", headers={"content-type": "text/plain"}
            ),
        }
    )
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, supabase=_supabase(b2_path=REAL_VIDEO_PATH))

    results = await tool.fetch_source_bundle("vid-abc")

    assert [r.file_name for r in results] == [REAL_TRANSCRIPT_PATH, REAL_SUMMARY_PATH]
    assert results[0].content == "the full transcript"
    assert fake.download_calls == 2, "only the two sidecars, never the .mp4"


@pytest.mark.asyncio
async def test_a_source_with_no_text_anywhere_downloads_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, supabase=_supabase(b2_path=TRANSCRIPT_ONLY_VIDEO_PATH))

    assert await tool.fetch_source_bundle("vid-abc") == []
    assert fake.download_calls == 0, "a media-only source must not be downloaded to find that out"


@pytest.mark.asyncio
async def test_the_transcript_leads_even_when_the_source_resolves_to_the_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no video uploaded, the source resolves to whichever sidecar the index
    yields — against the real bucket that was the summary, which then ate the
    truncation budget ahead of a 61KB transcript."""
    orphan_summary = ORPHAN_TRANSCRIPT_PATH.replace("_transcript.txt", "_summary.txt")
    fake = FakeB2(
        pages=[
            {
                "files": [
                    _file_row(orphan_summary, ts=9_000),
                    _file_row(ORPHAN_TRANSCRIPT_PATH, ts=8_000),
                ],
                "nextFileName": None,
            }
        ],
        downloads={
            ORPHAN_TRANSCRIPT_PATH: httpx.Response(
                200, text="T" * 500, headers={"content-type": "text/plain"}
            ),
            orphan_summary: httpx.Response(
                200, text="S" * 50, headers={"content-type": "text/plain"}
            ),
        },
    )
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, supabase=_supabase(source_file=ORPHAN_VIDEO_NAME))

    results = await tool.fetch_source_bundle("vid-abc")

    assert [r.file_name for r in results] == [ORPHAN_TRANSCRIPT_PATH, orphan_summary]


@pytest.mark.asyncio
async def test_publishing_refuses_to_sign_a_transcript_as_if_it_were_the_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Lookup is permissive so a video-less source can still be read via its
    transcript — but this URL goes to OpusClip, which cannot cut a clip out of a
    .txt. The publish gate must fail the way an unresolvable source always did."""
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, supabase=_supabase(source_file=ORPHAN_VIDEO_NAME))

    assert await tool.get_download_url("vid-abc") is None

    # The same source still ingests fine — the widening is intact.
    entry = await tool.find_file_by_name(file_name=ORPHAN_VIDEO_NAME)
    assert entry is not None and entry.file_name == ORPHAN_TRANSCRIPT_PATH


@pytest.mark.asyncio
async def test_publishing_still_signs_a_real_video(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeB2()
    _install(monkeypatch, fake)
    tool = _tool(tmp_path, supabase=_supabase(b2_path=REAL_VIDEO_PATH))

    url = await tool.get_download_url("vid-abc")
    assert url is not None and ".mp4" in url
