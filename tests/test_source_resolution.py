from __future__ import annotations

import re
from typing import Any

import pytest

from orchestrator.tools.local.supabase_search import (
    SupabaseSearchTool,
    _source_match_score,
    _source_reference_variants,
)
from tests.conftest import FAKE_CLIENT_ID


class _Response:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


def _compile_like(pattern: str) -> re.Pattern[str]:
    """Translate a SQL LIKE pattern to a regex, the way Postgres ILIKE does.

    ``%`` matches any run of characters, ``_`` matches exactly one, and the
    pattern is anchored against the whole value. Treating the pattern as a
    plain substring hid two divergences from the real database: internal ``%``
    wildcards, and the ``_`` that appears in nearly every source_video_id.
    """
    translated = "".join(
        ".*" if ch == "%" else "." if ch == "_" else re.escape(ch) for ch in pattern
    )
    return re.compile(translated + r"\Z", re.IGNORECASE | re.DOTALL)


class _FakeQuery:
    def __init__(self, table_name: str, rows: list[dict[str, Any]]) -> None:
        self._table_name = table_name
        self._rows = rows
        self._eq: dict[str, str] = {}
        self._ilike: tuple[str, re.Pattern[str]] | None = None
        self._limit = 10

    def select(self, _fields: str) -> _FakeQuery:
        return self

    def eq(self, field: str, value: str) -> _FakeQuery:
        self._eq[field] = value
        return self

    def ilike(self, field: str, pattern: str) -> _FakeQuery:
        self._ilike = (field, _compile_like(pattern))
        return self

    def limit(self, count: int) -> _FakeQuery:
        self._limit = count
        return self

    async def execute(self) -> _Response:
        matches: list[dict[str, Any]] = []
        for row in self._rows:
            if any(str(row.get(field)) != value for field, value in self._eq.items()):
                continue
            if self._ilike is not None:
                field, regex = self._ilike
                if not regex.match(str(row.get(field) or "")):
                    continue
            matches.append(row)
        return _Response(matches[: self._limit])


class _FakeSupabase:
    def __init__(self) -> None:
        self.rows = {
            "video_summaries": [],
            "transcript_segments": [
                {
                    "client_id": str(FAKE_CLIENT_ID),
                    "source_video_id": "tcp-001-ditl",
                    "source_title": "TCP001_DITL_20250502 - JUST TRY",
                    "source_file": "TCP001_DITL_20250502 - JUST TRY.mp3",
                    "has_timestamps": False,
                },
                {
                    "client_id": str(FAKE_CLIENT_ID),
                    "source_video_id": "tcp-003-meetings",
                    "source_title": (
                        "TCP003_MEETINGS_20230724 Interpersonal Conflict Dr Kieschnick"
                    ),
                    "source_file": "Interpersonal Conflict Dr Kieschnick.mp3",
                    "has_timestamps": True,
                }
            ],
        }

    def table(self, table_name: str) -> _FakeQuery:
        return _FakeQuery(table_name, self.rows[table_name])


def test_source_reference_variants_extract_backblaze_url_parts() -> None:
    variants = _source_reference_variants(
        "[https://f005.backblazeb2.com/file/TCP-MASTER/TCP001_DITL/"
        "TCP001_DITL_20250502%20-%20JUST%20TRY.mp3]"
        "(https://f005.backblazeb2.com/file/TCP-MASTER/TCP001_DITL/"
        "TCP001_DITL_20250502%20-%20JUST%20TRY.mp3)"
    )

    assert "TCP001_DITL/TCP001_DITL_20250502 - JUST TRY.mp3" in variants
    assert "TCP001_DITL_20250502 - JUST TRY.mp3" in variants
    assert "TCP001_DITL_20250502 - JUST TRY" in variants


def test_source_reference_variants_keep_dr_name_and_extract_date_code() -> None:
    variants = _source_reference_variants(
        "TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr. Kieschnick"
    )

    assert "TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr. Kieschnick" in variants
    assert "TCP003_MEETINGS_20230724" in variants
    assert "TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr" not in variants


def test_source_match_score_treats_title_and_file_stem_as_exact_match() -> None:
    score = _source_match_score(
        {
            "title": "TCP001_DITL_20250502 - JUST TRY",
            "source_file": "TCP001_DITL_20250502 - JUST TRY.mp3",
        },
        ["tcp001 ditl 20250502 just try"],
    )

    assert score[0] == 100


@pytest.mark.asyncio
async def test_resolve_source_video_falls_back_to_transcript_segment_metadata() -> None:
    tool = SupabaseSearchTool(
        supabase=_FakeSupabase(),  # type: ignore[arg-type]
        client_id=FAKE_CLIENT_ID,
    )

    source = await tool.resolve_source_video(
        "https://f005.backblazeb2.com/file/TCP-MASTER/TCP001_DITL/"
        "TCP001_DITL_20250502%20-%20JUST%20TRY.mp3"
    )

    assert source is not None
    assert source.source_video_id == "tcp-001-ditl"
    assert source.title == "TCP001_DITL_20250502 - JUST TRY"
    assert source.source_file == "TCP001_DITL_20250502 - JUST TRY.mp3"


@pytest.mark.asyncio
async def test_resolve_source_video_uses_date_code_when_title_punctuation_differs() -> None:
    tool = SupabaseSearchTool(
        supabase=_FakeSupabase(),  # type: ignore[arg-type]
        client_id=FAKE_CLIENT_ID,
    )

    source = await tool.resolve_source_video(
        "TCP003_MEETINGS_20230724 - Interpersonal Conflict Dr. Kieschnick"
    )

    assert source is not None
    assert source.source_video_id == "tcp-003-meetings"
