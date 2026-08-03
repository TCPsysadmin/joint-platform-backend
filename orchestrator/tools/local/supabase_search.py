from __future__ import annotations

import posixpath
import re
from typing import Any
from urllib.parse import unquote, urlparse
from uuid import UUID

from supabase import AsyncClient

from orchestrator.supabase_json import as_dict_list
from orchestrator.tools.protocols import AssetHit, SourceVideo, TranscriptHit

_SOURCE_SELECT = (
    "source_video_id, title, source_file, has_timestamps, duration_seconds, "
    "recorded_at, b2_path"
)
_SEGMENT_SELECT = (
    "segment_id, source_video_id, source_title, source_file, has_timestamps, "
    "chunk_index, start_seconds, end_seconds, transcript_text, speaker"
)
_SEGMENT_SOURCE_SELECT = "source_video_id, source_title, source_file, has_timestamps"
_SEGMENT_PAGE_SIZE = 1000
_MARKDOWN_LINK_RE = re.compile(r"^\s*\[(?P<label>[^\]]+)\]\((?P<url>[^)]+)\)\s*$")
_CODED_SOURCE_RE = re.compile(r"\b[A-Z0-9]{2,}(?:[_-][A-Z0-9]+)*_[0-9]{8}\b", re.IGNORECASE)
_SOURCE_FILE_EXTENSIONS = {
    ".aac",
    ".csv",
    ".docx",
    ".flac",
    ".json",
    ".log",
    ".m4a",
    ".m4v",
    ".md",
    ".mov",
    ".mp3",
    ".mp4",
    ".pdf",
    ".srt",
    ".text",
    ".txt",
    ".vtt",
    ".wav",
}


class SupabaseSearchTool:
    """Implements SearchTool against Supabase RPC functions."""

    def __init__(self, supabase: AsyncClient, client_id: UUID) -> None:
        self._supabase = supabase
        self._client_id = client_id

    async def resolve_source_video(self, source_reference: str) -> SourceVideo | None:
        reference = _clean_source_reference(source_reference)
        if not reference:
            return None

        rows_by_id: dict[str, dict[str, Any]] = {}
        variants = _source_reference_variants(reference)

        async def collect_summary(field: str, value: str, *, exact: bool = False) -> None:
            query = (
                self._supabase.table("video_summaries")
                .select(_SOURCE_SELECT)
                .eq("client_id", str(self._client_id))
            )
            query = query.eq(field, value) if exact else query.ilike(field, _like_pattern(value))
            response = await query.limit(10).execute()
            for row in as_dict_list(response.data if response else None):
                rows_by_id[str(row["source_video_id"])] = row

        async def collect_segment(field: str, value: str, *, exact: bool = False) -> None:
            query = (
                self._supabase.table("transcript_segments")
                .select(_SEGMENT_SOURCE_SELECT)
                .eq("client_id", str(self._client_id))
            )
            query = query.eq(field, value) if exact else query.ilike(field, _like_pattern(value))
            response = await query.limit(10).execute()
            for row in as_dict_list(response.data if response else None):
                source_video_id = str(row["source_video_id"])
                rows_by_id.setdefault(
                    source_video_id,
                    {
                        "source_video_id": source_video_id,
                        "title": row.get("source_title"),
                        "source_file": row.get("source_file"),
                        "has_timestamps": row.get("has_timestamps", True),
                    },
                )

        for variant in variants:
            await collect_summary("source_video_id", variant, exact=True)
            await collect_segment("source_video_id", variant, exact=True)

        for variant in variants:
            for field in ("title", "source_file", "b2_path", "source_video_id"):
                await collect_summary(field, variant)
            for field in ("source_title", "source_file", "source_video_id"):
                await collect_segment(field, variant)

        if not rows_by_id:
            return None

        ref_norms = [_normalize_source_text(v) for v in variants]
        best = max(
            rows_by_id.values(),
            key=lambda row: _source_match_score(row, ref_norms),
        )
        return _source_video_from_row(best)

    async def list_transcript_segments_for_video(
        self,
        *,
        source_video_id: str,
    ) -> list[TranscriptHit]:
        hits: list[TranscriptHit] = []
        offset = 0

        while True:
            response = (
                await self._supabase.table("transcript_segments")
                .select(_SEGMENT_SELECT)
                .eq("client_id", str(self._client_id))
                .eq("source_video_id", source_video_id)
                .order("chunk_index")
                .range(offset, offset + _SEGMENT_PAGE_SIZE - 1)
                .execute()
            )
            rows = as_dict_list(response.data if response else None)
            for row in rows:
                hits.append(_transcript_hit_from_row(row))
            if len(rows) < _SEGMENT_PAGE_SIZE:
                break
            offset += _SEGMENT_PAGE_SIZE

        return hits

    async def search_transcripts(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        match_count: int = 40,
        min_seconds: float = 0.0,
    ) -> list[TranscriptHit]:
        response = await self._supabase.rpc(
            "hybrid_search_transcripts",
            {
                "p_query_text": query_text,
                "p_query_embedding": query_embedding,
                "p_match_count": match_count,
                "p_min_seconds": min_seconds,
            },
        ).execute()

        hits: list[TranscriptHit] = []
        for row in as_dict_list(response.data if response else None):
            hits.append(
                TranscriptHit(
                    segment_id=str(row["segment_id"]),
                    video_id=str(row["source_video_id"]),
                    text=str(row["transcript_text"]),
                    start_seconds=float(row["start_seconds"]),
                    end_seconds=float(row["end_seconds"]),
                    has_timestamps=bool(row.get("has_timestamps", True)),
                    score=float(row.get("fused_score") or 0.0),
                    metadata={
                        k: row[k]
                        for k in ("source_title", "source_file", "speaker", "chunk_index")
                        if k in row and row[k] is not None
                    },
                )
            )
        return hits

    async def match_assets(
        self,
        *,
        query_embedding: list[float],
        asset_types: list[str] | None = None,
        match_count: int = 8,
    ) -> list[AssetHit]:
        params: dict[str, Any] = {
            "p_query_embedding": query_embedding,
            "p_match_count": match_count,
        }
        if asset_types is not None:
            params["p_asset_types"] = asset_types

        response = await self._supabase.rpc("match_assets", params).execute()

        return [
            AssetHit(
                asset_id=str(row["asset_id"]),
                asset_type=str(row["asset_type"]),
                description=str(row.get("description") or ""),
                score=float(row.get("similarity") or 0.0),
                url=None,
                metadata={"parameters": row.get("parameters") or {}},
            )
            for row in as_dict_list(response.data if response else None)
        ]


def _clean_source_reference(value: str) -> str:
    text = value.strip()
    match = _MARKDOWN_LINK_RE.match(text)
    if match:
        text = match.group("url") or match.group("label")
    return text.strip()


def _without_extension(value: str) -> str:
    root, extension = posixpath.splitext(value)
    if extension.lower() not in _SOURCE_FILE_EXTENSIONS:
        return value
    return root


def _source_reference_variants(value: str) -> list[str]:
    variants: list[str] = []

    def add(candidate: str | None) -> None:
        text = (candidate or "").strip().strip("/")
        if text and text not in variants:
            variants.append(text)

    cleaned = _clean_source_reference(value)
    decoded = unquote(cleaned)
    add(cleaned)
    add(decoded)

    parsed = urlparse(decoded)
    if parsed.scheme and parsed.netloc:
        path = unquote(parsed.path).strip("/")
        add(path)
        path_parts = path.split("/")
        # Backblaze public URLs look like /file/{bucket}/{key}. The database
        # usually stores either the object key, basename, or source title.
        if len(path_parts) >= 3 and path_parts[0] == "file":
            add("/".join(path_parts[2:]))
        add(posixpath.basename(path))

    for candidate in list(variants):
        basename = posixpath.basename(candidate)
        add(basename)
        add(_without_extension(basename))
        add(_without_extension(candidate))
        coded_match = _CODED_SOURCE_RE.search(candidate)
        if coded_match:
            add(coded_match.group(0))

    return variants


def _like_pattern(value: str) -> str:
    """Build a punctuation- and whitespace-agnostic ILIKE pattern.

    Stored titles and the reference we are handed rarely agree on separators.
    The ingestion workflow builds ``title`` by replacing ``[-_]+`` with a
    space, so ``TCP001_DITL_20250911 - SO WHAT`` is stored with a run of three
    spaces; by the time the model reads it back out of rendered markdown the
    run has collapsed to one. A literal ``%<value>%`` then matches nothing and
    we never get far enough to score the row.

    Collapsing every run of non-alphanumeric characters to ``%`` makes the
    lookup indifferent to separators in either direction. It is deliberately
    loose -- ``_source_match_score`` picks the best of the widened candidate
    set, and each query is still capped at 10 rows.
    """
    tokens = [t for t in re.split(r"[^0-9A-Za-z]+", value) if t]
    if not tokens:
        return "%"
    return "%" + "%".join(tokens) + "%"


def _normalize_source_text(value: str) -> str:
    decoded = unquote(value.strip().strip("/"))
    basename = posixpath.basename(decoded)
    normalized = _without_extension(basename).lower()
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


def _source_match_score(row: dict[str, Any], ref_norms: list[str]) -> tuple[int, int]:
    candidates = [
        str(row.get("source_video_id") or ""),
        str(row.get("title") or ""),
        str(row.get("source_title") or ""),
        str(row.get("source_file") or ""),
        str(row.get("b2_path") or ""),
    ]
    normalized = [_normalize_source_text(c) for c in candidates if c]
    exact_matches = [c for c in normalized if c in ref_norms]
    if exact_matches:
        return (100, -min(len(c) for c in exact_matches))
    for value in normalized:
        if any(ref_norm and (ref_norm in value or value in ref_norm) for ref_norm in ref_norms):
            return (50, -len(value))
    return (0, 0)


def _source_video_from_row(row: dict[str, Any]) -> SourceVideo:
    return SourceVideo(
        source_video_id=str(row["source_video_id"]),
        title=str(row["title"]) if row.get("title") else None,
        source_file=str(row["source_file"]) if row.get("source_file") else None,
        has_timestamps=bool(row.get("has_timestamps", True)),
        metadata={
            k: row[k]
            for k in ("duration_seconds", "recorded_at", "b2_path")
            if k in row and row[k] is not None
        },
    )


def _transcript_hit_from_row(row: dict[str, Any]) -> TranscriptHit:
    return TranscriptHit(
        segment_id=str(row["segment_id"]),
        video_id=str(row["source_video_id"]),
        text=str(row["transcript_text"]),
        start_seconds=float(row["start_seconds"]),
        end_seconds=float(row["end_seconds"]),
        has_timestamps=bool(row.get("has_timestamps", True)),
        score=1.0,
        metadata={
            k: row[k]
            for k in ("source_title", "source_file", "speaker", "chunk_index")
            if k in row and row[k] is not None
        },
    )
