from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable
from uuid import UUID


@dataclass
class TranscriptHit:
    segment_id: str
    video_id: str
    text: str
    start_seconds: float
    end_seconds: float
    has_timestamps: bool
    score: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class AssetHit:
    asset_id: str
    asset_type: str
    description: str
    score: float
    url: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class Doctrine:
    client_id: UUID
    name: str
    rubric: list[dict[str, object]]
    minimum_a_tier_score: float
    auto_reject_below: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class ClipPayload:
    video_id: str
    start_seconds: float
    end_seconds: float
    hook_quote: str
    broll_suggestions: list[dict[str, object]]
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class PublishResult:
    clip_id: str
    status: str
    url: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class B2Bucket:
    bucket_id: str
    bucket_name: str
    bucket_type: str
    account_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class B2FileEntry:
    """One entry from b2_list_file_names / b2_list_file_versions.

    `action` is "upload" for real files, "folder" for delimiter-collapsed
    directory placeholders, "hide" for hide markers, and "start" for
    in-progress large file uploads.
    """

    file_name: str
    file_id: str | None
    action: str
    content_length: int
    content_type: str | None = None
    upload_timestamp: int | None = None
    bucket_id: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class B2KeyInfo:
    application_key_id: str
    key_name: str
    capabilities: list[str]
    account_id: str | None = None
    bucket_id: str | None = None
    name_prefix: str | None = None
    expiration_timestamp: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


# ── OpusClip ─────────────────────────────────────────────────────────────────


@dataclass
class OpusProject:
    project_id: str
    status: str | None = None
    title: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class OpusClip:
    project_id: str
    clip_id: str
    title: str | None = None
    description: str | None = None
    hashtags: list[str] = field(default_factory=list)
    score: float | None = None
    duration_ms: int | None = None
    preview_url: str | None = None
    export_url: str | None = None
    thumbnail_url: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class OpusBrandTemplate:
    template_id: str
    name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class OpusCollection:
    collection_id: str
    name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class OpusSocialAccount:
    post_account_id: str
    sub_account_id: str | None
    platform: str
    name: str | None = None
    profile_url: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class OpusJob:
    """Generic OpusClip async job (censor, social-copy, edit-clip)."""

    job_id: str
    status: str | None = None
    result: dict[str, object] | None = None
    error: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass
class OpusUploadLink:
    upload_id: str
    url: str
    dns_url: str | None = None
    use_amount: int | None = None
    total_amount: int | None = None


@runtime_checkable
class SearchTool(Protocol):
    async def search_transcripts(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        match_count: int = 40,
        min_seconds: float = 0.0,
    ) -> list[TranscriptHit]: ...

    async def match_assets(
        self,
        *,
        query_embedding: list[float],
        asset_types: list[str] | None = None,
        match_count: int = 8,
    ) -> list[AssetHit]: ...


@runtime_checkable
class DoctrineTool(Protocol):
    async def get_active_doctrine(self) -> Doctrine: ...


@runtime_checkable
class PublishTool(Protocol):
    async def create_and_post_clip(self, payload: ClipPayload) -> PublishResult: ...


@runtime_checkable
class Embedder(Protocol):
    async def embed(self, text: str) -> list[float]: ...


@runtime_checkable
class FileTool(Protocol):
    async def fetch_source_file(self, source_video_id: str) -> str | None: ...

    async def list_buckets(
        self,
        *,
        bucket_id: str | None = None,
        bucket_name: str | None = None,
        bucket_types: list[str] | None = None,
    ) -> list[B2Bucket]: ...

    async def list_file_names(
        self,
        *,
        bucket_id: str,
        prefix: str | None = None,
        delimiter: str | None = None,
        start_file_name: str | None = None,
        max_file_count: int = 100,
    ) -> tuple[list[B2FileEntry], str | None]: ...

    async def list_file_versions(
        self,
        *,
        bucket_id: str,
        prefix: str | None = None,
        delimiter: str | None = None,
        start_file_name: str | None = None,
        start_file_id: str | None = None,
        max_file_count: int = 100,
    ) -> tuple[list[B2FileEntry], str | None, str | None]: ...

    async def list_keys(
        self,
        *,
        max_key_count: int = 100,
        start_application_key_id: str | None = None,
    ) -> tuple[list[B2KeyInfo], str | None]: ...

    async def find_file_by_name(
        self,
        *,
        file_name: str,
        bucket_name: str | None = None,
        prefix: str | None = None,
    ) -> B2FileEntry | None: ...
