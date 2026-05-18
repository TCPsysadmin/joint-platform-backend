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
