from __future__ import annotations

from uuid import UUID

from supabase import AsyncClient

from orchestrator.tools.protocols import AssetHit, TranscriptHit


class SupabaseSearchTool:
    """Implements SearchTool against Supabase RPC functions."""

    def __init__(self, supabase: AsyncClient, client_id: UUID) -> None:
        self._supabase = supabase
        self._client_id = client_id

    async def search_transcripts(
        self,
        *,
        query_text: str,
        query_embedding: list[float],
        match_count: int = 40,
        min_seconds: float = 0.0,
    ) -> list[TranscriptHit]:
        # TODO: optionally pre-filter via hybrid_search_summaries when corpus grows
        response = await self._supabase.rpc(
            "hybrid_search_transcripts",
            {
                "query_text": query_text,
                "query_embedding": query_embedding,
                "match_count": match_count,
                "min_seconds": min_seconds,
                "p_client_id": str(self._client_id),
            },
        ).execute()

        return [
            TranscriptHit(
                segment_id=row["segment_id"],
                video_id=row["video_id"],
                text=row["text"],
                start_seconds=float(row["start_seconds"]),
                end_seconds=float(row["end_seconds"]),
                has_timestamps=bool(row["has_timestamps"]),
                score=float(row["score"]),
                metadata=row.get("metadata") or {},
            )
            for row in (response.data or [])
        ]

    async def match_assets(
        self,
        *,
        query_embedding: list[float],
        asset_types: list[str] | None = None,
        match_count: int = 8,
    ) -> list[AssetHit]:
        params: dict[str, object] = {
            "query_embedding": query_embedding,
            "match_count": match_count,
            "p_client_id": str(self._client_id),
        }
        if asset_types is not None:
            params["asset_types"] = asset_types

        response = await self._supabase.rpc("match_assets", params).execute()

        return [
            AssetHit(
                asset_id=row["asset_id"],
                asset_type=row["asset_type"],
                description=row["description"],
                score=float(row["score"]),
                url=row.get("url"),
                metadata=row.get("metadata") or {},
            )
            for row in (response.data or [])
        ]
