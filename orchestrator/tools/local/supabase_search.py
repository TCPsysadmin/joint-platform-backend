from __future__ import annotations

from typing import Any
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
        for row in response.data or []:
            if not isinstance(row, dict):
                continue
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
            for row in (response.data or [])
            if isinstance(row, dict)
        ]
