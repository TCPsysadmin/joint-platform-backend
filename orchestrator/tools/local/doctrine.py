from __future__ import annotations

from uuid import UUID

from supabase import AsyncClient

from orchestrator.tools.protocols import Doctrine


class SupabaseDoctrineTool:
    """Implements DoctrineTool against the brand_doctrine Supabase table."""

    def __init__(self, supabase: AsyncClient, client_id: UUID) -> None:
        self._supabase = supabase
        self._client_id = client_id

    async def get_active_doctrine(self) -> Doctrine:
        response = (
            await self._supabase.table("brand_doctrine")
            .select("*")
            .eq("client_id", str(self._client_id))
            .eq("is_active", True)
            .single()
            .execute()
        )
        data: dict[str, object] = response.data
        return Doctrine(
            client_id=self._client_id,
            name=str(data["name"]),
            rubric=list(data["rubric"]),  # type: ignore[arg-type]
            minimum_a_tier_score=float(data["minimum_a_tier_score"]),  # type: ignore[arg-type]
            auto_reject_below=float(data["auto_reject_below"]),  # type: ignore[arg-type]
            metadata=data.get("metadata") or {},  # type: ignore[assignment]
        )
