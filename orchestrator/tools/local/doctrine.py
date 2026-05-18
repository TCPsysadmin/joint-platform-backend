from __future__ import annotations

import json
from typing import Any
from uuid import UUID

from supabase import AsyncClient

from orchestrator.tools.protocols import Doctrine


def _coerce_rubric_raw(rubric_raw: object) -> object:
    if isinstance(rubric_raw, str):
        return json.loads(rubric_raw)
    return rubric_raw


def parse_doctrine_row(data: dict[str, object], client_id: UUID) -> Doctrine:
    """Map a brand_doctrine row to Doctrine (rubric JSON matches Video Pilot schema)."""
    rubric_raw = _coerce_rubric_raw(data.get("rubric"))
    dimensions: list[dict[str, object]]
    minimum_a_tier_score = 0.78
    auto_reject_below = 0.45

    if isinstance(rubric_raw, dict):
        rubric_obj: dict[str, Any] = rubric_raw  # type: ignore[assignment]
        dimensions = list(rubric_obj.get("dimensions") or [])
        minimum_a_tier_score = float(rubric_obj.get("minimum_a_tier_score", minimum_a_tier_score))
        auto_reject_below = float(rubric_obj.get("auto_reject_below", auto_reject_below))
    elif isinstance(rubric_raw, list):
        dimensions = list(rubric_raw)  # type: ignore[arg-type]
    else:
        dimensions = []

    return Doctrine(
        client_id=client_id,
        name=str(data["name"]),
        rubric=dimensions,
        minimum_a_tier_score=minimum_a_tier_score,
        auto_reject_below=auto_reject_below,
        metadata={},
    )


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
            .limit(1)
            .execute()
        )
        rows = response.data or []
        if not rows:
            raise ValueError(f"No active brand doctrine for client {self._client_id}")
        data: dict[str, object] = rows[0]
        return parse_doctrine_row(data, self._client_id)
