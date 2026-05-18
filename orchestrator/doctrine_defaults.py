from __future__ import annotations

from typing import Any
from uuid import UUID

_DEFAULT_DIMENSIONS: list[dict[str, object]] = [
    {"key": "narrative_tension", "weight": 0.25, "description": "Conflict and stakes"},
    {"key": "doctrinal_alignment", "weight": 0.30, "description": "Brand fit"},
    {"key": "hook_strength", "weight": 0.20, "description": "Opening 3 seconds"},
    {"key": "quotability", "weight": 0.15, "description": "Memorable phrasing"},
    {"key": "production_quality", "weight": 0.10, "description": "Audio/video clarity"},
]


def default_brand_doctrine_dict(
    client_id: UUID | None = None,
    *,
    permissive: bool = False,
) -> dict[str, Any]:
    """Built-in rubric when REQUIRE_BRAND_DOCTRINE=false or DB fetch is skipped."""
    return {
        "client_id": str(client_id) if client_id else "",
        "name": "Default (doctrine bypass)" if permissive else "Core Doctrine",
        "rubric": list(_DEFAULT_DIMENSIONS),
        "minimum_a_tier_score": 0.0 if permissive else 0.78,
        "auto_reject_below": 0.0 if permissive else 0.45,
        "metadata": {"source": "builtin_default", "permissive": permissive},
    }
