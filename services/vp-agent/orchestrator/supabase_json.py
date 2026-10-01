from __future__ import annotations

from typing import Any, cast

JsonDict = dict[str, Any]


def as_dict(data: object) -> JsonDict | None:
    if isinstance(data, dict):
        return cast(JsonDict, data)
    return None


def as_dict_list(data: object) -> list[JsonDict]:
    if not isinstance(data, list):
        return []
    return [cast(JsonDict, item) for item in data if isinstance(item, dict)]
