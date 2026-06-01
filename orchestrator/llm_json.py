from __future__ import annotations

import json
import re
from typing import Any

from orchestrator.errors import LLMError

# Models frequently wrap JSON in a markdown code fence (```json ... ```) despite
# instructions to return raw JSON. Strip an optional leading language tag and the
# surrounding fences before parsing.
_FENCE_RE = re.compile(
    r"^\s*```(?:json|JSON)?\s*\n?(?P<body>.*?)\n?\s*```\s*$",
    re.DOTALL,
)


def parse_llm_json(raw: str, *, source: str) -> dict[str, Any]:
    """Parse an LLM response as a JSON object, tolerating markdown code fences.

    Args:
        raw: The model's text output.
        source: Node/caller name, used in the error message on failure.

    Raises:
        LLMError: if the (de-fenced) text is not valid JSON.
    """
    text = raw.strip()
    match = _FENCE_RE.match(text)
    if match:
        text = match.group("body").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMError(f"{source} returned non-JSON: {raw!r}") from exc
