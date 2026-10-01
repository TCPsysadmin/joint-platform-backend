from __future__ import annotations

from uuid import UUID

from orchestrator.tools.local.doctrine import parse_doctrine_row
from tests.conftest import FAKE_CLIENT_ID


def test_parse_doctrine_row_nested_rubric() -> None:
    row = {
        "name": "Core Doctrine",
        "rubric": {
            "dimensions": [{"key": "hook_strength", "weight": 0.2}],
            "minimum_a_tier_score": 0.78,
            "auto_reject_below": 0.45,
        },
    }
    doctrine = parse_doctrine_row(row, FAKE_CLIENT_ID)
    assert doctrine.name == "Core Doctrine"
    assert len(doctrine.rubric) == 1
    assert doctrine.minimum_a_tier_score == 0.78
    assert doctrine.auto_reject_below == 0.45


def test_parse_doctrine_row_json_string_rubric() -> None:
    row = {
        "name": "Core Doctrine",
        "rubric": (
            '{"dimensions": [{"key": "hook_strength", "weight": 0.2}], '
            '"minimum_a_tier_score": 0.78, "auto_reject_below": 0.45}'
        ),
    }
    doctrine = parse_doctrine_row(row, FAKE_CLIENT_ID)
    assert len(doctrine.rubric) == 1
    assert doctrine.minimum_a_tier_score == 0.78


def test_parse_doctrine_row_list_rubric() -> None:
    row = {
        "name": "Legacy",
        "rubric": [{"key": "hook_strength", "weight": 1.0}],
    }
    doctrine = parse_doctrine_row(row, UUID(int=0))
    assert len(doctrine.rubric) == 1
    assert doctrine.minimum_a_tier_score == 0.78
