"""Tests for strict JSON-only LLM response parsing."""

# ruff: noqa: S101

from __future__ import annotations

import pytest

from src import llm


def test_parse_json_object_accepts_single_object() -> None:
    """Verify one JSON object with surrounding whitespace is accepted."""
    parsed = llm.json.parse_json_object('  {"task_type":"trajectory","shape":"hover"}  ')

    assert parsed == {"task_type": "trajectory", "shape": "hover"}


@pytest.mark.parametrize(
    "response_text",
    [
        "",
        'Here is the task: {"task_type":"trajectory"}',
        '```json\n{"task_type":"trajectory"}\n```',
        '[{"task_type":"trajectory"}]',
        '{"task_type":"trajectory"}\n{"task_type":"trajectory"}',
        '{"task_type":"trajectory"}\nThanks',
        "print('no')",
    ],
)
def test_parse_json_object_rejects_non_object_or_wrapped_content(response_text: str) -> None:
    """Verify markdown, prose, arrays, code, and multiple objects are rejected."""
    with pytest.raises(llm.json.LLMJsonError):
        llm.json.parse_json_object(response_text)


def test_parse_json_object_rejects_malformed_json_with_clear_error() -> None:
    """Verify malformed JSON reports a parse error suitable for repair prompts."""
    with pytest.raises(llm.json.LLMJsonError, match="not valid JSON"):
        llm.json.parse_json_object('{"task_type": "trajectory",}')
