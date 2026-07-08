"""
===============================================================================
llm_json.py
===============================================================================
Parse strict JSON-only objects returned by local LLM curriculum providers.

Responsibilities:
  - Accept exactly one top-level JSON object
  - Reject markdown, prose, arrays, empty responses, and trailing content
  - Report clear parse errors for repair prompts and proposal logs

Design principles:
  - Use the standard JSON parser only
  - Keep parsing deterministic and free of executable-code paths

Boundaries:
  - Task schema checks belong in llm_task_schema.py
  - Provider calls and repair retries belong in other LLM modules
===============================================================================

"""

from __future__ import annotations

import json
from typing import Any


class LLMJsonError(ValueError):
    """Raised when an LLM response is not exactly one JSON object."""


def parse_json_object(response_text: str) -> dict[str, Any]:
    """
    Parse exactly one JSON object from an LLM response.

    Parameters
    ----------
    response_text
        Raw text returned by an LLM chat completion.

    Returns
    -------
    dict[str, Any]
        Decoded top-level JSON object.

    Raises
    ------
    LLMJsonError
        If the response is empty, fenced, prose-wrapped, an array, malformed,
        or contains more than one JSON value.

    """
    if not isinstance(response_text, str):
        message = "LLM response must be text"
        raise LLMJsonError(message)

    stripped = response_text.strip()
    if not stripped:
        message = "LLM response is empty"
        raise LLMJsonError(message)
    if "```" in stripped:
        message = "LLM response must not contain markdown fences"
        raise LLMJsonError(message)
    if stripped.startswith("["):
        message = "LLM response must be one JSON object, not an array"
        raise LLMJsonError(message)
    if not stripped.startswith("{"):
        message = "LLM response must start with a JSON object and contain no leading prose"
        raise LLMJsonError(message)

    decoder = json.JSONDecoder()
    try:
        decoded, end_index = decoder.raw_decode(stripped)
    except json.JSONDecodeError as exc:
        message = f"LLM response is not valid JSON: {exc.msg}"
        raise LLMJsonError(message) from exc

    if stripped[end_index:].strip():
        message = "LLM response must contain exactly one JSON object with no trailing prose or extra objects"
        raise LLMJsonError(message)
    if not isinstance(decoded, dict):
        message = "LLM response top-level JSON value must be an object"
        raise LLMJsonError(message)
    return decoded


__all__ = [
    "LLMJsonError",
    "parse_json_object",
]
