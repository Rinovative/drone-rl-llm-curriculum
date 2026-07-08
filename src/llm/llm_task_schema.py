"""
===============================================================================
llm_task_schema.py
===============================================================================
Describe and normalize LLM-proposed trajectory task dictionaries.

Responsibilities:
  - Build a compact JSON-serializable task schema from validation contracts
  - Provide bounded prompt-contract text for LLM curriculum prompts
  - Normalize raw decoded task mappings before deterministic validation

Design principles:
  - Treat validation contracts as the source of truth for supported fields
  - Keep schema helpers deterministic and free of external API calls

Boundaries:
  - Numeric feasibility checks belong in validation_tasks.py
  - LLM clients, retries, logging, and prompt execution belong elsewhere
===============================================================================

"""

from __future__ import annotations

import json
from collections.abc import Mapping

from src import validation

REASON_FIELD = "reason"
_FORBIDDEN_EXAMPLE_FIELDS = (
    "python_code",
    "command",
    "script",
    "shell",
    "imports",
)


def build_task_schema() -> dict[str, object]:
    """
    Build a JSON-serializable schema description for supported trajectory tasks.

    Returns
    -------
    dict[str, object]
        Compact schema-like dictionary describing supported task type, shapes,
        known fields, optional metadata fields, and shape-specific required keys.

    """
    contracts = validation.contracts
    return {
        "task_type": contracts.TASK_TYPE_TRAJECTORY,
        "shapes": list(contracts.SUPPORTED_TRAJECTORY_SHAPES),
        "required_fields": [contracts.FIELD_TASK_TYPE, contracts.FIELD_SHAPE],
        "known_fields": list(_known_task_fields()),
        "optional_metadata_fields": [REASON_FIELD],
        "forbidden_example_fields": list(_FORBIDDEN_EXAMPLE_FIELDS),
        "shape_required_fields": _shape_required_fields(),
        "shape_optional_fields": _shape_optional_fields(),
    }


def build_task_prompt_contract() -> str:
    """
    Build bounded plain-text instructions for trajectory task prompts.

    Returns
    -------
    str
        Deterministic prompt contract that names supported shapes, known keys,
        shape-specific required keys, and JSON-only output expectations.

    """
    shapes = ", ".join(validation.contracts.SUPPORTED_TRAJECTORY_SHAPES)
    known_fields = ", ".join(_known_task_fields())
    forbidden_fields = ", ".join(_FORBIDDEN_EXAMPLE_FIELDS)
    required_by_shape = json.dumps(_shape_required_fields(), sort_keys=True, separators=(",", ":"))
    optional_by_shape = json.dumps(_shape_optional_fields(), sort_keys=True, separators=(",", ":"))
    return (
        "Return exactly one JSON object and no prose. "
        f"Use task_type={validation.contracts.TASK_TYPE_TRAJECTORY!r}. "
        f"Supported shapes: {shapes}. "
        f"Known task keys: {known_fields}. "
        f"Required keys by shape: {required_by_shape}. "
        f"Optional keys by shape: {optional_by_shape}. "
        f"Optional metadata keys: {REASON_FIELD}. "
        "The optional reason field is metadata only and is never used for deterministic validation. "
        f"Do not include code, commands, markdown, unsupported keys, or fields such as {forbidden_fields}."
    )


def normalize_proposed_task(raw_task: object) -> dict[str, object]:
    """
    Normalize a decoded LLM task object into a plain dictionary.

    Parameters
    ----------
    raw_task
        Decoded JSON-like object expected to be a mapping.

    Returns
    -------
    dict[str, object]
        Plain copied task dictionary. Optional ``reason`` metadata is preserved
        by normalization but excluded before deterministic validation.

    Raises
    ------
    ValueError
        If the input is not a mapping, required keys are missing, or unknown
        top-level keys are present.

    """
    if not isinstance(raw_task, Mapping):
        message = "proposed task must be a mapping"
        raise ValueError(message)  # noqa: TRY004 - normalization contract requires ValueError.

    normalized = dict(raw_task)
    missing = [key for key in _required_top_level_fields() if key not in normalized]
    if missing:
        message = f"proposed task is missing required keys: {', '.join(missing)}"
        raise ValueError(message)

    allowed_keys = set(_known_task_fields()) | {REASON_FIELD}
    unknown_keys = sorted(set(normalized) - allowed_keys)
    if unknown_keys:
        message = f"proposed task contains unsupported keys: {', '.join(unknown_keys)}"
        raise ValueError(message)
    return normalized


def validate_proposed_task(raw_task: object) -> validation.tasks.ValidationResult:
    """
    Normalize a proposed task and validate it deterministically.

    Parameters
    ----------
    raw_task
        Decoded JSON-like task mapping.

    Returns
    -------
    validation.tasks.ValidationResult
        Existing validation result returned by ``validation.tasks.validate_task``.

    Raises
    ------
    ValueError
        If schema-level normalization fails before deterministic validation.

    """
    normalized = normalize_proposed_task(raw_task)
    validation_task = {key: value for key, value in normalized.items() if key != REASON_FIELD}
    return validation.tasks.validate_task(validation_task)


def task_without_metadata(task: Mapping[str, object]) -> dict[str, object]:
    """
    Return a copied task without LLM-only metadata fields.

    Parameters
    ----------
    task
        Normalized proposed task mapping.

    Returns
    -------
    dict[str, object]
        Copy suitable for deterministic validation and PPO task configs.

    """
    return {key: value for key, value in task.items() if key != REASON_FIELD}


def _required_top_level_fields() -> tuple[str, ...]:
    """Return required keys for every proposed task."""
    contracts = validation.contracts
    return (
        contracts.FIELD_TASK_TYPE,
        contracts.FIELD_SHAPE,
    )


def _shape_required_fields() -> dict[str, list[str]]:
    """Return shape-specific required fields supported by deterministic validation."""
    contracts = validation.contracts
    hover_fields = [
        contracts.FIELD_DURATION_SEC,
        contracts.FIELD_SAMPLE_RATE_HZ,
        contracts.FIELD_POSITION,
    ]
    line_fields = [
        contracts.FIELD_DURATION_SEC,
        contracts.FIELD_SAMPLE_RATE_HZ,
        contracts.FIELD_START,
        contracts.FIELD_END,
    ]
    return {
        contracts.SHAPE_HOVER_STABILIZATION: list(hover_fields),
        contracts.SHAPE_NEARBY_TARGET_HOVER: list(hover_fields),
        contracts.SHAPE_START_HOLD_THEN_SHORT_LINE: [
            contracts.FIELD_HOLD_DURATION_SEC,
            contracts.FIELD_MOVE_DURATION_SEC,
            contracts.FIELD_SAMPLE_RATE_HZ,
            contracts.FIELD_START,
            contracts.FIELD_END,
        ],
        contracts.SHAPE_SHORT_SLOW_LINE: list(line_fields),
        contracts.SHAPE_HOVER: list(hover_fields),
        contracts.SHAPE_CIRCLE: [
            contracts.FIELD_DURATION_SEC,
            contracts.FIELD_SAMPLE_RATE_HZ,
            contracts.FIELD_RADIUS,
            contracts.FIELD_HEIGHT,
        ],
        contracts.SHAPE_LINE: list(line_fields),
        contracts.SHAPE_VERTICAL: [
            contracts.FIELD_DURATION_SEC,
            contracts.FIELD_SAMPLE_RATE_HZ,
            contracts.FIELD_XY,
            contracts.FIELD_START_HEIGHT,
            contracts.FIELD_END_HEIGHT,
        ],
        contracts.SHAPE_POLYLINE: [
            contracts.FIELD_DURATION_SEC,
            contracts.FIELD_SAMPLE_RATE_HZ,
            contracts.FIELD_POINTS,
        ],
    }


def _shape_optional_fields() -> dict[str, list[str]]:
    """Return shape-specific optional fields supported by deterministic validation."""
    contracts = validation.contracts
    start_hold_fields = [
        contracts.FIELD_START_HOLD_ENABLED,
        contracts.FIELD_START_HOLD_SEC,
        contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS,
    ]
    optional_fields = {shape: list(start_hold_fields) for shape in contracts.SUPPORTED_TRAJECTORY_SHAPES}
    optional_fields[contracts.SHAPE_CIRCLE] = [contracts.FIELD_CENTER, contracts.FIELD_CLOCKWISE, *start_hold_fields]
    return optional_fields


def _known_task_fields() -> tuple[str, ...]:
    """Return known task fields from validation contract constants."""
    contracts = validation.contracts
    return (
        contracts.FIELD_TASK_TYPE,
        contracts.FIELD_SHAPE,
        contracts.FIELD_DURATION_SEC,
        contracts.FIELD_SAMPLE_RATE_HZ,
        contracts.FIELD_POSITION,
        contracts.FIELD_CENTER,
        contracts.FIELD_RADIUS,
        contracts.FIELD_HEIGHT,
        contracts.FIELD_CLOCKWISE,
        contracts.FIELD_START,
        contracts.FIELD_END,
        contracts.FIELD_XY,
        contracts.FIELD_START_HEIGHT,
        contracts.FIELD_END_HEIGHT,
        contracts.FIELD_POINTS,
        contracts.FIELD_HOLD_DURATION_SEC,
        contracts.FIELD_MOVE_DURATION_SEC,
        contracts.FIELD_START_HOLD_ENABLED,
        contracts.FIELD_START_HOLD_SEC,
        contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS,
    )


__all__ = [
    "REASON_FIELD",
    "build_task_prompt_contract",
    "build_task_schema",
    "normalize_proposed_task",
    "task_without_metadata",
    "validate_proposed_task",
]
