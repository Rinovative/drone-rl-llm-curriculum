"""
===============================================================================
llm_task_schema.py
===============================================================================
Describe and normalize LLM-proposed trajectory task dictionaries.

Responsibilities:
  - Build a compact JSON-serializable task schema from validation contracts
  - Provide bounded prompt-contract text for future LLM curriculum prompts
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

from collections.abc import Mapping

from src import validation

REASON_FIELD = "reason"


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
        "shape_required_fields": {
            contracts.SHAPE_HOVER: [
                contracts.FIELD_DURATION_SEC,
                contracts.FIELD_SAMPLE_RATE_HZ,
                contracts.FIELD_POSITION,
            ],
            contracts.SHAPE_CIRCLE: [
                contracts.FIELD_DURATION_SEC,
                contracts.FIELD_SAMPLE_RATE_HZ,
                contracts.FIELD_RADIUS,
                contracts.FIELD_HEIGHT,
            ],
            contracts.SHAPE_LINE: [
                contracts.FIELD_DURATION_SEC,
                contracts.FIELD_SAMPLE_RATE_HZ,
                contracts.FIELD_START,
                contracts.FIELD_END,
            ],
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
        },
        "shape_optional_fields": {
            contracts.SHAPE_CIRCLE: [contracts.FIELD_CENTER, contracts.FIELD_CLOCKWISE],
        },
    }


def build_task_prompt_contract() -> str:
    """
    Build bounded plain-text instructions for future trajectory task prompts.

    Returns
    -------
    str
        Deterministic prompt contract that names supported shapes, known keys,
        and JSON-only output expectations.

    """
    shapes = ", ".join(validation.contracts.SUPPORTED_TRAJECTORY_SHAPES)
    known_fields = ", ".join(_known_task_fields())
    optional_metadata = REASON_FIELD
    return (
        "Return exactly one JSON object and no prose. "
        f"Use task_type={validation.contracts.TASK_TYPE_TRAJECTORY!r}. "
        f"Supported shapes: {shapes}. "
        f"Known task keys: {known_fields}. "
        f"Optional metadata keys: {optional_metadata}. "
        "Do not include code, commands, markdown, or unsupported keys."
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


def _required_top_level_fields() -> tuple[str, ...]:
    """Return required keys for every proposed task."""
    contracts = validation.contracts
    return (
        contracts.FIELD_TASK_TYPE,
        contracts.FIELD_SHAPE,
    )


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
    )


__all__ = [
    "REASON_FIELD",
    "build_task_prompt_contract",
    "build_task_schema",
    "normalize_proposed_task",
    "validate_proposed_task",
]
