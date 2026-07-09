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
from pathlib import Path

from src import envs, validation

REASON_FIELD = "reason"
STAGE_BUDGET_PROFILE_FIELD = "stage_budget_profile"
BUDGET_RATIONALE_FIELD = "budget_rationale"
BUDGET_PROFILE_BOOTSTRAP = "bootstrap"
DEFAULT_STAGE_BUDGET_PROFILES = (BUDGET_PROFILE_BOOTSTRAP, "short", "normal", "recovery", "extend")
PROPOSAL_KIND_FIELD = "proposal_kind"
PROPOSAL_KIND_TASK = "task"
PROPOSAL_KIND_TASK_DISTRIBUTION = "task_distribution"
TASK_FIELD = "task"
TASK_DISTRIBUTION_ID_FIELD = "task_distribution_id"
TASK_DISTRIBUTION_CONFIG_PATH_FIELD = "task_distribution_config_path"
KNOWN_TASK_DISTRIBUTION_CONFIGS = {
    "bootstrap_randomized_hover_target": "configs/tasks/task_distribution_hover_bootstrap_medium.yaml",
    "hover_bootstrap": "configs/tasks/task_distribution_hover_bootstrap_medium.yaml",
    "short_line_bootstrap": "configs/tasks/task_distribution_short_line_bootstrap_medium.yaml",
    "line_bootstrap": "configs/tasks/task_distribution_line_small.yaml",
    "vertical_bootstrap": "configs/tasks/task_distribution_vertical_bootstrap_medium.yaml",
    "vertical_up_down_bootstrap": "configs/tasks/task_distribution_vertical_up_down_bootstrap_medium.yaml",
    "angled_vertical_bootstrap": "configs/tasks/task_distribution_angled_vertical_bootstrap_medium.yaml",
    "delayed_altitude_polyline_bootstrap": "configs/tasks/task_distribution_delayed_altitude_polyline_bootstrap_medium.yaml",
    "polyline_bootstrap": "configs/tasks/task_distribution_polyline_bootstrap_medium.yaml",
    "zigzag_bootstrap": "configs/tasks/task_distribution_zigzag_bootstrap_medium.yaml",
    "triangle_bootstrap": "configs/tasks/task_distribution_triangle_bootstrap_medium.yaml",
    "multi_height_polyline_bootstrap": "configs/tasks/task_distribution_multi_height_polyline_bootstrap_medium.yaml",
    "tracking_small": "configs/tasks/task_distribution_tracking_small.yaml",
    "tracking_medium": "configs/tasks/task_distribution_tracking_medium.yaml",
    "tracking_broad": "configs/tasks/task_distribution_tracking_broad.yaml",
}
_FORBIDDEN_EXAMPLE_FIELDS = (
    "python_code",
    "command",
    "script",
    "shell",
    "imports",
)


def _llm_supported_task_distribution_families() -> tuple[str, ...]:
    """Return focused task-distribution families exposed to the LLM curriculum designer."""
    return tuple(family for family in envs.task_distribution.supported_task_families() if family != "basic_training_show")


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
        "proposal_kinds": [PROPOSAL_KIND_TASK, PROPOSAL_KIND_TASK_DISTRIBUTION],
        "task_distribution_reference_fields": [TASK_DISTRIBUTION_ID_FIELD, TASK_DISTRIBUTION_CONFIG_PATH_FIELD],
        "known_task_distribution_ids": dict(KNOWN_TASK_DISTRIBUTION_CONFIGS),
        "sampling_bounds_scope": (
            "LLM proposals may select concrete validated tasks or known validated task distributions; "
            "arbitrary sampling-bound proposals are intentionally unsupported."
        ),
        "supported_task_distribution_families": list(_llm_supported_task_distribution_families()),
        "unsupported_task_distribution_families": list(envs.task_distribution.unsupported_requested_task_families()),
        "optional_metadata_fields": [REASON_FIELD, PROPOSAL_KIND_FIELD, STAGE_BUDGET_PROFILE_FIELD, BUDGET_RATIONALE_FIELD],
        "nested_task_wrapper_field": TASK_FIELD,
        "allowed_stage_budget_profiles": list(DEFAULT_STAGE_BUDGET_PROFILES),
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
    distribution_ids = json.dumps(KNOWN_TASK_DISTRIBUTION_CONFIGS, sort_keys=True, separators=(",", ":"))
    budget_profiles = ", ".join(DEFAULT_STAGE_BUDGET_PROFILES)
    supported_families = ", ".join(_llm_supported_task_distribution_families())
    unsupported_families = ", ".join(envs.task_distribution.unsupported_requested_task_families()) or "none"
    return (
        "Return exactly one JSON object and no prose. "
        f"Use proposal_kind={PROPOSAL_KIND_TASK!r} for one concrete validated task or "
        f"proposal_kind={PROPOSAL_KIND_TASK_DISTRIBUTION!r} to select a known task distribution. "
        f"Concrete tasks use task_type={validation.contracts.TASK_TYPE_TRAJECTORY!r}. "
        f"Concrete task proposals may put task keys at the top level or wrap them as {TASK_FIELD}={{...}} with metadata on the wrapper. "
        f"Supported shapes: {shapes}. "
        f"Known task keys: {known_fields}. "
        f"Required keys by shape: {required_by_shape}. "
        f"Optional keys by shape: {optional_by_shape}. "
        "Fixed tasks are degenerate task distributions; randomized distributions sample bounded valid tasks across supported families. "
        "The LLM may not propose arbitrary sampling bounds; choose a concrete validated task or one known distribution id/path. "
        "For stages after stage 1, do not repeat the immediately previous task family or shape; "
        "choose a controlled variation or a different focused distribution. "
        "Concrete tasks should use adjusted lower starts around 0.70-0.95m where safe, start_hold_enabled=true, start_hold_sec=1.2, "
        "exclude_start_hold_from_tracking_metrics=true, and may leave final_hold_enabled implicit "
        "or set final_hold_sec near 1.0 with final-hold metrics excluded. "
        f"Known task distribution ids and paths: {distribution_ids}. "
        f"Supported task-distribution families: {supported_families}. "
        f"Unsupported broad families for now: {unsupported_families}. "
        "Evaluation remains fixed and deterministic. Prefer PID with dynamics and previous-action observations; direct-RPM is experimental. "
        "For overnight screening prefer focused lower-start bootstraps; tracking_small is fallback/consolidation only, "
        "tracking_medium is late/final broad preparation only, and tracking_broad is experimental after simpler distributions are stable. "
        f"Optional stage budget metadata: {STAGE_BUDGET_PROFILE_FIELD} must be one of {budget_profiles}; "
        f"{BUDGET_RATIONALE_FIELD} may briefly justify the selected profile. "
        "Use bootstrap mainly for stage 1 policy warmup, short for easy confirmation stages, normal for progression, "
        "recovery after unstable but promising stages, and extend sparingly for appropriate but undertrained stages. "
        "Do not request arbitrary timesteps, num_envs, "
        "PPO hyperparameters, action interfaces, or reward changes. "
        f"Optional metadata keys: {REASON_FIELD}, {STAGE_BUDGET_PROFILE_FIELD}, {BUDGET_RATIONALE_FIELD}. "
        "Metadata fields are never used for deterministic trajectory validation. "
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
    proposal_kind = _infer_proposal_kind(normalized)
    if proposal_kind == PROPOSAL_KIND_TASK_DISTRIBUTION:
        return _normalize_task_distribution_reference(normalized)
    if proposal_kind != PROPOSAL_KIND_TASK:
        message = f"proposal_kind must be one of: {PROPOSAL_KIND_TASK}, {PROPOSAL_KIND_TASK_DISTRIBUTION}"
        raise ValueError(message)
    return _normalize_concrete_task_proposal(normalized)


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
    if normalized.get(PROPOSAL_KIND_FIELD) == PROPOSAL_KIND_TASK_DISTRIBUTION:
        try:
            envs.task_distribution.load_task_distribution_settings(_distribution_path_from_reference(normalized))
        except (OSError, TypeError, ValueError) as exc:
            return validation.tasks.ValidationResult(is_valid=False, messages=(str(exc),))
        return validation.tasks.ValidationResult(is_valid=True)
    metadata_keys = {REASON_FIELD, PROPOSAL_KIND_FIELD, STAGE_BUDGET_PROFILE_FIELD, BUDGET_RATIONALE_FIELD}
    validation_task = {key: value for key, value in normalized.items() if key not in metadata_keys}
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
    excluded = {REASON_FIELD, STAGE_BUDGET_PROFILE_FIELD, BUDGET_RATIONALE_FIELD}
    if task.get(PROPOSAL_KIND_FIELD) != PROPOSAL_KIND_TASK_DISTRIBUTION:
        excluded.add(PROPOSAL_KIND_FIELD)
    return {key: value for key, value in task.items() if key not in excluded}


def _infer_proposal_kind(raw: Mapping[str, object]) -> str:
    """Infer whether a proposal is a concrete task or distribution reference."""
    explicit_kind = raw.get(PROPOSAL_KIND_FIELD)
    if explicit_kind is not None:
        return str(explicit_kind)
    if TASK_DISTRIBUTION_ID_FIELD in raw or TASK_DISTRIBUTION_CONFIG_PATH_FIELD in raw:
        return PROPOSAL_KIND_TASK_DISTRIBUTION
    return PROPOSAL_KIND_TASK


def _normalize_concrete_task_proposal(raw: dict[str, object]) -> dict[str, object]:
    """Normalize a direct or nested concrete task proposal."""
    if TASK_FIELD in raw:
        return _normalize_nested_task_wrapper(raw)
    return _normalize_direct_concrete_task(raw)


def _normalize_nested_task_wrapper(raw: dict[str, object]) -> dict[str, object]:
    """Extract and normalize a concrete task from a proposal wrapper."""
    allowed_keys = {PROPOSAL_KIND_FIELD, TASK_FIELD, REASON_FIELD, STAGE_BUDGET_PROFILE_FIELD, BUDGET_RATIONALE_FIELD}
    unknown_keys = sorted(set(raw) - allowed_keys)
    if unknown_keys:
        message = f"proposed task wrapper contains unsupported keys: {', '.join(unknown_keys)}"
        raise ValueError(message)
    task_value = raw.get(TASK_FIELD)
    if not isinstance(task_value, Mapping):
        message = "task proposal wrapper must include a concrete task mapping"
        raise ValueError(message)  # noqa: TRY004 - normalization contract requires ValueError.
    normalized = _normalize_direct_concrete_task(dict(task_value))
    normalized[PROPOSAL_KIND_FIELD] = PROPOSAL_KIND_TASK
    for metadata_key in (REASON_FIELD, STAGE_BUDGET_PROFILE_FIELD, BUDGET_RATIONALE_FIELD):
        if metadata_key in raw:
            normalized[metadata_key] = raw[metadata_key]
    _validate_budget_metadata(normalized)
    return normalized


def _normalize_direct_concrete_task(raw: dict[str, object]) -> dict[str, object]:
    """Normalize one concrete task mapping without wrapper fields."""
    explicit_kind = raw.get(PROPOSAL_KIND_FIELD)
    if explicit_kind is not None and str(explicit_kind) != PROPOSAL_KIND_TASK:
        message = f"nested concrete task proposal_kind must be {PROPOSAL_KIND_TASK!r}"
        raise ValueError(message)
    missing = [key for key in _required_top_level_fields() if key not in raw]
    if missing:
        message = f"proposed task is missing required keys: {', '.join(missing)}"
        raise ValueError(message)

    allowed_keys = set(_known_task_fields()) | {REASON_FIELD, PROPOSAL_KIND_FIELD, STAGE_BUDGET_PROFILE_FIELD, BUDGET_RATIONALE_FIELD}
    unknown_keys = sorted(set(raw) - allowed_keys)
    if unknown_keys:
        message = f"proposed task contains unsupported keys: {', '.join(unknown_keys)}"
        raise ValueError(message)
    normalized = dict(raw)
    _validate_budget_metadata(normalized)
    return normalized


def _normalize_task_distribution_reference(raw: dict[str, object]) -> dict[str, object]:
    """Normalize a constrained task-distribution reference proposal."""
    allowed_keys = {
        PROPOSAL_KIND_FIELD,
        TASK_DISTRIBUTION_ID_FIELD,
        TASK_DISTRIBUTION_CONFIG_PATH_FIELD,
        REASON_FIELD,
        STAGE_BUDGET_PROFILE_FIELD,
        BUDGET_RATIONALE_FIELD,
    }
    unknown_keys = sorted(set(raw) - allowed_keys)
    if unknown_keys:
        message = f"proposed task distribution contains unsupported keys: {', '.join(unknown_keys)}"
        raise ValueError(message)
    distribution_id = raw.get(TASK_DISTRIBUTION_ID_FIELD)
    config_path = raw.get(TASK_DISTRIBUTION_CONFIG_PATH_FIELD)
    if distribution_id is None and config_path is None:
        message = "task-distribution proposal must include task_distribution_id or task_distribution_config_path"
        raise ValueError(message)
    normalized: dict[str, object] = {PROPOSAL_KIND_FIELD: PROPOSAL_KIND_TASK_DISTRIBUTION}
    if distribution_id is not None:
        if not isinstance(distribution_id, str) or distribution_id not in KNOWN_TASK_DISTRIBUTION_CONFIGS:
            available = ", ".join(sorted(KNOWN_TASK_DISTRIBUTION_CONFIGS))
            message = f"task_distribution_id must be one of: {available}"
            raise ValueError(message)
        normalized[TASK_DISTRIBUTION_ID_FIELD] = distribution_id
        normalized[TASK_DISTRIBUTION_CONFIG_PATH_FIELD] = KNOWN_TASK_DISTRIBUTION_CONFIGS[distribution_id]
    if config_path is not None:
        if not isinstance(config_path, str) or config_path not in set(KNOWN_TASK_DISTRIBUTION_CONFIGS.values()):
            available = ", ".join(KNOWN_TASK_DISTRIBUTION_CONFIGS.values())
            message = f"task_distribution_config_path must be one of: {available}"
            raise ValueError(message)
        normalized[TASK_DISTRIBUTION_CONFIG_PATH_FIELD] = config_path
        normalized.setdefault(TASK_DISTRIBUTION_ID_FIELD, _distribution_id_from_path(config_path))
    if REASON_FIELD in raw:
        normalized[REASON_FIELD] = raw[REASON_FIELD]
    if STAGE_BUDGET_PROFILE_FIELD in raw:
        normalized[STAGE_BUDGET_PROFILE_FIELD] = raw[STAGE_BUDGET_PROFILE_FIELD]
    if BUDGET_RATIONALE_FIELD in raw:
        normalized[BUDGET_RATIONALE_FIELD] = raw[BUDGET_RATIONALE_FIELD]
    _validate_budget_metadata(normalized)
    return normalized


def _validate_budget_metadata(task: dict[str, object]) -> None:
    """Validate optional LLM-only stage budget metadata."""
    profile = task.get(STAGE_BUDGET_PROFILE_FIELD)
    if profile is not None and (not isinstance(profile, str) or profile not in DEFAULT_STAGE_BUDGET_PROFILES):
        available = ", ".join(DEFAULT_STAGE_BUDGET_PROFILES)
        message = f"stage_budget_profile must be one of: {available}"
        raise ValueError(message)
    rationale = task.get(BUDGET_RATIONALE_FIELD)
    if rationale is not None and not isinstance(rationale, str):
        message = "budget_rationale metadata must be a string"
        raise ValueError(message)


def _distribution_path_from_reference(reference: Mapping[str, object]) -> str:
    """Return the config path from a normalized distribution reference."""
    value = reference.get(TASK_DISTRIBUTION_CONFIG_PATH_FIELD)
    if not isinstance(value, str) or not value:
        message = "normalized task-distribution reference is missing config path"
        raise ValueError(message)
    return value


def _distribution_id_from_path(config_path: str) -> str:
    """Return a known distribution id for a config path."""
    for distribution_id, path in KNOWN_TASK_DISTRIBUTION_CONFIGS.items():
        if path == config_path:
            return distribution_id
    return Path(config_path).stem


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
        contracts.SHAPE_ELLIPSE: [
            contracts.FIELD_DURATION_SEC,
            contracts.FIELD_SAMPLE_RATE_HZ,
            contracts.FIELD_RADIUS_X,
            contracts.FIELD_RADIUS_Y,
            contracts.FIELD_HEIGHT,
        ],
        contracts.SHAPE_FIGURE_EIGHT: [
            contracts.FIELD_DURATION_SEC,
            contracts.FIELD_SAMPLE_RATE_HZ,
            contracts.FIELD_RADIUS_X,
            contracts.FIELD_RADIUS_Y,
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
    hold_fields = [
        contracts.FIELD_START_HOLD_ENABLED,
        contracts.FIELD_START_HOLD_SEC,
        contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS,
        contracts.FIELD_FINAL_HOLD_ENABLED,
        contracts.FIELD_FINAL_HOLD_SEC,
        contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS,
    ]
    optional_fields = {shape: list(hold_fields) for shape in contracts.SUPPORTED_TRAJECTORY_SHAPES}
    optional_fields[contracts.SHAPE_CIRCLE] = [contracts.FIELD_CENTER, contracts.FIELD_CLOCKWISE, *hold_fields]
    optional_fields[contracts.SHAPE_ELLIPSE] = [contracts.FIELD_CENTER, contracts.FIELD_CLOCKWISE, *hold_fields]
    optional_fields[contracts.SHAPE_FIGURE_EIGHT] = [contracts.FIELD_CENTER, contracts.FIELD_CLOCKWISE, *hold_fields]
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
        contracts.FIELD_RADIUS_X,
        contracts.FIELD_RADIUS_Y,
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
        contracts.FIELD_FINAL_HOLD_ENABLED,
        contracts.FIELD_FINAL_HOLD_SEC,
        contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS,
    )


__all__ = [
    "BUDGET_RATIONALE_FIELD",
    "DEFAULT_STAGE_BUDGET_PROFILES",
    "KNOWN_TASK_DISTRIBUTION_CONFIGS",
    "PROPOSAL_KIND_FIELD",
    "PROPOSAL_KIND_TASK",
    "PROPOSAL_KIND_TASK_DISTRIBUTION",
    "REASON_FIELD",
    "STAGE_BUDGET_PROFILE_FIELD",
    "TASK_DISTRIBUTION_CONFIG_PATH_FIELD",
    "TASK_DISTRIBUTION_ID_FIELD",
    "build_task_prompt_contract",
    "build_task_schema",
    "normalize_proposed_task",
    "task_without_metadata",
    "validate_proposed_task",
]
