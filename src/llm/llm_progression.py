"""
===============================================================================
llm_progression.py
===============================================================================
Define deterministic curriculum progression and duplicate-transition semantics.

Responsibilities:
  - Distinguish exact repeats from valid same-family difficulty increases
  - Provide stable transition reason labels for proposal repair and fallback
  - Normalize concrete shapes, task-distribution ids, and sampled families

Design principles:
  - Keep transition rules deterministic and independent of training state
  - Treat exact repeats conservatively while allowing explicit skill progressions
  - Preserve hover/vertical loop protection without blocking line progression

Boundaries:
  - LLM prompt construction belongs in llm_prompts.py
  - PPO stage orchestration and task sampling belong in experiments modules
===============================================================================

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from src import validation

TRANSITION_EXACT_DUPLICATE = "exact_duplicate"
TRANSITION_VALID_DIFFICULTY_INCREASE = "valid_difficulty_increase"
TRANSITION_VALID_NEW_SKILL = "valid_new_skill"
TRANSITION_HOVER_VERTICAL_LOOP = "hover_vertical_loop"
TRANSITION_TOO_EARLY_BROAD = "too_early_broad"
TRANSITION_FALLBACK_CONSOLIDATION = "fallback_consolidation"
TRANSITION_INVALID_REPEAT = "invalid_repeat"
HOVER_VERTICAL_LOOP_WINDOW_SIZE = 3

SEMANTIC_SHAPE_ANGLED_VERTICAL = "angled_vertical"
SEMANTIC_SHAPE_DELAYED_ALTITUDE_POLYLINE = "delayed_altitude_polyline"
SEMANTIC_SHAPE_L_SHAPE = "l_shape"
SEMANTIC_SHAPE_MULTI_HEIGHT_POLYLINE = "multi_height_polyline"
SEMANTIC_SHAPE_RECTANGLE = "rectangle"
SEMANTIC_SHAPE_SHORT_LINE = "short_line"
SEMANTIC_SHAPE_TRIANGLE = "triangle"
SEMANTIC_SHAPE_VERTICAL_UP_DOWN = "vertical_up_down"
SEMANTIC_SHAPE_ZIGZAG = "zigzag"

_HOVER_SHAPES = {
    validation.contracts.SHAPE_HOVER,
    validation.contracts.SHAPE_HOVER_STABILIZATION,
    validation.contracts.SHAPE_NEARBY_TARGET_HOVER,
}
_PURE_VERTICAL_SHAPES = {
    validation.contracts.SHAPE_VERTICAL,
    SEMANTIC_SHAPE_VERTICAL_UP_DOWN,
}
_XY_LINE_SHAPES = {
    validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE,
    validation.contracts.SHAPE_SHORT_SLOW_LINE,
    SEMANTIC_SHAPE_SHORT_LINE,
    validation.contracts.SHAPE_LINE,
}
_ALTITUDE_COMBINED_SHAPES = {
    SEMANTIC_SHAPE_ANGLED_VERTICAL,
    SEMANTIC_SHAPE_DELAYED_ALTITUDE_POLYLINE,
    SEMANTIC_SHAPE_MULTI_HEIGHT_POLYLINE,
}
_TURN_POLYLINE_SHAPES = {
    validation.contracts.SHAPE_POLYLINE,
    SEMANTIC_SHAPE_L_SHAPE,
    SEMANTIC_SHAPE_ZIGZAG,
    SEMANTIC_SHAPE_TRIANGLE,
    SEMANTIC_SHAPE_RECTANGLE,
}
_CURVE_SHAPES = {
    validation.contracts.SHAPE_CIRCLE,
    validation.contracts.SHAPE_ELLIPSE,
    validation.contracts.SHAPE_FIGURE_EIGHT,
}
_BROAD_DISTRIBUTION_IDS = {
    "tracking_small",
    "tracking_medium",
    "tracking_broad",
}

DISTRIBUTION_SEMANTIC_SHAPES = {
    "bootstrap_randomized_hover_target": validation.contracts.SHAPE_HOVER_STABILIZATION,
    "hover_bootstrap": validation.contracts.SHAPE_HOVER_STABILIZATION,
    "vertical_bootstrap": validation.contracts.SHAPE_VERTICAL,
    "vertical_up_down_bootstrap": SEMANTIC_SHAPE_VERTICAL_UP_DOWN,
    "short_line_bootstrap": validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE,
    "line_bootstrap": validation.contracts.SHAPE_LINE,
    "angled_vertical_bootstrap": SEMANTIC_SHAPE_ANGLED_VERTICAL,
    "delayed_altitude_polyline_bootstrap": SEMANTIC_SHAPE_DELAYED_ALTITUDE_POLYLINE,
    "multi_height_polyline_bootstrap": SEMANTIC_SHAPE_MULTI_HEIGHT_POLYLINE,
    "polyline_bootstrap": validation.contracts.SHAPE_POLYLINE,
    "l_shape_bootstrap": SEMANTIC_SHAPE_L_SHAPE,
    "zigzag_bootstrap": SEMANTIC_SHAPE_ZIGZAG,
    "triangle_bootstrap": SEMANTIC_SHAPE_TRIANGLE,
    "rectangle_bootstrap": SEMANTIC_SHAPE_RECTANGLE,
    "circle_bootstrap": validation.contracts.SHAPE_CIRCLE,
    "ellipse_bootstrap": validation.contracts.SHAPE_ELLIPSE,
}

SAMPLED_FAMILY_SEMANTIC_SHAPES = {
    "hover_stabilization": validation.contracts.SHAPE_HOVER_STABILIZATION,
    "takeoff_stabilization": validation.contracts.SHAPE_VERTICAL,
    "vertical_up_down": SEMANTIC_SHAPE_VERTICAL_UP_DOWN,
    "angled_vertical": SEMANTIC_SHAPE_ANGLED_VERTICAL,
    "line": validation.contracts.SHAPE_LINE,
    "start_hold_then_line": validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE,
    "polyline": validation.contracts.SHAPE_POLYLINE,
    "l_shape": SEMANTIC_SHAPE_L_SHAPE,
    "zigzag": SEMANTIC_SHAPE_ZIGZAG,
    "triangle": SEMANTIC_SHAPE_TRIANGLE,
    "multi_height_polyline": SEMANTIC_SHAPE_MULTI_HEIGHT_POLYLINE,
    "delayed_altitude_polyline": SEMANTIC_SHAPE_DELAYED_ALTITUDE_POLYLINE,
    "rectangle": SEMANTIC_SHAPE_RECTANGLE,
    "square": SEMANTIC_SHAPE_RECTANGLE,
    "circle": validation.contracts.SHAPE_CIRCLE,
    "ellipse": validation.contracts.SHAPE_ELLIPSE,
    "figure_eight": validation.contracts.SHAPE_FIGURE_EIGHT,
}

ALLOWED_SEMANTIC_TRANSITIONS = {
    (validation.contracts.SHAPE_HOVER_STABILIZATION, validation.contracts.SHAPE_VERTICAL),
    (validation.contracts.SHAPE_VERTICAL, SEMANTIC_SHAPE_ANGLED_VERTICAL),
    (validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE, validation.contracts.SHAPE_LINE),
    (validation.contracts.SHAPE_SHORT_SLOW_LINE, validation.contracts.SHAPE_LINE),
    (SEMANTIC_SHAPE_SHORT_LINE, validation.contracts.SHAPE_LINE),
    (validation.contracts.SHAPE_LINE, SEMANTIC_SHAPE_ANGLED_VERTICAL),
    (validation.contracts.SHAPE_LINE, SEMANTIC_SHAPE_DELAYED_ALTITUDE_POLYLINE),
    (validation.contracts.SHAPE_LINE, SEMANTIC_SHAPE_MULTI_HEIGHT_POLYLINE),
    (validation.contracts.SHAPE_LINE, validation.contracts.SHAPE_POLYLINE),
    (validation.contracts.SHAPE_LINE, SEMANTIC_SHAPE_L_SHAPE),
    (validation.contracts.SHAPE_LINE, SEMANTIC_SHAPE_ZIGZAG),
    (validation.contracts.SHAPE_POLYLINE, SEMANTIC_SHAPE_L_SHAPE),
    (validation.contracts.SHAPE_POLYLINE, SEMANTIC_SHAPE_ZIGZAG),
    (validation.contracts.SHAPE_POLYLINE, SEMANTIC_SHAPE_TRIANGLE),
    (validation.contracts.SHAPE_POLYLINE, SEMANTIC_SHAPE_RECTANGLE),
    (SEMANTIC_SHAPE_L_SHAPE, SEMANTIC_SHAPE_ZIGZAG),
    (SEMANTIC_SHAPE_ZIGZAG, SEMANTIC_SHAPE_TRIANGLE),
    (SEMANTIC_SHAPE_ZIGZAG, SEMANTIC_SHAPE_RECTANGLE),
    (validation.contracts.SHAPE_CIRCLE, validation.contracts.SHAPE_ELLIPSE),
}


@dataclass(frozen=True)
class _ProgressionIdentity:
    """Normalized identity used for deterministic progression comparison."""

    distribution_id: str | None
    semantic_shape: str | None


def is_valid_progression_transition(
    previous_stage: Mapping[str, Any] | str | None,
    candidate_stage: Mapping[str, Any] | str | None,
    history: Sequence[Mapping[str, Any] | str] = (),
    diagnostics: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    """
    Return whether a candidate is a valid deterministic progression.

    Parameters
    ----------
    previous_stage
        Latest accepted stage context, shape, or distribution reference.
    candidate_stage
        Candidate stage context, shape, distribution reference, or sampled task metadata.
    history
        Optional recent accepted stages used to preserve hover/vertical loop protection.
    diagnostics
        Optional policy metadata such as stage index, max stages, or explicit fallback allowance.

    Returns
    -------
    tuple[bool, str]
        Boolean acceptance plus a stable reason label.

    """
    previous = _identity_from_stage(previous_stage)
    candidate = _identity_from_stage(candidate_stage)
    if previous.semantic_shape is None or candidate.semantic_shape is None:
        return True, TRANSITION_VALID_NEW_SKILL

    if _is_too_early_broad(candidate.distribution_id, diagnostics):
        return False, TRANSITION_TOO_EARLY_BROAD
    if _fallback_consolidation_allowed(candidate.distribution_id, diagnostics):
        return True, TRANSITION_FALLBACK_CONSOLIDATION
    if previous.distribution_id is not None and previous.distribution_id == candidate.distribution_id:
        return False, TRANSITION_EXACT_DUPLICATE
    if previous.semantic_shape == candidate.semantic_shape:
        return False, TRANSITION_EXACT_DUPLICATE
    if _hover_vertical_loop_detected(history=history, candidate_shape=candidate.semantic_shape):
        return False, TRANSITION_HOVER_VERTICAL_LOOP
    if (previous.semantic_shape, candidate.semantic_shape) in ALLOWED_SEMANTIC_TRANSITIONS:
        return True, TRANSITION_VALID_DIFFICULTY_INCREASE
    if _progression_bucket(previous.semantic_shape) != _progression_bucket(candidate.semantic_shape):
        return True, TRANSITION_VALID_NEW_SKILL
    return False, TRANSITION_INVALID_REPEAT


def semantic_shape_for_distribution(distribution_id: str | None) -> str | None:
    """Return the curriculum semantic shape represented by a known distribution id."""
    if distribution_id is None:
        return None
    return DISTRIBUTION_SEMANTIC_SHAPES.get(distribution_id)


def semantic_shape_for_sampled_family(sampled_family: str | None) -> str | None:
    """Return the curriculum semantic shape represented by a sampled task family."""
    if sampled_family is None:
        return None
    return SAMPLED_FAMILY_SEMANTIC_SHAPES.get(sampled_family)


def _identity_from_stage(stage: Mapping[str, Any] | str | None) -> _ProgressionIdentity:
    """Return a normalized progression identity from a stage-like value."""
    if stage is None:
        return _ProgressionIdentity(distribution_id=None, semantic_shape=None)
    if isinstance(stage, str):
        return _ProgressionIdentity(distribution_id=None, semantic_shape=_normalize_shape(stage))
    distribution_id = _optional_text(stage.get("task_distribution_id"))
    sample_metadata = stage.get("resolved_task_sample_metadata")
    sampled_family = _optional_text(stage.get("task_distribution_sampled_family"))
    if sampled_family is None and isinstance(sample_metadata, Mapping):
        sampled_family = _optional_text(sample_metadata.get("task_distribution_sampled_family"))
    semantic_shape = (
        semantic_shape_for_sampled_family(sampled_family)
        or _optional_text(stage.get("semantic_shape"))
        or semantic_shape_for_distribution(distribution_id)
        or _shape_from_stage_mapping(stage)
    )
    return _ProgressionIdentity(distribution_id=distribution_id, semantic_shape=_normalize_shape(semantic_shape))


def _shape_from_stage_mapping(stage: Mapping[str, Any]) -> str | None:
    """Return the best concrete shape field from a stage-like mapping."""
    for key in (
        validation.contracts.FIELD_SHAPE,
        "accepted_stage_task_shape",
        "resolved_task_shape",
        "task_shape",
        "requested_stage_task_shape",
    ):
        value = _optional_text(stage.get(key))
        if value is not None:
            return value
    task = stage.get("resolved_task") or stage.get("task")
    if isinstance(task, Mapping):
        return _optional_text(task.get(validation.contracts.FIELD_SHAPE))
    return None


def _normalize_shape(shape: str | None) -> str | None:
    """Normalize aliases that should share one exact-repeat identity."""
    if shape is None:
        return None
    if shape in _HOVER_SHAPES:
        return validation.contracts.SHAPE_HOVER_STABILIZATION
    return shape


def _optional_text(value: Any) -> str | None:
    """Return stripped text for non-empty optional values."""
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _progression_bucket(shape: str) -> str:
    """Return a coarse progression bucket for same-bucket repeat detection."""
    if shape in _HOVER_SHAPES or shape == validation.contracts.SHAPE_HOVER_STABILIZATION:
        return "hover"
    if shape in _PURE_VERTICAL_SHAPES:
        return "vertical_only"
    if shape in _XY_LINE_SHAPES:
        return "xy_line"
    if shape in _ALTITUDE_COMBINED_SHAPES:
        return "altitude_combined"
    if shape in _TURN_POLYLINE_SHAPES:
        return "turn_polyline"
    if shape in _CURVE_SHAPES:
        return "curve"
    return shape


def _hover_vertical_loop_detected(*, history: Sequence[Mapping[str, Any] | str], candidate_shape: str) -> bool:
    """Return whether the recent history plus candidate forms a hover/vertical loop."""
    recent_shapes = [_identity_from_stage(stage).semantic_shape for stage in history]
    buckets = [_progression_bucket(shape) for shape in (*recent_shapes, candidate_shape) if shape is not None]
    if len(buckets) < HOVER_VERTICAL_LOOP_WINDOW_SIZE:
        return False
    window = buckets[-HOVER_VERTICAL_LOOP_WINDOW_SIZE:]
    return all(bucket in {"hover", "vertical_only"} for bucket in window) and len(set(window)) > 1


def _is_too_early_broad(distribution_id: str | None, diagnostics: Mapping[str, Any] | None) -> bool:
    """Return whether a broad distribution is being used before the late curriculum."""
    if distribution_id not in _BROAD_DISTRIBUTION_IDS or diagnostics is None:
        return False
    if bool(diagnostics.get("allow_broad_fallback", False)):
        return False
    stage_index = diagnostics.get("stage_index")
    max_stages = diagnostics.get("max_stages")
    if stage_index is None or max_stages is None:
        return False
    return int(stage_index) < max(int(max_stages) - 1, 3)


def _fallback_consolidation_allowed(distribution_id: str | None, diagnostics: Mapping[str, Any] | None) -> bool:
    """Return whether a broad fallback has explicit consolidation permission."""
    if distribution_id not in _BROAD_DISTRIBUTION_IDS or diagnostics is None:
        return False
    return bool(diagnostics.get("allow_broad_fallback", False))
