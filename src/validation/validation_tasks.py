"""
===============================================================================
validation_tasks.py
===============================================================================
Validate simple trajectory tasks before they can enter training or evaluation.

Responsibilities:
  - Validate minimal hover, curriculum line, circle, vertical, and polyline task dictionaries
  - Check sampled trajectories against deterministic feasibility limits
  - Return structured validation results with diagnostic messages

Design principles:
  - Reject invalid tasks without executing generated code
  - Keep validation deterministic and independent of the simulator

Boundaries:
  - LLM prompting and repair logic belong in LLM modules
  - Policy training and rollout evaluation belong in experiments and evaluation modules
===============================================================================

"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src import trajectories, validation

MIN_TRAJECTORY_SAMPLES = 2
POSITION_ARRAY_NDIM = 2
XYZ_DIMENSIONS = 3
DEFAULT_START_HOLD_SEC = 1.0
START_HOLD_DEFAULT_SHAPES = (
    validation.contracts.SHAPE_CIRCLE,
    validation.contracts.SHAPE_LINE,
    validation.contracts.SHAPE_POLYLINE,
    validation.contracts.SHAPE_SHORT_SLOW_LINE,
    validation.contracts.SHAPE_VERTICAL,
)


@dataclass(frozen=True)
class ValidationLimits:
    """
    Deterministic feasibility limits for early trajectory tasks.

    Parameters
    ----------
    arena_xy_limit
        Absolute XY coordinate bound in meters.
    min_height
        Minimum allowed Z height in meters.
    max_height
        Maximum allowed Z height in meters.
    max_speed_m_s
        Maximum adjacent-sample speed in meters per second.
    max_acceleration_m_s2
        Maximum adjacent-sample acceleration magnitude in meters per second squared.
    min_duration_sec
        Minimum accepted task duration in seconds.
    max_step_distance_m
        Maximum accepted distance between adjacent trajectory samples in meters.

    """

    arena_xy_limit: float = 2.0
    min_height: float = 0.2
    max_height: float = 2.0
    max_speed_m_s: float = 2.0
    max_acceleration_m_s2: float = 5.0
    min_duration_sec: float = 1.0
    max_step_distance_m: float = 0.5


@dataclass(frozen=True)
class ValidationResult:
    """
    Result returned by deterministic task validation.

    Parameters
    ----------
    is_valid
        Whether the task or trajectory passed every check.
    messages
        Human-readable validation diagnostics. Empty when validation succeeds.
    trajectory
        Generated trajectory for valid task-level validation, when available.
    start_hold_enabled
        Whether a stationary start-hold phase is active for this task.
    start_hold_sec
        Effective start-hold duration in seconds after sample-grid alignment.
    exclude_start_hold_from_tracking_metrics
        Whether tracking-only metrics should omit start-hold samples.
    tracking_phase_start_step
        First reference row considered part of moving tracking.
    tracking_phase_start_time_sec
        Reference time in seconds for ``tracking_phase_start_step``.

    """

    is_valid: bool
    messages: tuple[str, ...] = ()
    trajectory: trajectories.primitives.Trajectory | None = None
    start_hold_enabled: bool = False
    start_hold_sec: float = 0.0
    exclude_start_hold_from_tracking_metrics: bool = False
    tracking_phase_start_step: int = 0
    tracking_phase_start_time_sec: float = 0.0


def validate_task(task: Mapping[str, Any], limits: ValidationLimits | None = None) -> ValidationResult:
    """
    Validate a minimal trajectory task dictionary.

    Parameters
    ----------
    task
        Task mapping containing ``task_type``, ``shape``, and shape-specific parameters.
    limits
        Optional feasibility limits. Defaults are used when omitted.

    Returns
    -------
    ValidationResult
        Validation result with diagnostic messages and a generated trajectory when valid.

    """
    active_limits = limits or ValidationLimits()
    if task.get(validation.contracts.FIELD_TASK_TYPE) != validation.contracts.TASK_TYPE_TRAJECTORY:
        return _invalid(f"{validation.contracts.FIELD_TASK_TYPE} must be '{validation.contracts.TASK_TYPE_TRAJECTORY}'")

    shape = task.get(validation.contracts.FIELD_SHAPE)
    if shape in {
        validation.contracts.SHAPE_HOVER,
        validation.contracts.SHAPE_HOVER_STABILIZATION,
        validation.contracts.SHAPE_NEARBY_TARGET_HOVER,
    }:
        return _validate_built_task(_build_hover_trajectory(task), active_limits, task)
    if shape == validation.contracts.SHAPE_CIRCLE:
        return _validate_built_task(_build_circle_trajectory(task), active_limits, task)
    if shape in {validation.contracts.SHAPE_LINE, validation.contracts.SHAPE_SHORT_SLOW_LINE}:
        return _validate_built_task(_build_line_trajectory(task), active_limits, task)
    if shape == validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE:
        return _validate_built_task(_build_start_hold_then_short_line_trajectory(task), active_limits, task)
    if shape == validation.contracts.SHAPE_VERTICAL:
        return _validate_built_task(_build_vertical_trajectory(task), active_limits, task)
    if shape == validation.contracts.SHAPE_POLYLINE:
        return _validate_built_task(_build_polyline_trajectory(task), active_limits, task)
    supported_shapes = ", ".join(validation.contracts.SUPPORTED_TRAJECTORY_SHAPES)
    return _invalid(f"{validation.contracts.FIELD_SHAPE} must be one of: {supported_shapes}")


def validate_trajectory(trajectory: trajectories.primitives.Trajectory, limits: ValidationLimits | None = None) -> ValidationResult:
    """
    Validate a sampled trajectory against deterministic feasibility limits.

    Parameters
    ----------
    trajectory
        Sampled reference trajectory to validate.
    limits
        Optional feasibility limits. Defaults are used when omitted.

    Returns
    -------
    ValidationResult
        Validation result with all detected diagnostic messages.

    """
    active_limits = limits or ValidationLimits()
    messages: list[str] = []
    times = np.asarray(trajectory.times, dtype=float)
    positions = np.asarray(trajectory.positions, dtype=float)

    if times.ndim != 1 or times.shape[0] < MIN_TRAJECTORY_SAMPLES:
        messages.append("trajectory must contain at least two time samples")
    if positions.ndim != POSITION_ARRAY_NDIM or positions.shape[1:] != (XYZ_DIMENSIONS,):
        messages.append("trajectory positions must have shape (num_samples, 3)")
    if times.shape[0] != positions.shape[0]:
        messages.append("times and positions must contain the same number of samples")
    if messages:
        return ValidationResult(is_valid=False, messages=tuple(messages))

    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(positions)):
        messages.append("trajectory must contain only finite values")
    if np.any(np.diff(times) <= 0.0):
        messages.append("trajectory times must be strictly increasing")
    duration_sec = float(times[-1] - times[0])
    if duration_sec < active_limits.min_duration_sec:
        messages.append("duration is shorter than the minimum allowed duration")

    _check_position_bounds(positions=positions, limits=active_limits, messages=messages)
    _check_motion_limits(times=times, positions=positions, limits=active_limits, messages=messages)
    return ValidationResult(is_valid=not messages, messages=tuple(messages))


def _validate_built_task(
    build_result: tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]],
    limits: ValidationLimits,
    task: Mapping[str, Any],
) -> ValidationResult:
    """Validate a task after attempting to build its sampled trajectory."""
    trajectory, build_messages = build_result
    if build_messages or trajectory is None:
        return ValidationResult(is_valid=False, messages=build_messages)
    try:
        trajectory, start_hold_metadata = _apply_task_start_hold(task=task, trajectory=trajectory)
    except (TypeError, ValueError) as exc:
        return ValidationResult(is_valid=False, messages=(str(exc),))
    result = validate_trajectory(trajectory=trajectory, limits=limits)
    return ValidationResult(
        is_valid=result.is_valid,
        messages=result.messages,
        trajectory=trajectory if result.is_valid else None,
        start_hold_enabled=bool(start_hold_metadata["start_hold_enabled"]),
        start_hold_sec=float(start_hold_metadata["start_hold_sec"]),
        exclude_start_hold_from_tracking_metrics=bool(start_hold_metadata["exclude_start_hold_from_tracking_metrics"]),
        tracking_phase_start_step=int(start_hold_metadata["tracking_phase_start_step"]),
        tracking_phase_start_time_sec=float(start_hold_metadata["tracking_phase_start_time_sec"]),
    )


def _build_hover_trajectory(task: Mapping[str, Any]) -> tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]]:
    """Build a hover trajectory from a task mapping, returning messages on failure."""
    try:
        duration_sec = _require_float(task, validation.contracts.FIELD_DURATION_SEC)
        sample_rate_hz = _require_float(task, validation.contracts.FIELD_SAMPLE_RATE_HZ)
        position = _require_sequence(task, validation.contracts.FIELD_POSITION)
        trajectory = trajectories.primitives.make_hover_trajectory(
            position=position,
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
        )
    except (TypeError, ValueError) as exc:
        return None, (str(exc),)
    return trajectory, ()


def _build_circle_trajectory(task: Mapping[str, Any]) -> tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]]:
    """Build a circle trajectory from a task mapping, returning messages on failure."""
    try:
        duration_sec = _require_float(task, validation.contracts.FIELD_DURATION_SEC)
        sample_rate_hz = _require_float(task, validation.contracts.FIELD_SAMPLE_RATE_HZ)
        radius = _require_float(task, validation.contracts.FIELD_RADIUS)
        height = _require_float(task, validation.contracts.FIELD_HEIGHT)
        center = _optional_sequence(task, validation.contracts.FIELD_CENTER, default=(0.0, 0.0))
        clockwise = bool(task.get(validation.contracts.FIELD_CLOCKWISE, False))
        trajectory = trajectories.primitives.make_circle_trajectory(
            radius=radius,
            height=height,
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
            center=center,
            clockwise=clockwise,
        )
    except (TypeError, ValueError) as exc:
        return None, (str(exc),)
    return trajectory, ()


def _build_line_trajectory(task: Mapping[str, Any]) -> tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]]:
    """Build a line trajectory from a task mapping, returning messages on failure."""
    try:
        duration_sec = _require_float(task, validation.contracts.FIELD_DURATION_SEC)
        sample_rate_hz = _require_float(task, validation.contracts.FIELD_SAMPLE_RATE_HZ)
        start = _require_sequence(task, validation.contracts.FIELD_START)
        end = _require_sequence(task, validation.contracts.FIELD_END)
        trajectory = trajectories.primitives.make_line_trajectory(
            start=start,
            end=end,
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
        )
    except (TypeError, ValueError) as exc:
        return None, (str(exc),)
    return trajectory, ()


def _build_start_hold_then_short_line_trajectory(
    task: Mapping[str, Any],
) -> tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]]:
    """Build the moving segment for a held-start then line trajectory."""
    try:
        hold_duration_sec = _require_float(task, validation.contracts.FIELD_HOLD_DURATION_SEC)
        move_duration_sec = _require_float(task, validation.contracts.FIELD_MOVE_DURATION_SEC)
        sample_rate_hz = _require_float(task, validation.contracts.FIELD_SAMPLE_RATE_HZ)
        start = _require_sequence(task, validation.contracts.FIELD_START)
        end = _require_sequence(task, validation.contracts.FIELD_END)
        _validate_positive_curriculum_durations(hold_duration_sec, move_duration_sec)
        trajectory = trajectories.primitives.make_line_trajectory(
            start=start,
            end=end,
            duration_sec=move_duration_sec,
            sample_rate_hz=sample_rate_hz,
        )
    except (TypeError, ValueError) as exc:
        return None, (str(exc),)
    return trajectory, ()


def _build_vertical_trajectory(task: Mapping[str, Any]) -> tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]]:
    """Build a vertical trajectory from a task mapping, returning messages on failure."""
    try:
        duration_sec = _require_float(task, validation.contracts.FIELD_DURATION_SEC)
        sample_rate_hz = _require_float(task, validation.contracts.FIELD_SAMPLE_RATE_HZ)
        xy = _require_sequence(task, validation.contracts.FIELD_XY)
        start_height = _require_float(task, validation.contracts.FIELD_START_HEIGHT)
        end_height = _require_float(task, validation.contracts.FIELD_END_HEIGHT)
        trajectory = trajectories.primitives.make_vertical_trajectory(
            xy=xy,
            start_height=start_height,
            end_height=end_height,
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
        )
    except (TypeError, ValueError) as exc:
        return None, (str(exc),)
    return trajectory, ()


def _build_polyline_trajectory(task: Mapping[str, Any]) -> tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]]:
    """Build a polyline trajectory from a task mapping, returning messages on failure."""
    try:
        duration_sec = _require_float(task, validation.contracts.FIELD_DURATION_SEC)
        sample_rate_hz = _require_float(task, validation.contracts.FIELD_SAMPLE_RATE_HZ)
        points = _require_sequence(task, validation.contracts.FIELD_POINTS)
        trajectory = trajectories.primitives.make_polyline_trajectory(
            points=points,
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
        )
    except (TypeError, ValueError) as exc:
        return None, (str(exc),)
    return trajectory, ()


def _apply_task_start_hold(
    task: Mapping[str, Any],
    trajectory: trajectories.primitives.Trajectory,
) -> tuple[trajectories.primitives.Trajectory, dict[str, Any]]:
    """Prepend a stationary start-hold segment when the task contract requests it."""
    enabled = _start_hold_enabled(task)
    requested_sec = _start_hold_seconds(task, enabled=enabled)
    exclude_from_tracking = _optional_bool(
        task,
        validation.contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS,
        default=enabled,
    )
    if not enabled:
        return trajectory, _start_hold_metadata(
            enabled=False,
            start_hold_sec=0.0,
            exclude_from_tracking=False,
            tracking_phase_start_step=0,
            tracking_phase_start_time_sec=float(np.asarray(trajectory.times, dtype=float)[0]),
        )
    if requested_sec <= 0.0:
        message = f"{validation.contracts.FIELD_START_HOLD_SEC} must be positive when start hold is enabled"
        raise ValueError(message)

    times = np.asarray(trajectory.times, dtype=float)
    positions = np.asarray(trajectory.positions, dtype=float)
    sample_interval = _sample_interval_sec(times)
    hold_steps = max(1, int(round(requested_sec / sample_interval)))
    effective_hold_sec = float(hold_steps * sample_interval)
    hold_times = times[0] + sample_interval * np.arange(hold_steps, dtype=float)
    hold_positions = np.repeat(positions[0].reshape(1, XYZ_DIMENSIONS), repeats=hold_steps, axis=0)
    shifted_times = times + effective_hold_sec
    held_trajectory = trajectories.primitives.Trajectory(
        times=np.concatenate((hold_times, shifted_times)),
        positions=np.vstack((hold_positions, positions)),
    )
    return held_trajectory, _start_hold_metadata(
        enabled=True,
        start_hold_sec=effective_hold_sec,
        exclude_from_tracking=exclude_from_tracking,
        tracking_phase_start_step=hold_steps,
        tracking_phase_start_time_sec=float(shifted_times[0]),
    )


def _start_hold_metadata(
    enabled: bool,
    start_hold_sec: float,
    exclude_from_tracking: bool,
    tracking_phase_start_step: int,
    tracking_phase_start_time_sec: float,
) -> dict[str, Any]:
    """Return JSON-ready start-hold metadata for validation consumers."""
    return {
        "start_hold_enabled": bool(enabled),
        "start_hold_sec": float(start_hold_sec),
        "exclude_start_hold_from_tracking_metrics": bool(exclude_from_tracking),
        "tracking_phase_start_step": int(tracking_phase_start_step),
        "tracking_phase_start_time_sec": float(tracking_phase_start_time_sec),
    }


def _start_hold_enabled(task: Mapping[str, Any]) -> bool:
    """Return whether start-hold is enabled for a task."""
    shape = str(task.get(validation.contracts.FIELD_SHAPE, ""))
    default_enabled = shape in START_HOLD_DEFAULT_SHAPES or shape == validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE
    return _optional_bool(task, validation.contracts.FIELD_START_HOLD_ENABLED, default=default_enabled)


def _start_hold_seconds(task: Mapping[str, Any], enabled: bool) -> float:
    """Return configured or default start-hold seconds for a task."""
    if not enabled:
        return 0.0
    if validation.contracts.FIELD_START_HOLD_SEC in task:
        return _optional_float(task, validation.contracts.FIELD_START_HOLD_SEC, default=DEFAULT_START_HOLD_SEC)
    if str(task.get(validation.contracts.FIELD_SHAPE, "")) == validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE:
        return _require_float(task, validation.contracts.FIELD_HOLD_DURATION_SEC)
    return DEFAULT_START_HOLD_SEC


def _sample_interval_sec(times: np.ndarray) -> float:
    """Return a representative positive sample interval from built trajectory times."""
    diffs = np.diff(times)
    if diffs.size == 0 or np.any(diffs <= 0.0):
        message = "trajectory times must be strictly increasing before start-hold can be applied"
        raise ValueError(message)
    sample_interval = float(np.median(diffs))
    if not np.isfinite(sample_interval) or sample_interval <= 0.0:
        message = "trajectory sample interval must be finite and positive"
        raise ValueError(message)
    return sample_interval


def _optional_bool(task: Mapping[str, Any], key: str, default: bool) -> bool:
    """Read an optional boolean field without treating strings as truthy."""
    if key not in task:
        return default
    value = task[key]
    if not isinstance(value, bool):
        message = f"{key} must be a boolean"
        raise TypeError(message)
    return value


def _optional_float(task: Mapping[str, Any], key: str, default: float) -> float:
    """Read an optional finite float field."""
    if key not in task:
        return default
    value = float(task[key])
    if not np.isfinite(value):
        message = f"{key} must be finite"
        raise ValueError(message)
    return value


def _check_position_bounds(positions: np.ndarray, limits: ValidationLimits, messages: list[str]) -> None:
    """Append diagnostic messages for positions outside the configured arena."""
    xy_abs = np.abs(positions[:, :2])
    z_values = positions[:, 2]
    if np.any(xy_abs > limits.arena_xy_limit):
        messages.append("trajectory leaves the XY arena bounds")
    if np.any(z_values < limits.min_height) or np.any(z_values > limits.max_height):
        messages.append("trajectory height is outside allowed bounds")


def _check_motion_limits(times: np.ndarray, positions: np.ndarray, limits: ValidationLimits, messages: list[str]) -> None:
    """Append diagnostic messages for excessive sample jumps, speed, or acceleration."""
    deltas = np.diff(positions, axis=0)
    dt = np.diff(times)
    if np.any(dt <= 0.0):
        return

    step_distances = np.linalg.norm(deltas, axis=1)
    speeds = step_distances / dt
    if np.any(step_distances > limits.max_step_distance_m):
        messages.append("trajectory contains a discontinuous jump")
    if np.any(speeds > limits.max_speed_m_s):
        messages.append("trajectory exceeds the maximum speed")

    if speeds.shape[0] >= MIN_TRAJECTORY_SAMPLES:
        velocities = deltas / dt[:, np.newaxis]
        acceleration_dt = (dt[:-1] + dt[1:]) / 2.0
        accelerations = np.linalg.norm(np.diff(velocities, axis=0), axis=1) / acceleration_dt
        if np.any(accelerations > limits.max_acceleration_m_s2):
            messages.append("trajectory exceeds the maximum acceleration")


def _require_float(task: Mapping[str, Any], key: str) -> float:
    """Read a required finite float value from a task mapping."""
    if key not in task:
        message = f"{key} is required"
        raise ValueError(message)
    value = float(task[key])
    if not np.isfinite(value):
        message = f"{key} must be finite"
        raise ValueError(message)
    return value


def _require_sequence(task: Mapping[str, Any], key: str) -> Sequence[Any]:
    """Read a required numeric sequence from a task mapping."""
    if key not in task:
        message = f"{key} is required"
        raise ValueError(message)
    value = task[key]
    if isinstance(value, str) or not isinstance(value, Sequence):
        message = f"{key} must be a numeric sequence"
        raise TypeError(message)
    return value


def _optional_sequence(task: Mapping[str, Any], key: str, default: Sequence[Any]) -> Sequence[Any]:
    """Read an optional numeric sequence from a task mapping."""
    if key not in task:
        return default
    return _require_sequence(task, key)


def _invalid(message: str) -> ValidationResult:
    """Create an invalid validation result with one diagnostic message."""
    return ValidationResult(is_valid=False, messages=(message,))


def _validate_positive_curriculum_durations(hold_duration_sec: float, move_duration_sec: float) -> None:
    """Validate positive curriculum task durations."""
    if hold_duration_sec <= 0.0 or move_duration_sec <= 0.0:
        message = "hold_duration_sec and move_duration_sec must be positive"
        raise ValueError(message)
