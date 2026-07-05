"""
===============================================================================
validation_tasks.py
===============================================================================
Validate simple trajectory tasks before they can enter training or evaluation.

Responsibilities:
  - Validate minimal hover and circle task dictionaries
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

from src import trajectories

from . import validation_contracts as contracts

MIN_TRAJECTORY_SAMPLES = 2
POSITION_ARRAY_NDIM = 2
XYZ_DIMENSIONS = 3


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

    """

    is_valid: bool
    messages: tuple[str, ...] = ()
    trajectory: trajectories.primitives.Trajectory | None = None


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
    if task.get(contracts.FIELD_TASK_TYPE) != contracts.TASK_TYPE_TRAJECTORY:
        return _invalid(f"{contracts.FIELD_TASK_TYPE} must be '{contracts.TASK_TYPE_TRAJECTORY}'")

    shape = task.get(contracts.FIELD_SHAPE)
    if shape == contracts.SHAPE_HOVER:
        return _validate_built_task(_build_hover_trajectory(task), active_limits)
    if shape == contracts.SHAPE_CIRCLE:
        return _validate_built_task(_build_circle_trajectory(task), active_limits)
    supported_shapes = ", ".join(contracts.SUPPORTED_TRAJECTORY_SHAPES)
    return _invalid(f"{contracts.FIELD_SHAPE} must be one of: {supported_shapes}")


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
) -> ValidationResult:
    """Validate a task after attempting to build its sampled trajectory."""
    trajectory, build_messages = build_result
    if build_messages or trajectory is None:
        return ValidationResult(is_valid=False, messages=build_messages)
    result = validate_trajectory(trajectory=trajectory, limits=limits)
    return ValidationResult(is_valid=result.is_valid, messages=result.messages, trajectory=trajectory if result.is_valid else None)


def _build_hover_trajectory(task: Mapping[str, Any]) -> tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]]:
    """Build a hover trajectory from a task mapping, returning messages on failure."""
    try:
        duration_sec = _require_float(task, contracts.FIELD_DURATION_SEC)
        sample_rate_hz = _require_float(task, contracts.FIELD_SAMPLE_RATE_HZ)
        position = _require_sequence(task, contracts.FIELD_POSITION)
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
        duration_sec = _require_float(task, contracts.FIELD_DURATION_SEC)
        sample_rate_hz = _require_float(task, contracts.FIELD_SAMPLE_RATE_HZ)
        radius = _require_float(task, contracts.FIELD_RADIUS)
        height = _require_float(task, contracts.FIELD_HEIGHT)
        center = _optional_sequence(task, contracts.FIELD_CENTER, default=(0.0, 0.0))
        clockwise = bool(task.get(contracts.FIELD_CLOCKWISE, False))
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


def _require_sequence(task: Mapping[str, Any], key: str) -> Sequence[float]:
    """Read a required numeric sequence from a task mapping."""
    if key not in task:
        message = f"{key} is required"
        raise ValueError(message)
    value = task[key]
    if isinstance(value, str) or not isinstance(value, Sequence):
        message = f"{key} must be a numeric sequence"
        raise TypeError(message)
    return value


def _optional_sequence(task: Mapping[str, Any], key: str, default: Sequence[float]) -> Sequence[float]:
    """Read an optional numeric sequence from a task mapping."""
    if key not in task:
        return default
    return _require_sequence(task, key)


def _invalid(message: str) -> ValidationResult:
    """Create an invalid validation result with one diagnostic message."""
    return ValidationResult(is_valid=False, messages=(message,))
