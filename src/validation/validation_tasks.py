"""
===============================================================================
validation_tasks.py
===============================================================================
Validate simple trajectory tasks before they can enter training or evaluation.

Responsibilities:
  - Validate minimal hover, curriculum line, circle, ellipse, figure-eight, vertical, and polyline task dictionaries
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

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src import trajectories, validation

MIN_TRAJECTORY_SAMPLES = 2
POSITION_ARRAY_NDIM = 2
XYZ_DIMENSIONS = 3
DEFAULT_START_HOLD_SEC = 1.2
DEFAULT_FINAL_HOLD_SEC = 1.0
START_HOLD_DEFAULT_SHAPES = (*validation.contracts.SUPPORTED_TRAJECTORY_SHAPES, validation.contracts.SHAPE_BASIC_TRAINING_SHOW)
FINAL_HOLD_DEFAULT_SHAPES = (*validation.contracts.SUPPORTED_TRAJECTORY_SHAPES, validation.contracts.SHAPE_BASIC_TRAINING_SHOW)
BASIC_SHOW_FINAL_HOLD_SHAPE = "final_hold"
BASIC_SHOW_CONTINUITY_TOLERANCE_M = 1.0e-6


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
    final_hold_enabled
        Whether a stationary final-hold phase is active for this task.
    final_hold_sec
        Effective final-hold duration in seconds after sample-grid alignment.
    exclude_final_hold_from_tracking_metrics
        Whether tracking-only metrics should omit final-hold samples.
    tracking_phase_end_step
        Exclusive reference row where moving tracking ends.
    tracking_phase_end_time_sec
        Reference time in seconds at the end of moving tracking.

    """

    is_valid: bool
    messages: tuple[str, ...] = ()
    trajectory: trajectories.primitives.Trajectory | None = None
    start_hold_enabled: bool = False
    start_hold_sec: float = 0.0
    exclude_start_hold_from_tracking_metrics: bool = False
    tracking_phase_start_step: int = 0
    tracking_phase_start_time_sec: float = 0.0
    final_hold_enabled: bool = False
    final_hold_sec: float = 0.0
    exclude_final_hold_from_tracking_metrics: bool = False
    tracking_phase_end_step: int = 0
    tracking_phase_end_time_sec: float = 0.0


def validate_task(task: Mapping[str, Any], limits: ValidationLimits | None = None) -> ValidationResult:  # noqa: PLR0911
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
    if shape == validation.contracts.SHAPE_BASIC_TRAINING_SHOW:
        return _validate_built_task(_build_basic_training_show_trajectory(task), active_limits, task)
    if shape == validation.contracts.SHAPE_CIRCLE:
        return _validate_built_task(_build_circle_trajectory(task), active_limits, task)
    if shape == validation.contracts.SHAPE_ELLIPSE:
        return _validate_built_task(_build_ellipse_trajectory(task), active_limits, task)
    if shape == validation.contracts.SHAPE_FIGURE_EIGHT:
        return _validate_built_task(_build_figure_eight_trajectory(task), active_limits, task)
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
        trajectory, final_hold_metadata = _apply_task_final_hold(task=task, trajectory=trajectory)
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
        final_hold_enabled=bool(final_hold_metadata["final_hold_enabled"]),
        final_hold_sec=float(final_hold_metadata["final_hold_sec"]),
        exclude_final_hold_from_tracking_metrics=bool(final_hold_metadata["exclude_final_hold_from_tracking_metrics"]),
        tracking_phase_end_step=int(final_hold_metadata["tracking_phase_end_step"]),
        tracking_phase_end_time_sec=float(final_hold_metadata["tracking_phase_end_time_sec"]),
    )


def _build_basic_training_show_trajectory(
    task: Mapping[str, Any],
) -> tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]]:
    """Build a continuous composed basic-training-show trajectory."""
    try:
        sample_rate_hz = _require_float(task, validation.contracts.FIELD_SAMPLE_RATE_HZ)
        segments = _require_show_segments(task)
        combined_times: list[np.ndarray] = []
        combined_positions: list[np.ndarray] = []
        previous_end: np.ndarray | None = None
        current_time_end = 0.0
        built_segment_count = 0
        for segment_index, segment in enumerate(segments):
            shape = _show_segment_shape(segment)
            if shape == BASIC_SHOW_FINAL_HOLD_SHAPE:
                if segment_index != len(segments) - 1:
                    _raise_basic_show_final_hold_not_last()
                _validate_basic_show_final_hold_segment(segment=segment, previous_end=previous_end)
                continue

            trajectory = _build_basic_show_segment(segment=segment, sample_rate_hz=sample_rate_hz)
            local_times = np.asarray(trajectory.times, dtype=float) - float(trajectory.times[0])
            local_positions = np.asarray(trajectory.positions, dtype=float)
            if previous_end is not None:
                gap = float(np.linalg.norm(local_positions[0] - previous_end))
                if gap > BASIC_SHOW_CONTINUITY_TOLERANCE_M:
                    _raise_basic_show_discontinuity(segment_index=segment_index, gap=gap)

            adjusted_times = _basic_show_adjusted_times(
                local_times=local_times,
                current_time_end=current_time_end,
                is_first_segment=built_segment_count == 0,
            )
            combined_times.append(adjusted_times)
            combined_positions.append(local_positions)
            previous_end = np.array(local_positions[-1], dtype=float, copy=True)
            current_time_end = float(adjusted_times[-1])
            built_segment_count += 1

        if not combined_times:
            _raise_basic_show_no_motion_segments()
        return (
            trajectories.primitives.Trajectory(
                times=np.concatenate(combined_times),
                positions=np.vstack(combined_positions),
            ),
            (),
        )
    except (TypeError, ValueError) as exc:
        return None, (str(exc),)


def _raise_basic_show_final_hold_not_last() -> None:
    """Raise for misplaced final-hold metadata segments."""
    message = "basic_training_show final_hold segment must be last"
    raise ValueError(message)


def _raise_basic_show_discontinuity(segment_index: int, gap: float) -> None:
    """Raise for discontinuous adjacent basic-show segments."""
    message = f"basic_training_show segment {segment_index - 1}->{segment_index} is discontinuous by {gap:.6g} m"
    raise ValueError(message)


def _raise_basic_show_no_motion_segments() -> None:
    """Raise when a basic show contains no generated motion or hold segment."""
    message = "basic_training_show must contain at least one non-final-hold segment"
    raise ValueError(message)


def _build_basic_show_segment(segment: Mapping[str, Any], sample_rate_hz: float) -> trajectories.primitives.Trajectory:
    """Build one absolute-position segment for a basic training show."""
    shape = _show_segment_shape(segment)
    duration_sec = _show_segment_duration(segment)
    if shape in {"hover", "hover_stabilization", "hold"}:
        position = _show_segment_xyz(segment, validation.contracts.FIELD_SEGMENT_START)
        _validate_declared_segment_end(segment=segment, actual_end=position)
        return trajectories.primitives.make_hover_trajectory(
            position=position.tolist(),
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
        )
    if shape in {"line", "horizontal_line", "diagonal_line", "slanted_line", "vertical"}:
        start = _show_segment_xyz(segment, validation.contracts.FIELD_SEGMENT_START)
        end = _show_segment_xyz(segment, validation.contracts.FIELD_SEGMENT_END)
        return trajectories.primitives.make_line_trajectory(
            start=start.tolist(),
            end=end.tolist(),
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
        )
    if shape in {"polyline", "l_shape", "zigzag"}:
        points = _show_segment_points(segment)
        _validate_declared_segment_start(segment=segment, actual_start=points[0])
        _validate_declared_segment_end(segment=segment, actual_end=points[-1])
        return trajectories.primitives.make_polyline_trajectory(
            points=points.tolist(),
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
        )
    if shape == "ellipse":
        start = _show_segment_xyz(segment, validation.contracts.FIELD_SEGMENT_START)
        radius_x = _show_segment_float(segment, "radius_x_m")
        radius_y = _show_segment_float(segment, "radius_y_m")
        phase_rad = math.radians(float(segment.get("phase_deg", segment.get("start_angle_deg", 180.0))))
        clockwise = _show_segment_bool(segment, validation.contracts.FIELD_CLOCKWISE, default=False)
        center = np.asarray(segment.get(validation.contracts.FIELD_CENTER), dtype=float) if validation.contracts.FIELD_CENTER in segment else None
        if center is None:
            center = np.array(
                [start[0] - radius_x * math.cos(phase_rad), start[1] - radius_y * math.sin(phase_rad)],
                dtype=float,
            )
        if center.shape != (2,) or not np.all(np.isfinite(center)):
            message = "ellipse segment center must contain two finite values"
            raise ValueError(message)
        trajectory = trajectories.primitives.make_ellipse_trajectory(
            radius_x=radius_x,
            radius_y=radius_y,
            height=float(start[2]),
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
            center=center.tolist(),
            clockwise=clockwise,
            phase_rad=phase_rad,
        )
        _validate_declared_segment_start(segment=segment, actual_start=trajectory.positions[0])
        _validate_declared_segment_end(segment=segment, actual_end=trajectory.positions[-1])
        return trajectory
    message = f"unsupported basic_training_show segment_shape: {shape}"
    raise ValueError(message)


def _require_show_segments(task: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    """Return validated basic-show segment mappings."""
    raw_segments = task.get(validation.contracts.FIELD_SEGMENTS)
    if isinstance(raw_segments, str) or not isinstance(raw_segments, Sequence):
        message = f"{validation.contracts.FIELD_SEGMENTS} must be a non-empty sequence"
        raise TypeError(message)
    if not raw_segments:
        message = f"{validation.contracts.FIELD_SEGMENTS} must be non-empty"
        raise ValueError(message)
    segments: list[Mapping[str, Any]] = []
    for index, segment in enumerate(raw_segments):
        if not isinstance(segment, Mapping):
            message = f"basic_training_show segment {index} must be a mapping"
            raise TypeError(message)
        segments.append(segment)
    return tuple(segments)


def _show_segment_shape(segment: Mapping[str, Any]) -> str:
    """Return a normalized segment shape."""
    value = segment.get(validation.contracts.FIELD_SEGMENT_SHAPE, segment.get(validation.contracts.FIELD_SHAPE))
    if value is None or not str(value).strip():
        message = f"basic_training_show segment requires {validation.contracts.FIELD_SEGMENT_SHAPE}"
        raise ValueError(message)
    return str(value).strip().lower()


def _show_segment_duration(segment: Mapping[str, Any]) -> float:
    """Return a positive segment duration."""
    key = validation.contracts.FIELD_SEGMENT_DURATION_SEC
    if key not in segment:
        key = validation.contracts.FIELD_DURATION_SEC
    return _show_segment_float(segment, key)


def _show_segment_float(segment: Mapping[str, Any], key: str) -> float:
    """Read a finite positive segment float."""
    if key not in segment:
        message = f"basic_training_show segment requires {key}"
        raise ValueError(message)
    value = float(segment[key])
    if not np.isfinite(value) or value <= 0.0:
        message = f"basic_training_show segment {key} must be finite and positive"
        raise ValueError(message)
    return value


def _show_segment_xyz(segment: Mapping[str, Any], key: str) -> np.ndarray:
    """Read one finite XYZ segment vector."""
    if key not in segment:
        message = f"basic_training_show segment requires {key}"
        raise ValueError(message)
    array = np.asarray(segment[key], dtype=float)
    if array.shape != (XYZ_DIMENSIONS,) or not np.all(np.isfinite(array)):
        message = f"basic_training_show segment {key} must contain three finite values"
        raise ValueError(message)
    return array


def _show_segment_points(segment: Mapping[str, Any]) -> np.ndarray:
    """Read finite XYZ waypoints from a composed-show segment."""
    key = validation.contracts.FIELD_SEGMENT_POINTS if validation.contracts.FIELD_SEGMENT_POINTS in segment else validation.contracts.FIELD_POINTS
    if key not in segment:
        message = f"basic_training_show polyline-like segment requires {validation.contracts.FIELD_SEGMENT_POINTS}"
        raise ValueError(message)
    points = np.asarray(segment[key], dtype=float)
    if points.ndim != POSITION_ARRAY_NDIM or points.shape[1:] != (XYZ_DIMENSIONS,) or points.shape[0] < MIN_TRAJECTORY_SAMPLES:
        message = f"basic_training_show segment {key} must have shape (num_points, 3) with at least two points"
        raise ValueError(message)
    if not np.all(np.isfinite(points)):
        message = f"basic_training_show segment {key} must contain only finite values"
        raise ValueError(message)
    return points


def _show_segment_bool(segment: Mapping[str, Any], key: str, default: bool) -> bool:
    """Read an optional segment boolean."""
    if key not in segment:
        return default
    value = segment[key]
    if not isinstance(value, bool):
        message = f"basic_training_show segment {key} must be a boolean"
        raise TypeError(message)
    return value


def _validate_declared_segment_start(segment: Mapping[str, Any], actual_start: np.ndarray) -> None:
    """Validate an optional declared segment start against generated geometry."""
    if validation.contracts.FIELD_SEGMENT_START not in segment:
        return
    declared = _show_segment_xyz(segment, validation.contracts.FIELD_SEGMENT_START)
    gap = float(np.linalg.norm(declared - np.asarray(actual_start, dtype=float)))
    if gap > BASIC_SHOW_CONTINUITY_TOLERANCE_M:
        message = f"basic_training_show declared segment_start differs from generated start by {gap:.6g} m"
        raise ValueError(message)


def _validate_declared_segment_end(segment: Mapping[str, Any], actual_end: np.ndarray) -> None:
    """Validate an optional declared segment end against generated geometry."""
    if validation.contracts.FIELD_SEGMENT_END not in segment:
        return
    declared = _show_segment_xyz(segment, validation.contracts.FIELD_SEGMENT_END)
    gap = float(np.linalg.norm(declared - np.asarray(actual_end, dtype=float)))
    if gap > BASIC_SHOW_CONTINUITY_TOLERANCE_M:
        message = f"basic_training_show declared segment_end differs from generated end by {gap:.6g} m"
        raise ValueError(message)


def _validate_basic_show_final_hold_segment(segment: Mapping[str, Any], previous_end: np.ndarray | None) -> None:
    """Validate final-hold metadata against the previous segment endpoint."""
    if previous_end is None:
        message = "basic_training_show final_hold segment requires a previous endpoint"
        raise ValueError(message)
    start = _show_segment_xyz(segment, validation.contracts.FIELD_SEGMENT_START)
    end = _show_segment_xyz(segment, validation.contracts.FIELD_SEGMENT_END)
    start_gap = float(np.linalg.norm(start - previous_end))
    hold_gap = float(np.linalg.norm(end - previous_end))
    if start_gap > BASIC_SHOW_CONTINUITY_TOLERANCE_M or hold_gap > BASIC_SHOW_CONTINUITY_TOLERANCE_M:
        message = "basic_training_show final_hold must hold the final segment endpoint"
        raise ValueError(message)


def _basic_show_adjusted_times(local_times: np.ndarray, current_time_end: float, is_first_segment: bool) -> np.ndarray:
    """Shift local segment times into one strictly increasing show timeline."""
    if is_first_segment:
        return np.array(local_times, dtype=float, copy=True)
    sample_interval = _sample_interval_sec(local_times)
    return np.array(float(current_time_end) + sample_interval + local_times, dtype=float, copy=True)


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


def _build_ellipse_trajectory(task: Mapping[str, Any]) -> tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]]:
    """Build an ellipse trajectory from a task mapping, returning messages on failure."""
    try:
        duration_sec = _require_float(task, validation.contracts.FIELD_DURATION_SEC)
        sample_rate_hz = _require_float(task, validation.contracts.FIELD_SAMPLE_RATE_HZ)
        radius_x = _require_float(task, validation.contracts.FIELD_RADIUS_X)
        radius_y = _require_float(task, validation.contracts.FIELD_RADIUS_Y)
        height = _require_float(task, validation.contracts.FIELD_HEIGHT)
        center = _optional_sequence(task, validation.contracts.FIELD_CENTER, default=(0.0, 0.0))
        clockwise = bool(task.get(validation.contracts.FIELD_CLOCKWISE, False))
        trajectory = trajectories.primitives.make_ellipse_trajectory(
            radius_x=radius_x,
            radius_y=radius_y,
            height=height,
            duration_sec=duration_sec,
            sample_rate_hz=sample_rate_hz,
            center=center,
            clockwise=clockwise,
        )
    except (TypeError, ValueError) as exc:
        return None, (str(exc),)
    return trajectory, ()


def _build_figure_eight_trajectory(task: Mapping[str, Any]) -> tuple[trajectories.primitives.Trajectory | None, tuple[str, ...]]:
    """Build a figure-eight trajectory from a task mapping, returning messages on failure."""
    try:
        duration_sec = _require_float(task, validation.contracts.FIELD_DURATION_SEC)
        sample_rate_hz = _require_float(task, validation.contracts.FIELD_SAMPLE_RATE_HZ)
        radius_x = _require_float(task, validation.contracts.FIELD_RADIUS_X)
        radius_y = _require_float(task, validation.contracts.FIELD_RADIUS_Y)
        height = _require_float(task, validation.contracts.FIELD_HEIGHT)
        center = _optional_sequence(task, validation.contracts.FIELD_CENTER, default=(0.0, 0.0))
        clockwise = bool(task.get(validation.contracts.FIELD_CLOCKWISE, False))
        trajectory = trajectories.primitives.make_figure_eight_trajectory(
            radius_x=radius_x,
            radius_y=radius_y,
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
    hold_steps = max(1, round(requested_sec / sample_interval))
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


def _apply_task_final_hold(
    task: Mapping[str, Any],
    trajectory: trajectories.primitives.Trajectory,
) -> tuple[trajectories.primitives.Trajectory, dict[str, Any]]:
    """Append a stationary final-hold segment when the task contract requests it."""
    enabled = _final_hold_enabled(task)
    requested_sec = _final_hold_seconds(task, enabled=enabled)
    exclude_from_tracking = _optional_bool(
        task,
        validation.contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS,
        default=enabled,
    )
    times = np.asarray(trajectory.times, dtype=float)
    if not enabled:
        return trajectory, _final_hold_metadata(
            enabled=False,
            final_hold_sec=0.0,
            exclude_from_tracking=False,
            tracking_phase_end_step=int(times.shape[0]),
            tracking_phase_end_time_sec=float(times[-1]),
        )
    if requested_sec <= 0.0:
        message = f"{validation.contracts.FIELD_FINAL_HOLD_SEC} must be positive when final hold is enabled"
        raise ValueError(message)

    positions = np.asarray(trajectory.positions, dtype=float)
    sample_interval = _sample_interval_sec(times)
    hold_steps = max(1, round(requested_sec / sample_interval))
    effective_hold_sec = float(hold_steps * sample_interval)
    hold_times = times[-1] + sample_interval * np.arange(1, hold_steps + 1, dtype=float)
    hold_positions = np.repeat(positions[-1].reshape(1, XYZ_DIMENSIONS), repeats=hold_steps, axis=0)
    held_trajectory = trajectories.primitives.Trajectory(
        times=np.concatenate((times, hold_times)),
        positions=np.vstack((positions, hold_positions)),
    )
    return held_trajectory, _final_hold_metadata(
        enabled=True,
        final_hold_sec=effective_hold_sec,
        exclude_from_tracking=exclude_from_tracking,
        tracking_phase_end_step=int(times.shape[0]),
        tracking_phase_end_time_sec=float(times[-1]),
    )


def _final_hold_metadata(
    enabled: bool,
    final_hold_sec: float,
    exclude_from_tracking: bool,
    tracking_phase_end_step: int,
    tracking_phase_end_time_sec: float,
) -> dict[str, Any]:
    """Return JSON-ready final-hold metadata for validation consumers."""
    return {
        "final_hold_enabled": bool(enabled),
        "final_hold_sec": float(final_hold_sec),
        "exclude_final_hold_from_tracking_metrics": bool(exclude_from_tracking),
        "tracking_phase_end_step": int(tracking_phase_end_step),
        "tracking_phase_end_time_sec": float(tracking_phase_end_time_sec),
    }


def _final_hold_enabled(task: Mapping[str, Any]) -> bool:
    """Return whether final-hold is enabled for a task."""
    shape = str(task.get(validation.contracts.FIELD_SHAPE, ""))
    default_enabled = shape in FINAL_HOLD_DEFAULT_SHAPES
    return _optional_bool(task, validation.contracts.FIELD_FINAL_HOLD_ENABLED, default=default_enabled)


def _final_hold_seconds(task: Mapping[str, Any], enabled: bool) -> float:
    """Return configured or default final-hold seconds for a task."""
    if not enabled:
        return 0.0
    return _optional_float(task, validation.contracts.FIELD_FINAL_HOLD_SEC, default=DEFAULT_FINAL_HOLD_SEC)


def _sample_interval_sec(times: np.ndarray) -> float:
    """Return a representative positive sample interval from built trajectory times."""
    diffs = np.diff(times)
    if diffs.size == 0 or np.any(diffs <= 0.0):
        message = "trajectory times must be strictly increasing before hold metadata can be applied"
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
