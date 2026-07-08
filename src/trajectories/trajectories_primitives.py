"""
===============================================================================
trajectories_primitives.py
===============================================================================
Generate basic deterministic reference trajectories for drone tracking tasks.

Responsibilities:
  - Represent sampled reference trajectories as time and position arrays
  - Generate hover trajectories at a fixed XYZ position
  - Generate horizontal circle and ellipse trajectories at a fixed height
  - Generate smooth figure-eight trajectories at a fixed height
  - Generate line trajectories between two XYZ positions
  - Generate vertical and polyline trajectories for foundation validation

Design principles:
  - Keep trajectory generation pure and deterministic
  - Validate primitive inputs before returning sampled paths

Boundaries:
  - Feasibility checks across task constraints belong in validation modules
  - Drone simulation and controller logic belong in environment modules
===============================================================================

"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

MIN_SAMPLE_COUNT = 2
POINT_ARRAY_NDIM = 2
XYZ_DIMENSIONS = 3


@dataclass(frozen=True)
class Trajectory:
    """
    Sampled reference path for a single drone.

    Parameters
    ----------
    times
        One-dimensional array of sample times in seconds.
    positions
        Two-dimensional array of XYZ positions with shape ``(num_samples, 3)``.

    """

    times: np.ndarray
    positions: np.ndarray


def make_hover_trajectory(position: Sequence[float], duration_sec: float, sample_rate_hz: float) -> Trajectory:
    """
    Generate a fixed-position hover trajectory.

    Parameters
    ----------
    position
        XYZ hover position in meters.
    duration_sec
        Total trajectory duration in seconds.
    sample_rate_hz
        Number of trajectory samples per second.

    Returns
    -------
    Trajectory
        Sampled hover trajectory.

    Raises
    ------
    ValueError
        If the position, duration, or sample rate is invalid.

    """
    xyz = _as_float_array(position, expected_size=3, name="position")
    times = _make_times(duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)
    positions = np.repeat(xyz[np.newaxis, :], repeats=times.shape[0], axis=0)
    return Trajectory(times=times, positions=positions)


def make_circle_trajectory(
    radius: float,
    height: float,
    duration_sec: float,
    sample_rate_hz: float,
    center: Sequence[float] = (0.0, 0.0),
    clockwise: bool = False,
    phase_rad: float = 0.0,
) -> Trajectory:
    """
    Generate a horizontal circular trajectory at fixed height.

    Parameters
    ----------
    radius
        Circle radius in meters.
    height
        Fixed Z height in meters.
    duration_sec
        Total trajectory duration in seconds.
    sample_rate_hz
        Number of trajectory samples per second.
    center
        XY center of the circle in meters.
    clockwise
        Whether samples should move clockwise when viewed from above.
    phase_rad
        Initial angular phase in radians.

    Returns
    -------
    Trajectory
        Sampled circle trajectory.

    Raises
    ------
    ValueError
        If any trajectory parameter is invalid.

    """
    _ensure_positive(radius, name="radius")
    _ensure_finite(height, name="height")
    _ensure_finite(phase_rad, name="phase_rad")
    center_array = _as_float_array(center, expected_size=2, name="center")
    times = _make_times(duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)

    direction = -1.0 if clockwise else 1.0
    angles = phase_rad + direction * 2.0 * pi * times / duration_sec
    positions = np.column_stack(
        (
            center_array[0] + radius * np.cos(angles),
            center_array[1] + radius * np.sin(angles),
            np.full(times.shape, height, dtype=float),
        )
    )
    return Trajectory(times=times, positions=positions)


def make_ellipse_trajectory(
    radius_x: float,
    radius_y: float,
    height: float,
    duration_sec: float,
    sample_rate_hz: float,
    center: Sequence[float] = (0.0, 0.0),
    clockwise: bool = False,
    phase_rad: float = 0.0,
) -> Trajectory:
    """
    Generate a horizontal elliptical trajectory at fixed height.

    Parameters
    ----------
    radius_x
        Ellipse semi-axis in the X direction in meters.
    radius_y
        Ellipse semi-axis in the Y direction in meters.
    height
        Fixed Z height in meters.
    duration_sec
        Total trajectory duration in seconds.
    sample_rate_hz
        Number of trajectory samples per second.
    center
        XY center of the ellipse in meters.
    clockwise
        Whether samples should move clockwise when viewed from above.
    phase_rad
        Initial angular phase in radians.

    Returns
    -------
    Trajectory
        Sampled ellipse trajectory.

    Raises
    ------
    ValueError
        If any trajectory parameter is invalid.

    """
    _ensure_positive(radius_x, name="radius_x")
    _ensure_positive(radius_y, name="radius_y")
    _ensure_finite(height, name="height")
    _ensure_finite(phase_rad, name="phase_rad")
    center_array = _as_float_array(center, expected_size=2, name="center")
    times = _make_times(duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)

    direction = -1.0 if clockwise else 1.0
    angles = phase_rad + direction * 2.0 * pi * times / duration_sec
    positions = np.column_stack(
        (
            center_array[0] + radius_x * np.cos(angles),
            center_array[1] + radius_y * np.sin(angles),
            np.full(times.shape, height, dtype=float),
        )
    )
    return Trajectory(times=times, positions=positions)


def make_figure_eight_trajectory(
    radius_x: float,
    radius_y: float,
    height: float,
    duration_sec: float,
    sample_rate_hz: float,
    center: Sequence[float] = (0.0, 0.0),
    clockwise: bool = False,
    phase_rad: float = 0.0,
) -> Trajectory:
    """
    Generate a smooth horizontal figure-eight trajectory at fixed height.

    Parameters
    ----------
    radius_x
        Horizontal scale of the figure-eight in meters.
    radius_y
        Vertical scale of the figure-eight in meters.
    height
        Fixed Z height in meters.
    duration_sec
        Total trajectory duration in seconds.
    sample_rate_hz
        Number of trajectory samples per second.
    center
        XY crossing point of the figure-eight in meters.
    clockwise
        Whether to reverse traversal direction.
    phase_rad
        Initial phase in radians.

    Returns
    -------
    Trajectory
        Sampled figure-eight trajectory.

    Raises
    ------
    ValueError
        If any trajectory parameter is invalid.

    Notes
    -----
    The path uses a Gerono lemniscate, ``x = a sin(t)`` and
    ``y = b sin(t) cos(t)``, which is smooth and bounded.

    """
    _ensure_positive(radius_x, name="radius_x")
    _ensure_positive(radius_y, name="radius_y")
    _ensure_finite(height, name="height")
    _ensure_finite(phase_rad, name="phase_rad")
    center_array = _as_float_array(center, expected_size=2, name="center")
    times = _make_times(duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)

    direction = -1.0 if clockwise else 1.0
    angles = phase_rad + direction * 2.0 * pi * times / duration_sec
    sin_angles = np.sin(angles)
    positions = np.column_stack(
        (
            center_array[0] + radius_x * sin_angles,
            center_array[1] + radius_y * sin_angles * np.cos(angles),
            np.full(times.shape, height, dtype=float),
        )
    )
    return Trajectory(times=times, positions=positions)


def make_line_trajectory(
    start: Sequence[float],
    end: Sequence[float],
    duration_sec: float,
    sample_rate_hz: float,
) -> Trajectory:
    """
    Generate a straight line trajectory between two XYZ positions.

    Parameters
    ----------
    start
        XYZ start position in meters.
    end
        XYZ end position in meters.
    duration_sec
        Total trajectory duration in seconds.
    sample_rate_hz
        Number of trajectory samples per second.

    Returns
    -------
    Trajectory
        Sampled line trajectory.

    Raises
    ------
    ValueError
        If any trajectory parameter is invalid.

    """
    start_array = _as_float_array(start, expected_size=3, name="start")
    end_array = _as_float_array(end, expected_size=3, name="end")
    times = _make_times(duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)
    positions = np.linspace(start_array, end_array, num=times.shape[0], axis=0)

    # Ensure the first and last positions are exactly start and end
    positions[0] = start_array
    positions[-1] = end_array

    return Trajectory(times=times, positions=positions)


def make_vertical_trajectory(
    xy: Sequence[float],
    start_height: float,
    end_height: float,
    duration_sec: float,
    sample_rate_hz: float,
) -> Trajectory:
    """
    Generate a vertical trajectory with fixed XY and interpolated height.

    Parameters
    ----------
    xy
        XY position in meters.
    start_height
        Starting Z height in meters.
    end_height
        Ending Z height in meters.
    duration_sec
        Total trajectory duration in seconds.
    sample_rate_hz
        Number of trajectory samples per second.

    Returns
    -------
    Trajectory
        Sampled vertical trajectory.

    Raises
    ------
    ValueError
        If any trajectory parameter is invalid.

    """
    xy_array = _as_float_array(xy, expected_size=2, name="xy")
    _ensure_finite(start_height, name="start_height")
    _ensure_finite(end_height, name="end_height")
    times = _make_times(duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)
    positions = np.column_stack(
        (
            np.full(times.shape, xy_array[0], dtype=float),
            np.full(times.shape, xy_array[1], dtype=float),
            np.linspace(start_height, end_height, num=times.shape[0], dtype=float),
        )
    )
    return Trajectory(times=times, positions=positions)


def make_polyline_trajectory(
    points: Sequence[Sequence[float]],
    duration_sec: float,
    sample_rate_hz: float,
) -> Trajectory:
    """
    Generate a trajectory that follows XYZ waypoints at constant path-distance spacing.

    Parameters
    ----------
    points
        Sequence of XYZ waypoints in meters with shape ``(num_points, 3)``.
    duration_sec
        Total trajectory duration in seconds.
    sample_rate_hz
        Number of trajectory samples per second.

    Returns
    -------
    Trajectory
        Sampled polyline trajectory.

    Raises
    ------
    ValueError
        If the waypoint array, duration, or sample rate is invalid.

    """
    points_array = _as_points_array(points)
    times = _make_times(duration_sec=duration_sec, sample_rate_hz=sample_rate_hz)

    segment_lengths = np.linalg.norm(np.diff(points_array, axis=0), axis=1)
    total_length = float(np.sum(segment_lengths))
    if total_length <= 0.0:
        message = "points must define a nonzero path length"
        raise ValueError(message)

    cumulative_distances = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    nonzero_waypoint_mask = np.concatenate(([True], segment_lengths > 0.0))
    interpolation_distances = cumulative_distances[nonzero_waypoint_mask]
    interpolation_points = points_array[nonzero_waypoint_mask]
    sample_distances = np.linspace(0.0, total_length, num=times.shape[0], dtype=float)
    positions = np.column_stack(
        [np.interp(sample_distances, interpolation_distances, interpolation_points[:, axis]) for axis in range(XYZ_DIMENSIONS)]
    )
    positions[0] = points_array[0]
    positions[-1] = points_array[-1]
    return Trajectory(times=times, positions=positions)


def _make_times(duration_sec: float, sample_rate_hz: float) -> np.ndarray:
    """Create inclusive sample times for a positive duration and sample rate."""
    _ensure_positive(duration_sec, name="duration_sec")
    _ensure_positive(sample_rate_hz, name="sample_rate_hz")
    sample_count = max(MIN_SAMPLE_COUNT, round(duration_sec * sample_rate_hz) + 1)
    return np.linspace(0.0, duration_sec, num=sample_count, dtype=float)


def _as_float_array(values: Sequence[float], expected_size: int, name: str) -> np.ndarray:
    """Convert a numeric sequence into a finite one-dimensional float array."""
    array = np.asarray(values, dtype=float)
    if array.shape != (expected_size,):
        message = f"{name} must contain exactly {expected_size} values"
        raise ValueError(message)
    if not np.all(np.isfinite(array)):
        message = f"{name} must contain only finite values"
        raise ValueError(message)
    return array


def _as_points_array(points: Sequence[Sequence[float]]) -> np.ndarray:
    """Convert waypoint data into a finite two-dimensional XYZ float array."""
    array = np.asarray(points, dtype=float)
    if array.ndim != POINT_ARRAY_NDIM or array.shape[1:] != (XYZ_DIMENSIONS,):
        message = "points must have shape (num_points, 3)"
        raise ValueError(message)
    if array.shape[0] < MIN_SAMPLE_COUNT:
        message = "points must contain at least two waypoints"
        raise ValueError(message)
    if not np.all(np.isfinite(array)):
        message = "points must contain only finite values"
        raise ValueError(message)
    return array


def _ensure_positive(value: float, name: str) -> None:
    """Raise ValueError when a scalar value is not finite and strictly positive."""
    _ensure_finite(value, name=name)
    if value <= 0.0:
        message = f"{name} must be positive"
        raise ValueError(message)


def _ensure_finite(value: float, name: str) -> None:
    """Raise ValueError when a scalar value is not finite."""
    if not np.isfinite(value):
        message = f"{name} must be finite"
        raise ValueError(message)
