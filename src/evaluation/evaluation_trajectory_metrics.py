"""
===============================================================================
evaluation_trajectory_metrics.py
===============================================================================
Compute deterministic metrics for sampled trajectory tracking errors.

Responsibilities:
  - Validate comparable sampled reference and actual trajectory arrays
  - Compute per-sample Euclidean XYZ position errors
  - Summarize tracking errors with lightweight scalar metrics

Design principles:
  - Keep metrics pure, deterministic, and independent of simulator state
  - Reject mismatched samples instead of silently interpolating trajectories

Boundaries:
  - Plotting, file I/O, rollout collection, and training belong elsewhere
  - Environment construction and task validation are outside this module
===============================================================================

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src import trajectories

MIN_TRAJECTORY_SAMPLES = 2
POSITION_ARRAY_NDIM = 2
XYZ_DIMENSIONS = 3
TIME_MATCH_ATOL = 1e-12


@dataclass(frozen=True)
class TrajectoryMetricSummary:
    """
    Summary statistics for sampled trajectory tracking error.

    Parameters
    ----------
    mean_position_error_m
        Mean Euclidean XYZ position error in meters.
    max_position_error_m
        Maximum Euclidean XYZ position error in meters.
    rmse_position_error_m
        Root mean squared Euclidean XYZ position error in meters.
    final_position_error_m
        Final-sample Euclidean XYZ position error in meters.
    duration_sec
        Reference trajectory duration in seconds.
    num_samples
        Number of compared trajectory samples.

    """

    mean_position_error_m: float
    max_position_error_m: float
    rmse_position_error_m: float
    final_position_error_m: float
    duration_sec: float
    num_samples: int


def compute_position_errors(
    reference: trajectories.primitives.Trajectory,
    actual: trajectories.primitives.Trajectory,
) -> np.ndarray:
    """
    Compute per-sample Euclidean XYZ position errors between two trajectories.

    Parameters
    ----------
    reference
        Sampled reference trajectory.
    actual
        Sampled actual or executed trajectory sampled at the same times.

    Returns
    -------
    np.ndarray
        One-dimensional array of Euclidean position errors in meters.

    Raises
    ------
    ValueError
        If trajectories have incompatible sample counts, time samples, position
        shapes, nonfinite values, or fewer than two samples.

    """
    reference_times, reference_positions = _validated_trajectory_arrays(reference, name="reference")
    actual_times, actual_positions = _validated_trajectory_arrays(actual, name="actual")
    _validate_compatible_trajectories(
        reference_times=reference_times,
        reference_positions=reference_positions,
        actual_times=actual_times,
        actual_positions=actual_positions,
    )
    return np.linalg.norm(actual_positions - reference_positions, axis=1)


def summarize_tracking_error(
    reference: trajectories.primitives.Trajectory,
    actual: trajectories.primitives.Trajectory,
) -> TrajectoryMetricSummary:
    """
    Summarize sampled trajectory tracking errors as scalar metrics.

    Parameters
    ----------
    reference
        Sampled reference trajectory.
    actual
        Sampled actual or executed trajectory sampled at the same times.

    Returns
    -------
    TrajectoryMetricSummary
        Mean, maximum, RMSE, final error, duration, and sample count.

    Raises
    ------
    ValueError
        If the trajectories cannot be compared sample by sample.

    """
    errors = compute_position_errors(reference=reference, actual=actual)
    reference_times = np.asarray(reference.times, dtype=float)
    return TrajectoryMetricSummary(
        mean_position_error_m=float(np.mean(errors)),
        max_position_error_m=float(np.max(errors)),
        rmse_position_error_m=float(np.sqrt(np.mean(np.square(errors)))),
        final_position_error_m=float(errors[-1]),
        duration_sec=float(reference_times[-1] - reference_times[0]),
        num_samples=int(errors.shape[0]),
    )


def _validated_trajectory_arrays(
    trajectory: trajectories.primitives.Trajectory,
    name: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return validated time and position arrays for a sampled trajectory."""
    times = np.asarray(trajectory.times, dtype=float)
    positions = np.asarray(trajectory.positions, dtype=float)

    if times.ndim != 1:
        message = f"{name} times must be a one-dimensional array"
        raise ValueError(message)
    if times.shape[0] < MIN_TRAJECTORY_SAMPLES:
        message = f"{name} trajectory must contain at least two samples"
        raise ValueError(message)
    if positions.ndim != POSITION_ARRAY_NDIM or positions.shape[1:] != (XYZ_DIMENSIONS,):
        message = f"{name} positions must have shape (num_samples, 3)"
        raise ValueError(message)
    if positions.shape[0] != times.shape[0]:
        message = f"{name} times and positions must contain the same number of samples"
        raise ValueError(message)
    if not np.all(np.isfinite(times)):
        message = f"{name} times must contain only finite values"
        raise ValueError(message)
    if not np.all(np.isfinite(positions)):
        message = f"{name} positions must contain only finite values"
        raise ValueError(message)
    return times, positions


def _validate_compatible_trajectories(
    reference_times: np.ndarray,
    reference_positions: np.ndarray,
    actual_times: np.ndarray,
    actual_positions: np.ndarray,
) -> None:
    """Raise ValueError if two validated trajectories cannot be compared."""
    if reference_times.shape[0] != actual_times.shape[0]:
        message = "reference and actual trajectories must have matching sample counts"
        raise ValueError(message)
    if reference_positions.shape != actual_positions.shape:
        message = "reference and actual positions must have matching shape"
        raise ValueError(message)
    if not np.allclose(reference_times, actual_times, rtol=0.0, atol=TIME_MATCH_ATOL):
        message = "reference and actual trajectories must have matching time samples"
        raise ValueError(message)


__all__ = [
    "TrajectoryMetricSummary",
    "compute_position_errors",
    "summarize_tracking_error",
]
