"""Tests for deterministic trajectory evaluation metrics."""

# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pytest

from src import evaluation, trajectories

HAND_COMPUTABLE_SAMPLE_COUNT = 3


def _trajectory(times: list[float], positions: list[list[float]]) -> trajectories.primitives.Trajectory:
    """Build a trajectory object for metric tests."""
    return trajectories.primitives.Trajectory(
        times=np.asarray(times, dtype=float),
        positions=np.asarray(positions, dtype=float),
    )


def test_identical_trajectories_have_zero_position_error() -> None:
    """Verify identical sampled trajectories produce exactly zero error."""
    reference = trajectories.primitives.make_line_trajectory(
        start=(0.0, 0.0, 1.0),
        end=(1.0, 0.0, 1.0),
        duration_sec=1.0,
        sample_rate_hz=4.0,
    )

    errors = evaluation.trajectory_metrics.compute_position_errors(reference, reference)
    summary = evaluation.trajectory_metrics.summarize_tracking_error(reference, reference)

    np.testing.assert_allclose(errors, np.zeros(reference.times.shape[0]))
    assert summary.mean_position_error_m == 0.0
    assert summary.max_position_error_m == 0.0
    assert summary.rmse_position_error_m == 0.0
    assert summary.final_position_error_m == 0.0
    assert summary.duration_sec == 1.0
    assert summary.num_samples == reference.times.shape[0]


def test_tracking_summary_matches_hand_computable_errors() -> None:
    """Verify summary metrics on a small hand-computable trajectory pair."""
    reference = _trajectory(
        times=[0.0, 1.0, 2.0],
        positions=[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
    )
    actual = _trajectory(
        times=[0.0, 1.0, 2.0],
        positions=[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [4.0, 0.0, 0.0]],
    )

    errors = evaluation.trajectory_metrics.compute_position_errors(reference, actual)
    summary = evaluation.trajectory_metrics.summarize_tracking_error(reference, actual)

    np.testing.assert_allclose(errors, np.array([0.0, 1.0, 2.0]))
    assert summary.mean_position_error_m == pytest.approx(1.0)
    assert summary.max_position_error_m == pytest.approx(2.0)
    assert summary.rmse_position_error_m == pytest.approx(np.sqrt(5.0 / 3.0))
    assert summary.final_position_error_m == pytest.approx(2.0)
    assert summary.duration_sec == pytest.approx(2.0)
    assert summary.num_samples == HAND_COMPUTABLE_SAMPLE_COUNT


def test_mismatched_time_samples_raise_value_error() -> None:
    """Verify trajectories must share identical time samples."""
    reference = _trajectory([0.0, 1.0], [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    actual = _trajectory([0.0, 1.1], [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])

    with pytest.raises(ValueError, match="time samples"):
        evaluation.trajectory_metrics.compute_position_errors(reference, actual)


def test_mismatched_position_shapes_raise_value_error() -> None:
    """Verify trajectories must have matching position array shapes."""
    reference = _trajectory([0.0, 1.0], [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    actual = _trajectory(
        [0.0, 0.5, 1.0],
        [[0.0, 0.0, 1.0], [0.5, 0.0, 1.0], [1.0, 0.0, 1.0]],
    )

    with pytest.raises(ValueError, match="sample counts"):
        evaluation.trajectory_metrics.compute_position_errors(reference, actual)


def test_nonfinite_position_values_raise_value_error() -> None:
    """Verify nonfinite position values are rejected."""
    reference = _trajectory([0.0, 1.0], [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]])
    actual = _trajectory([0.0, 1.0], [[0.0, 0.0, 1.0], [float("nan"), 0.0, 1.0]])

    with pytest.raises(ValueError, match="positions must contain only finite values"):
        evaluation.trajectory_metrics.compute_position_errors(reference, actual)


def test_trajectory_metrics_imports_through_package_alias() -> None:
    """Verify trajectory metrics are exposed by the evaluation package."""
    assert evaluation.trajectory_metrics.compute_position_errors is not None
