"""Tests for deterministic trajectory primitive generation."""

# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pytest

from src import trajectories

FIGURE_EIGHT_RADIUS_Y = 0.2


def test_hover_trajectory_keeps_constant_position() -> None:
    """Verify hover trajectories repeat the requested XYZ position."""
    trajectory = trajectories.primitives.make_hover_trajectory(
        position=(0.25, -0.5, 1.0),
        duration_sec=2.0,
        sample_rate_hz=4.0,
    )

    assert trajectory.times.shape == (9,)
    assert trajectory.positions.shape == (9, 3)
    np.testing.assert_allclose(trajectory.times[[0, -1]], np.array([0.0, 2.0]))
    np.testing.assert_allclose(trajectory.positions, np.tile(np.array([0.25, -0.5, 1.0]), (9, 1)))


def test_circle_trajectory_uses_radius_height_and_center() -> None:
    """Verify circle trajectories stay at the requested radius and height."""
    trajectory = trajectories.primitives.make_circle_trajectory(
        radius=0.75,
        height=1.2,
        duration_sec=4.0,
        sample_rate_hz=8.0,
        center=(0.5, -0.25),
    )

    offsets = trajectory.positions[:, :2] - np.array([0.5, -0.25])
    radii = np.linalg.norm(offsets, axis=1)

    assert trajectory.times.shape == (33,)
    assert trajectory.positions.shape == (33, 3)
    np.testing.assert_allclose(radii, 0.75, atol=1e-12)
    np.testing.assert_allclose(trajectory.positions[:, 2], 1.2)


def test_ellipse_trajectory_uses_axis_radii_height_and_center() -> None:
    """Verify ellipse trajectories respect both horizontal semi-axes."""
    trajectory = trajectories.primitives.make_ellipse_trajectory(
        radius_x=0.6,
        radius_y=0.25,
        height=1.1,
        duration_sec=6.0,
        sample_rate_hz=10.0,
        center=(0.1, -0.2),
    )

    offsets = trajectory.positions[:, :2] - np.array([0.1, -0.2])

    assert trajectory.times.shape == (61,)
    assert trajectory.positions.shape == (61, 3)
    assert np.max(np.abs(offsets[:, 0])) == pytest.approx(0.6)
    assert np.max(np.abs(offsets[:, 1])) == pytest.approx(0.25, abs=1e-2)
    np.testing.assert_allclose(trajectory.positions[:, 2], 1.1)


def test_figure_eight_trajectory_crosses_center_and_stays_bounded() -> None:
    """Verify figure-eight trajectories are bounded around the configured center."""
    trajectory = trajectories.primitives.make_figure_eight_trajectory(
        radius_x=0.4,
        radius_y=FIGURE_EIGHT_RADIUS_Y,
        height=1.0,
        duration_sec=8.0,
        sample_rate_hz=10.0,
        center=(0.1, -0.2),
    )

    offsets = trajectory.positions[:, :2] - np.array([0.1, -0.2])

    assert trajectory.times.shape == (81,)
    assert trajectory.positions.shape == (81, 3)
    assert np.max(np.abs(offsets[:, 0])) == pytest.approx(0.4)
    assert np.max(np.abs(offsets[:, 1])) <= FIGURE_EIGHT_RADIUS_Y
    np.testing.assert_allclose(trajectory.positions[0], np.array([0.1, -0.2, 1.0]))
    np.testing.assert_allclose(trajectory.positions[-1], np.array([0.1, -0.2, 1.0]), atol=1e-12)


def test_line_trajectory_has_correct_shape_and_points() -> None:
    """Verify line trajectories have the correct shape and start/end points."""
    trajectory = trajectories.primitives.make_line_trajectory(
        start=(0.0, 0.0, 0.0),
        end=(1.0, 1.0, 1.0),
        duration_sec=2.0,
        sample_rate_hz=4.0,
    )

    assert trajectory.times.shape == (9,)
    assert trajectory.positions.shape == (9, 3)
    np.testing.assert_allclose(trajectory.positions[0], np.array([0.0, 0.0, 0.0]))
    np.testing.assert_allclose(trajectory.positions[-1], np.array([1.0, 1.0, 1.0]))
    assert np.all(np.isfinite(trajectory.positions))


def test_line_trajectory_invalid_start() -> None:
    """Verify line trajectories raise ValueError for invalid start positions."""
    with pytest.raises(ValueError, match="start must contain exactly 3 values"):
        trajectories.primitives.make_line_trajectory(
            start=(0.0, 0.0),
            end=(1.0, 1.0, 1.0),
            duration_sec=2.0,
            sample_rate_hz=4.0,
        )


def test_line_trajectory_invalid_end() -> None:
    """Verify line trajectories raise ValueError for invalid end positions."""
    with pytest.raises(ValueError, match="end must contain exactly 3 values"):
        trajectories.primitives.make_line_trajectory(
            start=(0.0, 0.0, 0.0),
            end=(1.0, 1.0),
            duration_sec=2.0,
            sample_rate_hz=4.0,
        )


def test_vertical_trajectory_has_correct_shape_and_points() -> None:
    """Verify vertical trajectories have the correct shape and start/end points."""
    trajectory = trajectories.primitives.make_vertical_trajectory(
        xy=(0.25, -0.5),
        start_height=0.8,
        end_height=1.4,
        duration_sec=2.0,
        sample_rate_hz=5.0,
    )

    assert trajectory.times.shape == (11,)
    assert trajectory.positions.shape == (11, 3)
    np.testing.assert_allclose(trajectory.positions[0], np.array([0.25, -0.5, 0.8]))
    np.testing.assert_allclose(trajectory.positions[-1], np.array([0.25, -0.5, 1.4]))
    assert np.all(np.isfinite(trajectory.positions))


def test_vertical_trajectory_invalid_xy() -> None:
    """Verify vertical trajectories raise ValueError for invalid XY positions."""
    with pytest.raises(ValueError, match="xy must contain exactly 2 values"):
        trajectories.primitives.make_vertical_trajectory(
            xy=(0.25,),
            start_height=0.8,
            end_height=1.4,
            duration_sec=2.0,
            sample_rate_hz=5.0,
        )


def test_vertical_trajectory_invalid_start_height() -> None:
    """Verify vertical trajectories raise ValueError for invalid start heights."""
    with pytest.raises(ValueError, match="start_height must be finite"):
        trajectories.primitives.make_vertical_trajectory(
            xy=(0.25, -0.5),
            start_height=float("inf"),
            end_height=1.4,
            duration_sec=2.0,
            sample_rate_hz=5.0,
        )


def test_vertical_trajectory_invalid_end_height() -> None:
    """Verify vertical trajectories raise ValueError for invalid end heights."""
    with pytest.raises(ValueError, match="end_height must be finite"):
        trajectories.primitives.make_vertical_trajectory(
            xy=(0.25, -0.5),
            start_height=0.8,
            end_height=float("nan"),
            duration_sec=2.0,
            sample_rate_hz=5.0,
        )


def test_polyline_trajectory_interpolates_by_path_length() -> None:
    """Verify polyline trajectories sample waypoints by cumulative path length."""
    trajectory = trajectories.primitives.make_polyline_trajectory(
        points=((0.0, 0.0, 1.0), (2.0, 0.0, 1.0), (2.0, 2.0, 1.0)),
        duration_sec=4.0,
        sample_rate_hz=1.0,
    )

    expected_positions = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [2.0, 0.0, 1.0],
            [2.0, 1.0, 1.0],
            [2.0, 2.0, 1.0],
        ]
    )
    assert trajectory.times.shape == (5,)
    assert trajectory.positions.shape == (5, 3)
    np.testing.assert_allclose(trajectory.positions, expected_positions)


def test_polyline_trajectory_rejects_too_few_points() -> None:
    """Verify polyline trajectories require at least two waypoints."""
    with pytest.raises(ValueError, match="points must contain at least two waypoints"):
        trajectories.primitives.make_polyline_trajectory(
            points=((0.0, 0.0, 1.0),),
            duration_sec=2.0,
            sample_rate_hz=4.0,
        )


def test_polyline_trajectory_rejects_wrong_shape() -> None:
    """Verify polyline trajectories require XYZ waypoints."""
    with pytest.raises(ValueError, match="points must have shape"):
        trajectories.primitives.make_polyline_trajectory(
            points=((0.0, 0.0), (1.0, 1.0)),
            duration_sec=2.0,
            sample_rate_hz=4.0,
        )


def test_polyline_trajectory_rejects_nonfinite_points() -> None:
    """Verify polyline trajectories require finite waypoint values."""
    with pytest.raises(ValueError, match="points must contain only finite values"):
        trajectories.primitives.make_polyline_trajectory(
            points=((0.0, 0.0, 1.0), (1.0, float("nan"), 1.0)),
            duration_sec=2.0,
            sample_rate_hz=4.0,
        )


def test_polyline_trajectory_rejects_zero_length_path() -> None:
    """Verify polyline trajectories reject paths without movement."""
    with pytest.raises(ValueError, match="points must define a nonzero path length"):
        trajectories.primitives.make_polyline_trajectory(
            points=((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)),
            duration_sec=2.0,
            sample_rate_hz=4.0,
        )
