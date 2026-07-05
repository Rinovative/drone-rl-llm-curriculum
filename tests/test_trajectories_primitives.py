"""Tests for deterministic trajectory primitive generation."""

# ruff: noqa: S101

from __future__ import annotations

import numpy as np

from src import trajectories


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
