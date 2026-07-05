"""Tests for deterministic trajectory task validation."""

# ruff: noqa: S101

from __future__ import annotations

from src import validation


def test_valid_hover_task_passes_validation() -> None:
    """Verify a feasible hover task is accepted."""
    contracts = validation.contracts
    result = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_HOVER,
            contracts.FIELD_DURATION_SEC: 2.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 5.0,
            contracts.FIELD_POSITION: [0.0, 0.0, 1.0],
        }
    )

    assert result.is_valid
    assert result.messages == ()
    assert result.trajectory is not None


def test_valid_circle_task_passes_validation() -> None:
    """Verify a feasible circle task is accepted."""
    contracts = validation.contracts
    result = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_CIRCLE,
            contracts.FIELD_DURATION_SEC: 8.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_RADIUS: 0.5,
            contracts.FIELD_HEIGHT: 1.0,
            contracts.FIELD_CENTER: [0.0, 0.0],
        }
    )

    assert result.is_valid
    assert result.messages == ()
    assert result.trajectory is not None


def test_unsupported_shape_fails_validation() -> None:
    """Verify unsupported task shapes are rejected."""
    result = validation.tasks.validate_task(
        {
            "task_type": "trajectory",
            "shape": "figure_eight",
            "duration_sec": 2.0,
            "sample_rate_hz": 5.0,
        }
    )

    assert not result.is_valid
    assert any("shape" in message for message in result.messages)


def test_out_of_bounds_hover_fails_validation() -> None:
    """Verify hover tasks outside the arena are rejected."""
    result = validation.tasks.validate_task(
        {
            "task_type": "trajectory",
            "shape": "hover",
            "duration_sec": 2.0,
            "sample_rate_hz": 5.0,
            "position": [3.0, 0.0, 1.0],
        }
    )

    assert not result.is_valid
    assert any("arena" in message for message in result.messages)


def test_negative_circle_radius_fails_validation() -> None:
    """Verify circle tasks with negative radii are rejected."""
    result = validation.tasks.validate_task(
        {
            "task_type": "trajectory",
            "shape": "circle",
            "duration_sec": 8.0,
            "sample_rate_hz": 10.0,
            "radius": -0.5,
            "height": 1.0,
            "center": [0.0, 0.0],
        }
    )

    assert not result.is_valid
    assert any("radius" in message for message in result.messages)


def test_too_fast_circle_fails_validation() -> None:
    """Verify circle tasks above the configured speed limit are rejected."""
    result = validation.tasks.validate_task(
        {
            "task_type": "trajectory",
            "shape": "circle",
            "duration_sec": 1.0,
            "sample_rate_hz": 20.0,
            "radius": 1.0,
            "height": 1.0,
            "center": [0.0, 0.0],
        }
    )

    assert not result.is_valid
    assert any("speed" in message for message in result.messages)
