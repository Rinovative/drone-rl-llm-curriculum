"""Tests for deterministic trajectory task validation."""

# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pytest

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


def test_valid_ellipse_task_passes_validation() -> None:
    """Verify a feasible ellipse task is accepted."""
    contracts = validation.contracts
    result = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_ELLIPSE,
            contracts.FIELD_DURATION_SEC: 12.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_RADIUS_X: 0.35,
            contracts.FIELD_RADIUS_Y: 0.20,
            contracts.FIELD_HEIGHT: 1.0,
            contracts.FIELD_CENTER: [0.0, 0.0],
        }
    )

    assert result.is_valid
    assert result.messages == ()
    assert result.trajectory is not None


def test_valid_figure_eight_task_passes_validation() -> None:
    """Verify a feasible figure-eight task is accepted."""
    contracts = validation.contracts
    result = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_FIGURE_EIGHT,
            contracts.FIELD_DURATION_SEC: 14.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_RADIUS_X: 0.30,
            contracts.FIELD_RADIUS_Y: 0.18,
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
            "shape": "spiral",
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


def test_valid_line_task_passes_validation() -> None:
    """Verify a feasible line task is accepted."""
    contracts = validation.contracts
    result = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_LINE,
            contracts.FIELD_DURATION_SEC: 3.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_START: [0.0, 0.0, 1.0],
            contracts.FIELD_END: [0.5, 0.5, 1.2],
        }
    )

    assert result.is_valid
    assert result.messages == ()
    assert result.trajectory is not None


def test_manual_curriculum_hover_shapes_pass_validation() -> None:
    """Verify hover-like manual curriculum tasks are accepted explicitly."""
    contracts = validation.contracts
    for shape, position in (
        (contracts.SHAPE_HOVER_STABILIZATION, [0.0, 0.0, 1.0]),
        (contracts.SHAPE_NEARBY_TARGET_HOVER, [0.15, 0.0, 1.0]),
    ):
        result = validation.tasks.validate_task(
            {
                contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
                contracts.FIELD_SHAPE: shape,
                contracts.FIELD_DURATION_SEC: 2.0,
                contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
                contracts.FIELD_POSITION: position,
            }
        )

        assert result.is_valid
        assert result.messages == ()
        assert result.trajectory is not None


def test_manual_curriculum_line_shapes_pass_validation() -> None:
    """Verify short line and start-hold line curriculum tasks are accepted."""
    contracts = validation.contracts
    short_line = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_SHORT_SLOW_LINE,
            contracts.FIELD_DURATION_SEC: 4.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_START: [0.0, 0.0, 1.0],
            contracts.FIELD_END: [0.5, 0.0, 1.0],
        }
    )
    held_line = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_START_HOLD_THEN_SHORT_LINE,
            contracts.FIELD_HOLD_DURATION_SEC: 1.0,
            contracts.FIELD_MOVE_DURATION_SEC: 3.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_START: [0.0, 0.0, 1.0],
            contracts.FIELD_END: [0.25, 0.0, 1.0],
        }
    )

    assert short_line.is_valid
    assert short_line.messages == ()
    assert short_line.trajectory is not None
    assert held_line.is_valid
    assert held_line.messages == ()
    assert held_line.trajectory is not None


def test_valid_vertical_task_passes_validation() -> None:
    """Verify a feasible vertical task is accepted."""
    contracts = validation.contracts
    result = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_VERTICAL,
            contracts.FIELD_DURATION_SEC: 3.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_XY: [0, 0],
            contracts.FIELD_START_HEIGHT: 0.8,
            contracts.FIELD_END_HEIGHT: 1.4,
        }
    )

    assert result.is_valid
    assert result.messages == ()
    assert result.trajectory is not None


def test_invalid_vertical_task_fails_validation() -> None:
    """Verify infeasible vertical tasks are rejected."""
    contracts = validation.contracts
    result = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_VERTICAL,
            contracts.FIELD_DURATION_SEC: 3.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_XY: [0.0, 0.0],
            contracts.FIELD_START_HEIGHT: 0.8,
            contracts.FIELD_END_HEIGHT: 3.0,
        }
    )

    assert not result.is_valid
    assert any("height" in message for message in result.messages)


def test_valid_polyline_task_passes_validation() -> None:
    """Verify a feasible polyline task is accepted."""
    contracts = validation.contracts
    result = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_POLYLINE,
            contracts.FIELD_DURATION_SEC: 6.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_POINTS: [[0.0, 0.0, 1.0], [0.5, 0.0, 1.1], [0.5, 0.5, 1.0]],
        }
    )

    assert result.is_valid
    assert result.messages == ()
    assert result.trajectory is not None


def test_invalid_polyline_task_fails_validation() -> None:
    """Verify invalid polyline task fields are rejected."""
    contracts = validation.contracts
    result = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_POLYLINE,
            contracts.FIELD_DURATION_SEC: 6.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_POINTS: [[0.0, 0.0, 1.0]],
        }
    )

    assert not result.is_valid
    assert any("points" in message for message in result.messages)


def test_basic_training_show_passes_validation_and_holds_final_point() -> None:
    """Verify composed basic training shows validate as continuous references."""
    contracts = validation.contracts
    task = {
        contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
        contracts.FIELD_SHAPE: contracts.SHAPE_BASIC_TRAINING_SHOW,
        contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
        contracts.FIELD_FINAL_HOLD_ENABLED: True,
        contracts.FIELD_FINAL_HOLD_SEC: 1.0,
        contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS: True,
        contracts.FIELD_SEGMENTS: [
            {
                contracts.FIELD_SEGMENT_SHAPE: "hover_stabilization",
                contracts.FIELD_SEGMENT_START: [0.0, 0.0, 1.0],
                contracts.FIELD_SEGMENT_END: [0.0, 0.0, 1.0],
                contracts.FIELD_SEGMENT_DURATION_SEC: 1.0,
            },
            {
                contracts.FIELD_SEGMENT_SHAPE: "vertical",
                contracts.FIELD_SEGMENT_START: [0.0, 0.0, 1.0],
                contracts.FIELD_SEGMENT_END: [0.0, 0.0, 1.2],
                contracts.FIELD_SEGMENT_DURATION_SEC: 1.5,
            },
            {
                contracts.FIELD_SEGMENT_SHAPE: "diagonal_line",
                contracts.FIELD_SEGMENT_START: [0.0, 0.0, 1.2],
                contracts.FIELD_SEGMENT_END: [0.3, 0.2, 1.2],
                contracts.FIELD_SEGMENT_DURATION_SEC: 2.0,
            },
            {
                contracts.FIELD_SEGMENT_SHAPE: "ellipse",
                contracts.FIELD_SEGMENT_START: [0.3, 0.2, 1.2],
                contracts.FIELD_SEGMENT_END: [0.3, 0.2, 1.2],
                contracts.FIELD_SEGMENT_DURATION_SEC: 3.0,
                "radius_x_m": 0.12,
                "radius_y_m": 0.08,
                "phase_deg": 180.0,
            },
            {
                contracts.FIELD_SEGMENT_SHAPE: "final_hold",
                contracts.FIELD_SEGMENT_START: [0.3, 0.2, 1.2],
                contracts.FIELD_SEGMENT_END: [0.3, 0.2, 1.2],
                contracts.FIELD_SEGMENT_DURATION_SEC: 1.0,
                contracts.FIELD_SEGMENT_FINAL_HOLD_SEC: 1.0,
            },
        ],
    }

    result = validation.tasks.validate_task(task)

    assert result.is_valid, result.messages
    assert result.trajectory is not None
    assert result.final_hold_enabled is True
    assert result.final_hold_sec == pytest.approx(1.0)
    np.testing.assert_allclose(result.trajectory.positions[result.tracking_phase_end_step - 1], result.trajectory.positions[-1])


def test_basic_training_show_rejects_discontinuous_segments() -> None:
    """Verify composed-show validation rejects hidden jumps between segments."""
    contracts = validation.contracts
    result = validation.tasks.validate_task(
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_BASIC_TRAINING_SHOW,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_SEGMENTS: [
                {
                    contracts.FIELD_SEGMENT_SHAPE: "hover_stabilization",
                    contracts.FIELD_SEGMENT_START: [0.0, 0.0, 1.0],
                    contracts.FIELD_SEGMENT_END: [0.0, 0.0, 1.0],
                    contracts.FIELD_SEGMENT_DURATION_SEC: 1.0,
                },
                {
                    contracts.FIELD_SEGMENT_SHAPE: "line",
                    contracts.FIELD_SEGMENT_START: [0.5, 0.0, 1.0],
                    contracts.FIELD_SEGMENT_END: [0.8, 0.0, 1.0],
                    contracts.FIELD_SEGMENT_DURATION_SEC: 2.0,
                },
            ],
        }
    )

    assert not result.is_valid
    assert any("discontinuous" in message for message in result.messages)
