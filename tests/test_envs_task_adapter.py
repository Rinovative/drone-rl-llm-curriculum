"""Tests for environment task adapter reference packaging."""

# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pytest

from src import envs, validation

XYZ_DIMENSIONS = 3
BASIC_TRAINING_SHOW_SEGMENT_COUNT = 9


def _line_task() -> dict[str, object]:
    """Return a valid line task for adapter tests."""
    contracts = validation.contracts
    return {
        contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
        contracts.FIELD_SHAPE: contracts.SHAPE_LINE,
        contracts.FIELD_DURATION_SEC: 3.0,
        contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
        contracts.FIELD_START: [0.0, 0.0, 1.0],
        contracts.FIELD_END: [0.5, 0.0, 1.0],
    }


def test_valid_task_returns_copied_reference_arrays() -> None:
    """Verify a valid task returns copied time and position arrays."""
    reference = envs.task_adapter.make_task_reference(_line_task())

    assert reference.times.ndim == 1
    assert reference.positions.shape == (reference.times.shape[0], 3)
    assert reference.times.flags.owndata
    assert reference.positions.flags.owndata
    assert reference.validation_messages == ()


def test_returned_task_metadata_is_copied_from_input() -> None:
    """Verify later input mutation does not change returned task metadata."""
    task = _line_task()
    reference = envs.task_adapter.make_task_reference(task)

    task[validation.contracts.FIELD_SHAPE] = validation.contracts.SHAPE_HOVER

    assert reference.task[validation.contracts.FIELD_SHAPE] == validation.contracts.SHAPE_LINE
    assert reference.shape == validation.contracts.SHAPE_LINE


def test_mutating_returned_arrays_does_not_affect_separately_built_reference() -> None:
    """Verify returned arrays do not alias validation-owned trajectory data."""
    task = _line_task()
    first_reference = envs.task_adapter.make_task_reference(task)
    second_reference = envs.task_adapter.make_task_reference(task)

    first_reference.times[0] = 99.0
    first_reference.positions[0, 0] = 99.0

    assert second_reference.times[0] == 0.0
    assert second_reference.positions[0, 0] == 0.0


def test_line_task_reference_adds_default_start_and_final_holds() -> None:
    """Verify moving tracking tasks include start alignment and final settling phases."""
    reference = envs.task_adapter.make_task_reference(_line_task())

    expected_tracking_phase_start_step = 12
    expected_tracking_phase_end_step = 43
    expected_total_reference_steps = 53
    expected_start_position = np.array([[0.0, 0.0, 1.0]])
    expected_final_position = np.array([[0.5, 0.0, 1.0]])
    expected_start_hold_positions = np.repeat(
        expected_start_position,
        expected_tracking_phase_start_step,
        axis=0,
    )

    assert reference.start_hold_enabled is True
    assert reference.start_hold_sec == pytest.approx(1.2)
    assert reference.exclude_start_hold_from_tracking_metrics is True
    assert reference.tracking_phase_start_step == expected_tracking_phase_start_step
    assert reference.tracking_phase_start_time_sec == pytest.approx(1.2)
    assert reference.final_hold_enabled is True
    assert reference.final_hold_sec == pytest.approx(1.0)
    assert reference.exclude_final_hold_from_tracking_metrics is True
    assert reference.tracking_phase_end_step == expected_tracking_phase_end_step
    assert reference.tracking_phase_end_time_sec == pytest.approx(4.2)
    assert reference.times.shape[0] == expected_total_reference_steps
    np.testing.assert_allclose(
        reference.positions[: reference.tracking_phase_start_step],
        expected_start_hold_positions,
    )
    np.testing.assert_allclose(
        reference.positions[reference.tracking_phase_end_step :],
        np.repeat(expected_final_position, reference.positions.shape[0] - expected_tracking_phase_end_step, axis=0),
    )


def test_invalid_task_raises_value_error_with_validation_diagnostics() -> None:
    """Verify invalid tasks raise ValueError with validation messages."""
    task = _line_task()
    task[validation.contracts.FIELD_END] = [3.0, 0.0, 1.0]

    with pytest.raises(ValueError, match=r"arena|speed|jump"):
        envs.task_adapter.make_task_reference(task)


def test_shape_metadata_is_populated_from_validated_task() -> None:
    """Verify shape metadata reflects the validated task shape."""
    reference = envs.task_adapter.make_task_reference(_line_task())

    assert reference.shape == validation.contracts.SHAPE_LINE


def test_hover_task_returns_expected_reference_position_shape() -> None:
    """Verify hover tasks also produce reference arrays with XYZ positions."""
    contracts = validation.contracts
    task = {
        contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
        contracts.FIELD_SHAPE: contracts.SHAPE_HOVER,
        contracts.FIELD_DURATION_SEC: 2.0,
        contracts.FIELD_SAMPLE_RATE_HZ: 5.0,
        contracts.FIELD_POSITION: [0.0, 0.0, 1.0],
    }

    reference = envs.task_adapter.make_task_reference(task)

    assert reference.shape == validation.contracts.SHAPE_HOVER
    assert reference.positions.shape[1] == XYZ_DIMENSIONS
    np.testing.assert_allclose(reference.positions[:, 2], 1.0)


def test_task_adapter_imports_through_package_alias() -> None:
    """Verify task adapter helpers are exposed by the envs package."""
    assert envs.task_adapter.make_task_reference is not None


def test_basic_training_show_reference_uses_composed_segments_and_final_hold() -> None:
    """Verify the adapter packages sampled basic training shows with final-hold metadata."""
    settings = envs.task_distribution.load_task_distribution_settings("configs/tasks/task_distribution_basic_training_show.yaml")
    task = envs.task_distribution.sample_task(settings)

    reference = envs.task_adapter.make_task_reference(task, limits=settings.validation_limits)

    assert reference.shape == validation.contracts.SHAPE_BASIC_TRAINING_SHOW
    assert reference.start_hold_enabled is True
    assert reference.start_hold_sec > 0.0
    assert reference.exclude_start_hold_from_tracking_metrics is True
    np.testing.assert_allclose(reference.positions[0], reference.positions[reference.tracking_phase_start_step])
    assert reference.final_hold_enabled is True
    assert reference.final_hold_sec > 0.0
    assert reference.task["show_name"] == "basic_training_show"
    assert reference.task["segment_count"] == BASIC_TRAINING_SHOW_SEGMENT_COUNT
    np.testing.assert_allclose(reference.positions[reference.tracking_phase_end_step - 1], reference.positions[-1])
