"""Tests for environment task adapter reference packaging."""

# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pytest

from src import envs, validation

XYZ_DIMENSIONS = 3


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
