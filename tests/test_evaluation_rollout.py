"""Tests for deterministic MVP rollout evaluation helpers."""

# ruff: noqa: S101

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src import envs, evaluation, validation

if TYPE_CHECKING:
    from pathlib import Path


def _hover_task() -> dict[str, object]:
    """Return a valid hover task for rollout tests."""
    contracts = validation.contracts
    return {
        contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
        contracts.FIELD_SHAPE: contracts.SHAPE_HOVER,
        contracts.FIELD_DURATION_SEC: 2.0,
        contracts.FIELD_SAMPLE_RATE_HZ: 5.0,
        contracts.FIELD_POSITION: [0.0, 0.0, 1.0],
    }


def _line_task() -> dict[str, object]:
    """Return a valid line task for rollout tests."""
    contracts = validation.contracts
    return {
        contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
        contracts.FIELD_SHAPE: contracts.SHAPE_LINE,
        contracts.FIELD_DURATION_SEC: 3.0,
        contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
        contracts.FIELD_START: [0.0, 0.0, 1.0],
        contracts.FIELD_END: [0.5, 0.0, 1.0],
    }


def test_zero_offset_rollout_has_zero_tracking_error() -> None:
    """Verify a zero-offset deterministic rollout exactly matches reference."""
    result = evaluation.rollout.evaluate_task_rollout(_hover_task(), offset=(0.0, 0.0, 0.0))

    assert result.metrics["task_shape"] == validation.contracts.SHAPE_HOVER
    assert result.metrics["sample_count"] == result.reference.times.shape[0]
    assert result.metrics["mean_position_error_m"] == pytest.approx(0.0)
    assert result.metrics["max_position_error_m"] == pytest.approx(0.0)
    assert result.metrics["rmse_position_error_m"] == pytest.approx(0.0)
    assert result.metrics["final_position_error_m"] == pytest.approx(0.0)


def test_nonzero_offset_rollout_has_positive_error() -> None:
    """Verify nonzero deterministic offset produces nonzero tracking error."""
    result = evaluation.rollout.evaluate_task_rollout(_line_task(), offset=(0.1, 0.0, 0.0))

    assert result.metrics["mean_position_error_m"] == pytest.approx(0.1)
    assert result.metrics["max_position_error_m"] == pytest.approx(0.1)
    assert result.metrics["rmse_position_error_m"] == pytest.approx(0.1)
    assert result.metrics["mean_reward"] < 0.0


def test_rollout_accepts_existing_task_reference() -> None:
    """Verify callers can pass an already validated task reference."""
    reference = envs.task_adapter.make_task_reference(_line_task())

    result = evaluation.rollout.evaluate_task_rollout(reference, offset=(0.0, 0.0, 0.0))

    assert result.metrics["task_shape"] == validation.contracts.SHAPE_LINE
    assert result.metrics["sample_count"] == reference.positions.shape[0]


def test_rollout_metrics_write_json_to_temporary_directory(tmp_path: Path) -> None:
    """Verify rollout metrics are written as small JSON artifacts."""
    output_path = tmp_path / "rollout_metrics.json"

    result = evaluation.rollout.write_task_rollout_evaluation(_hover_task(), output_path, offset=(0.0, 0.0, 0.0))
    payload = json.loads(output_path.read_text(encoding="utf-8"))

    assert result.output_path == str(output_path)
    assert payload["sample_count"] == result.metrics["sample_count"]
    assert payload["duration_sec"] == result.metrics["duration_sec"]


def test_invalid_task_rejected_without_writing_output(tmp_path: Path) -> None:
    """Verify invalid tasks fail before output artifacts are created."""
    task = _hover_task()
    task[validation.contracts.FIELD_POSITION] = [10.0, 0.0, 1.0]
    output_path = tmp_path / "should_not_exist.json"

    with pytest.raises(ValueError, match="invalid trajectory task"):
        evaluation.rollout.write_task_rollout_evaluation(task, output_path)

    assert not output_path.exists()


def test_invalid_offset_and_lag_raise_value_error() -> None:
    """Verify rollout inputs are validated before metrics are computed."""
    with pytest.raises(ValueError, match="offset"):
        evaluation.rollout.evaluate_task_rollout(_hover_task(), offset=(0.0, 0.0))
    with pytest.raises(ValueError, match="lag_steps"):
        evaluation.rollout.evaluate_task_rollout(_hover_task(), lag_steps=-1)


def test_rollout_imports_through_package_alias() -> None:
    """Verify rollout helpers are exposed by the evaluation package."""
    assert evaluation.rollout.evaluate_task_rollout is not None
