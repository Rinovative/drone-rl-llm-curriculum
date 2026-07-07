"""Tests for experiment curriculum task summaries."""

# ruff: noqa: S101

from __future__ import annotations

from copy import deepcopy

import pytest

from src import experiments, validation

VALID_TASK_COUNT = 2


def _valid_tasks() -> list[dict[str, object]]:
    """Return a small deterministic valid curriculum task list."""
    contracts = validation.contracts
    return [
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_HOVER,
            contracts.FIELD_DURATION_SEC: 2.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 5.0,
            contracts.FIELD_POSITION: [0.0, 0.0, 1.0],
        },
        {
            contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
            contracts.FIELD_SHAPE: contracts.SHAPE_LINE,
            contracts.FIELD_DURATION_SEC: 3.0,
            contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
            contracts.FIELD_START: [0.0, 0.0, 1.0],
            contracts.FIELD_END: [0.5, 0.0, 1.0],
        },
    ]


def test_summarize_loaded_config_dictionary_with_valid_tasks() -> None:
    """Verify a loaded config dictionary summarizes valid supported tasks."""
    config = {"tasks": _valid_tasks()}

    summary = experiments.curriculum.summarize_config_tasks(config)

    assert summary.total_count == VALID_TASK_COUNT
    assert summary.valid_count == VALID_TASK_COUNT
    assert summary.invalid_count == 0
    assert summary.shape_counts == {"hover": 1, "line": 1}
    assert all(task_summary.is_valid for task_summary in summary.task_summaries)
    assert all(task_summary.messages == () for task_summary in summary.task_summaries)


def test_summarize_smoke_config_path() -> None:
    """Verify the smoke config can be loaded and summarized through the path helper."""
    summary = experiments.curriculum.summarize_config_path("configs/smoke/trajectory_validation.yaml")

    expected_smoke_shapes = {"hover", "circle", "line", "vertical", "polyline"}

    assert expected_smoke_shapes.issubset(validation.contracts.SUPPORTED_TRAJECTORY_SHAPES)
    assert summary.total_count == len(expected_smoke_shapes)
    assert summary.valid_count == len(expected_smoke_shapes)
    assert summary.invalid_count == 0
    assert set(summary.shape_counts) == expected_smoke_shapes
    assert summary.valid_count == summary.total_count


def test_invalid_task_is_reported_without_crashing_summary() -> None:
    """Verify invalid tasks become invalid summary entries with diagnostics."""
    config = {
        "tasks": [
            {
                "task_type": "trajectory",
                "shape": "hover",
                "duration_sec": 2.0,
                "sample_rate_hz": 5.0,
                "position": [3.0, 0.0, 1.0],
            }
        ]
    }

    summary = experiments.curriculum.summarize_config_tasks(config)

    assert summary.total_count == 1
    assert summary.valid_count == 0
    assert summary.invalid_count == 1
    assert summary.task_summaries[0].shape == "hover"
    assert any("arena" in message for message in summary.task_summaries[0].messages)


def test_missing_task_container_raises_value_error() -> None:
    """Verify configs must contain a task list."""
    with pytest.raises(ValueError, match="tasks"):
        experiments.curriculum.summarize_config_tasks({"name": "missing"})


def test_non_list_task_container_raises_value_error() -> None:
    """Verify task containers must be lists."""
    with pytest.raises(ValueError, match="must be a list"):
        experiments.curriculum.summarize_config_tasks({"tasks": {"shape": "hover"}})


def test_non_mapping_task_entry_raises_value_error() -> None:
    """Verify each task entry must be a mapping."""
    with pytest.raises(ValueError, match="task at index 0"):
        experiments.curriculum.summarize_config_tasks({"tasks": ["not-a-task"]})


def test_summarize_config_does_not_mutate_input() -> None:
    """Verify summarization does not mutate config or task dictionaries."""
    config = {"tasks": _valid_tasks()}
    original = deepcopy(config)

    experiments.curriculum.summarize_config_tasks(config)

    assert config == original


def test_task_shape_summary_counts_shapes() -> None:
    """Verify the convenience shape summary returns shape counts only."""
    config = {"tasks": _valid_tasks()}

    assert experiments.curriculum.summarize_task_shapes(config) == {"hover": 1, "line": 1}


def test_curriculum_imports_through_package_alias() -> None:
    """Verify curriculum helpers are exposed by the experiments package."""
    assert experiments.curriculum.summarize_config_tasks is not None
