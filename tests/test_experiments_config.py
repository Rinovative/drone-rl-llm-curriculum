"""Tests for minimal experiment configuration loading."""

# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from src import experiments, validation

EXPECTED_SMOKE_TASK_COUNT = 2

if TYPE_CHECKING:
    from pathlib import Path


def test_smoke_config_loads_and_contains_valid_tasks() -> None:
    """Verify the smoke config loads and its tasks pass deterministic validation."""
    config = experiments.config.load_experiment_config("configs/smoke/trajectory_validation.yaml")

    assert config["name"] == "trajectory_validation_smoke"
    assert config["seed"] == 0
    assert len(config["tasks"]) == EXPECTED_SMOKE_TASK_COUNT

    limits = validation.tasks.ValidationLimits(**config["validation_limits"])
    for task in config["tasks"]:
        result = validation.tasks.validate_task(task, limits=limits)

        assert result.is_valid, result.messages
        assert result.trajectory is not None


def test_empty_yaml_config_fails(tmp_path: Path) -> None:
    """Verify empty YAML files are rejected."""
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("", encoding="utf-8")

    with pytest.raises(ValueError, match="empty"):
        experiments.config.load_experiment_config(config_path)


def test_non_mapping_yaml_config_fails(tmp_path: Path) -> None:
    """Verify YAML roots must be mappings."""
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        experiments.config.load_experiment_config(config_path)
