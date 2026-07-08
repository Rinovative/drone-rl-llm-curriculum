"""Tests for minimal experiment configuration loading."""

# ruff: noqa: S101

from __future__ import annotations

from pathlib import Path

import pytest

from src import validation
from src.experiments import experiments_config
from src.experiments.curriculum import experiments_curriculum_training as curriculum_training
from src.experiments.training import experiments_training_ppo_config as ppo_config
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking

EXPECTED_SMOKE_TASK_COUNT = 5
REQUIRED_SHAPES = {"hover", "circle", "line", "vertical", "polyline"}


def test_smoke_config_loads_and_contains_valid_tasks() -> None:
    """Verify the smoke config loads and its tasks pass deterministic validation."""
    config = experiments_config.load_experiment_config("configs/smoke/trajectory_validation.yaml")

    assert config["name"] == "trajectory_validation_smoke"
    assert config["seed"] == 0
    assert len(config["tasks"]) == EXPECTED_SMOKE_TASK_COUNT

    shapes = [task["shape"] for task in config["tasks"]]
    assert set(shapes) == REQUIRED_SHAPES, f"Expected shapes {REQUIRED_SHAPES}, but got {set(shapes)}"
    for shape in REQUIRED_SHAPES:
        assert shapes.count(shape) == 1

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
        experiments_config.load_experiment_config(config_path)


def test_non_mapping_yaml_config_fails(tmp_path: Path) -> None:
    """Verify YAML roots must be mappings."""
    config_path = tmp_path / "list.yaml"
    config_path.write_text("- not\n- a\n- mapping\n", encoding="utf-8")

    with pytest.raises(ValueError, match="mapping"):
        experiments_config.load_experiment_config(config_path)


def test_real_direct_ppo_training_configs_use_production_tasks_and_nested_ppo() -> None:
    """Verify direct PPO smoke/medium/final configs use the canonical task source and nested PPO."""
    expected = {
        "configs/training/ppo_tracking_smoke.yaml": {
            "run_name": "direct_ppo_line_smoke_seed0",
            "total_timesteps": 4096,
            "num_envs": 1,
            "n_steps": 256,
            "batch_size": 64,
            "n_epochs": 5,
            "wandb_mode": "disabled",
            "task_index": 0,
        },
        "configs/training/ppo_tracking_medium.yaml": {
            "run_name": "direct_ppo_line_medium_seed0",
            "total_timesteps": 250000,
            "num_envs": 4,
            "n_steps": 512,
            "batch_size": 128,
            "n_epochs": 5,
            "wandb_mode": "offline",
            "task_index": 0,
        },
        "configs/training/ppo_tracking_final.yaml": {
            "run_name": "direct_ppo_line_final_seed0",
            "total_timesteps": 1000000,
            "num_envs": 8,
            "n_steps": 1024,
            "batch_size": 256,
            "n_epochs": 10,
            "wandb_mode": "auto",
            "task_index": 1,
        },
    }

    for config_path, values in expected.items():
        config = experiments_config.load_experiment_config(config_path)
        flat_ppo_keys = set(config) & set(ppo_config.PPO_CONFIG_KEYS)

        assert flat_ppo_keys == set()
        assert config["task_config_path"] == "configs/training/ppo_tracking_tasks.yaml"
        assert config["run_name"] == values["run_name"]
        assert config["total_timesteps"] == values["total_timesteps"]
        assert config["num_envs"] == values["num_envs"]
        assert config["task_index"] == values["task_index"]
        assert config["wandb_mode"] == values["wandb_mode"]
        assert config["ppo"]["n_steps"] == values["n_steps"]
        assert config["ppo"]["batch_size"] == values["batch_size"]
        assert config["ppo"]["n_epochs"] == values["n_epochs"]

        settings = ppo_tracking.load_ppo_tracking_settings(config_path)
        assert settings.run_name == values["run_name"]
        assert settings.num_envs == values["num_envs"]
        assert settings.ppo_config.to_dict() == config["ppo"]


def test_real_direct_ppo_task_source_validates_and_final_uses_hard_line() -> None:
    """Verify the production direct PPO task source is valid and final selects the hard line."""
    task_config = experiments_config.load_experiment_config("configs/training/ppo_tracking_tasks.yaml")
    final_config = experiments_config.load_experiment_config("configs/training/ppo_tracking_final.yaml")
    limits = validation.tasks.ValidationLimits(**task_config["validation_limits"])

    assert [task["task_name"] for task in task_config["tasks"]] == [
        "line_basic",
        "line_long_final",
        "line_diagonal_validation",
    ]
    for task in task_config["tasks"]:
        result = validation.tasks.validate_task(task, limits=limits)
        assert result.is_valid, result.messages
        assert result.trajectory is not None

    final_task = task_config["tasks"][final_config["task_index"]]
    assert final_task["task_name"] == "line_long_final"
    assert final_task["end"] == [1.5, 0.0, 1.0]


def test_real_manual_curriculum_configs_are_canonical_and_valid() -> None:
    """Verify manual curriculum tiers have canonical names, budgets, and run IDs."""
    expected = {
        "configs/curricula/curriculum_manual_line_smoke.yaml": {
            "name": "curriculum_manual_line_smoke",
            "run_name": "curriculum_manual_line_smoke_seed0",
            "base": "configs/training/ppo_tracking_smoke.yaml",
            "budgets": [4096, 4096, 4096, 4096, 4096],
            "wandb_mode": "disabled",
        },
        "configs/curricula/curriculum_manual_line_medium.yaml": {
            "name": "curriculum_manual_line_medium",
            "run_name": "curriculum_manual_line_medium_seed0",
            "base": "configs/training/ppo_tracking_medium.yaml",
            "budgets": [25000, 25000, 50000, 50000, 100000],
            "wandb_mode": "offline",
        },
        "configs/curricula/curriculum_manual_line_final.yaml": {
            "name": "curriculum_manual_line_final",
            "run_name": "curriculum_manual_line_final_seed0",
            "base": "configs/training/ppo_tracking_final.yaml",
            "budgets": [100000, 100000, 150000, 175000, 225000, 250000],
            "wandb_mode": "auto",
        },
    }

    for config_path, values in expected.items():
        settings = curriculum_training.load_manual_curriculum_settings(config_path)

        assert settings.curriculum_name == values["name"]
        assert settings.base_training_config == Path(values["base"])
        assert settings.wandb_mode == values["wandb_mode"]
        assert [stage.total_timesteps for stage in settings.stages] == values["budgets"]
        assert curriculum_training._curriculum_artifact_run_name(settings.curriculum_name, settings.seed) == values["run_name"]  # noqa: SLF001
        assert [stage.stage_name for stage in settings.stages[:5]] == [
            "hover_stabilization",
            "nearby_target_hover",
            "start_hold_then_short_line",
            "short_slow_line",
            "line",
        ]
        if config_path.endswith("final.yaml"):
            assert settings.stages[-1].stage_name == "long_line"
            assert settings.stages[-1].task["end"] == [1.5, 0.0, 1.0]
        curriculum_training.validate_manual_curriculum(settings)


def test_active_configs_do_not_reference_removed_benchmark_or_legacy_storage_roots() -> None:
    """Verify active configs do not reintroduce removed benchmark or storage roots."""
    benchmark_kind = "benchmarks"
    storage_run_suffix = "runs"
    report_suffix = "reports"
    removed_benchmark_name = f"curriculum_{benchmark_kind}.yaml"
    removed_benchmark_path = Path("configs") / "evaluation" / removed_benchmark_name
    forbidden = (
        str(removed_benchmark_path),
        "storage/" + f"training_{storage_run_suffix}",
        "storage/" + f"evaluation_{storage_run_suffix}",
        "storage/" + f"comparison_{report_suffix}",
    )

    assert not removed_benchmark_path.exists()
    for config_path in Path("configs").rglob("*.yaml"):
        text = config_path.read_text(encoding="utf-8")
        for value in forbidden:
            assert value not in text, f"{config_path} references {value}"
