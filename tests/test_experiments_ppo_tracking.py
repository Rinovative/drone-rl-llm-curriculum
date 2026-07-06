"""Tests for tiny PPO trajectory-tracking smoke training helpers."""

# ruff: noqa: S101

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from src import experiments

EXPECTED_SMOKE_TASK_INDEX = 2
EXPECTED_SMOKE_TIMESTEPS = 128
EXPECTED_SMOKE_EVAL_STEPS = 32


def test_ppo_tracking_imports_through_package_alias() -> None:
    """Verify PPO tracking helpers are exposed by the experiments package."""
    assert experiments.ppo_tracking is not None
    assert experiments.ppo_tracking.PPOTrackingSmokeSettings is not None


def test_load_ppo_tracking_smoke_config_returns_valid_settings() -> None:
    """Verify the smoke YAML file loads into validated settings."""
    settings = experiments.ppo_tracking.load_ppo_tracking_settings("configs/smoke/ppo_tracking_smoke.yaml")

    assert settings.task_config_path == Path("configs/smoke/trajectory_validation.yaml")
    assert settings.task_index == EXPECTED_SMOKE_TASK_INDEX
    assert settings.total_timesteps == EXPECTED_SMOKE_TIMESTEPS
    assert settings.eval_steps == EXPECTED_SMOKE_EVAL_STEPS
    assert settings.seed == 0


def test_ppo_tracking_settings_reject_invalid_timesteps() -> None:
    """Verify invalid PPO timestep budgets are rejected."""
    with pytest.raises(ValueError, match="total_timesteps must be positive"):
        experiments.ppo_tracking.PPOTrackingSmokeSettings(total_timesteps=0)


def test_ppo_tracking_settings_reject_invalid_eval_steps() -> None:
    """Verify invalid evaluation lengths are rejected."""
    with pytest.raises(ValueError, match="eval_steps must be positive"):
        experiments.ppo_tracking.PPOTrackingSmokeSettings(eval_steps=0)


def test_ppo_tracking_paths_resolve_under_caller_directories(tmp_path: Path) -> None:
    """Verify caller-provided output directories control generated artifact paths."""
    settings = experiments.ppo_tracking.PPOTrackingSmokeSettings(
        output_dir=tmp_path / "results",
        model_dir=tmp_path / "models",
    )

    model_path = experiments.ppo_tracking._resolve_model_path(settings)  # noqa: SLF001
    metrics_path = experiments.ppo_tracking._resolve_metrics_path(settings)  # noqa: SLF001

    assert model_path == (tmp_path / "models" / "ppo_tracking_smoke.zip").resolve(strict=False)
    assert metrics_path == (tmp_path / "results" / "ppo_tracking_smoke_metrics.json").resolve(strict=False)


def test_ppo_tracking_dependency_detection_returns_booleans() -> None:
    """Verify dependency detection reports simple boolean availability flags."""
    dependencies = experiments.ppo_tracking.detect_ppo_tracking_dependencies()

    assert isinstance(dependencies["stable_baselines3"], bool)
    assert isinstance(dependencies["gymnasium"], bool)
    assert isinstance(dependencies["torch"], bool)


def test_cli_train_tracking_help_works() -> None:
    """Verify the PPO tracking CLI exposes help without running training."""
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.cli_train_tracking", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--total-timesteps" in completed.stdout
    assert "--eval-steps" in completed.stdout
