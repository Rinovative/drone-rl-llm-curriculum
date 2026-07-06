"""Tests for tiny PPO trajectory-tracking smoke training helpers."""

# ruff: noqa: S101

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src import envs, experiments, utils
from src.experiments import cli_train_tracking

EXPECTED_SMOKE_TASK_INDEX = 0
LINE_TASK_INDEX = 2
EXPECTED_SMOKE_TIMESTEPS = 4096
EXPECTED_SMOKE_EVAL_STEPS = 120
DIAGNOSTIC_STEPS = 6


def test_ppo_tracking_imports_through_package_alias() -> None:
    """Verify PPO tracking helpers are exposed by the experiments package."""
    assert experiments.ppo_tracking is not None
    assert experiments.ppo_tracking.PPOTrackingSmokeSettings is not None


def test_load_ppo_tracking_smoke_config_returns_valid_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the smoke YAML file loads into validated settings."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = experiments.ppo_tracking.load_ppo_tracking_settings("configs/smoke/ppo_tracking_smoke.yaml")

    assert settings.task_config_path == Path("configs/smoke/trajectory_validation.yaml")
    assert settings.task_index == EXPECTED_SMOKE_TASK_INDEX
    assert settings.task_shape is None
    assert settings.run_name is None
    assert settings.total_timesteps == EXPECTED_SMOKE_TIMESTEPS
    assert settings.eval_steps == EXPECTED_SMOKE_EVAL_STEPS
    assert settings.seed == 0
    assert settings.output_dir is None
    assert settings.model_dir is None
    assert settings.wandb_mode == "disabled"
    assert settings.wandb_project == utils.wandb.DEFAULT_WANDB_PROJECT
    assert settings.wandb_entity is None
    assert settings.wandb_group is None
    assert settings.wandb_name is None
    assert settings.wandb_tags == ()
    assert settings.wandb_dir is None
    assert experiments.ppo_tracking.default_output_dir() == tmp_path / "runs" / "ppo_tracking_smoke"
    assert experiments.ppo_tracking.default_model_dir() == tmp_path / "runs" / "ppo_tracking_smoke" / "models"
    assert utils.wandb.default_wandb_dir() == tmp_path / "runs" / "ppo_tracking_smoke" / "wandb"


def test_ppo_tracking_settings_reject_invalid_timesteps() -> None:
    """Verify invalid PPO timestep budgets are rejected."""
    with pytest.raises(ValueError, match="total_timesteps must be positive"):
        experiments.ppo_tracking.PPOTrackingSmokeSettings(total_timesteps=0)


def test_ppo_tracking_settings_reject_invalid_eval_steps() -> None:
    """Verify invalid evaluation lengths are rejected."""
    with pytest.raises(ValueError, match="eval_steps must be positive"):
        experiments.ppo_tracking.PPOTrackingSmokeSettings(eval_steps=0)


def test_ppo_tracking_settings_reject_invalid_run_name() -> None:
    """Verify training run names cannot escape storage/runs."""
    with pytest.raises(ValueError, match="run_name"):
        experiments.ppo_tracking.PPOTrackingSmokeSettings(run_name="../bad")


def test_ppo_tracking_select_task_by_shape_uses_configured_task() -> None:
    """Verify task-shape selection reuses the configured task list."""
    task, task_source, task_index, warnings = experiments.ppo_tracking._select_task(  # noqa: SLF001
        task_config_path=Path("configs/smoke/trajectory_validation.yaml"),
        default_task_index=0,
        task_shape="line",
    )

    assert task["shape"] == "line"
    assert task_source == "shape_override"
    assert task_index == LINE_TASK_INDEX
    assert warnings


def test_ppo_tracking_paths_resolve_under_run_directories(tmp_path: Path) -> None:
    """Verify caller-provided run roots control generated artifact paths."""
    settings = experiments.ppo_tracking.PPOTrackingSmokeSettings(
        output_dir=tmp_path / "run",
        model_dir=tmp_path / "run" / "models",
    )

    model_path = experiments.ppo_tracking._resolve_model_path(settings)  # noqa: SLF001
    metrics_path = experiments.ppo_tracking._resolve_metrics_path(settings)  # noqa: SLF001

    assert model_path == (tmp_path / "run" / "models" / "ppo_tracking_smoke.zip").resolve(strict=False)
    assert metrics_path == (tmp_path / "run" / "metrics" / "ppo_tracking_smoke_metrics.json").resolve(strict=False)


def test_ppo_tracking_run_name_controls_default_artifact_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run-name defaults place training artifacts under one run root."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = experiments.ppo_tracking.PPOTrackingSmokeSettings(run_name="ppo_tracking_line_smoke")

    assert (
        experiments.ppo_tracking._resolve_model_path(settings) == tmp_path / "runs" / "ppo_tracking_line_smoke" / "models" / "ppo_tracking_smoke.zip"  # noqa: SLF001
    )
    assert (
        experiments.ppo_tracking._resolve_metrics_path(settings)  # noqa: SLF001
        == tmp_path / "runs" / "ppo_tracking_line_smoke" / "metrics" / "ppo_tracking_smoke_metrics.json"
    )
    assert experiments.ppo_tracking._wandb_settings(settings).dir == tmp_path / "runs" / "ppo_tracking_line_smoke" / "wandb"  # noqa: SLF001


def test_ppo_tracking_legacy_output_dir_remains_direct(tmp_path: Path) -> None:
    """Verify storage/results-style output overrides preserve legacy metrics placement."""
    settings = experiments.ppo_tracking.PPOTrackingSmokeSettings(output_dir=tmp_path / "storage" / "results" / "custom")

    metrics_path = experiments.ppo_tracking._resolve_metrics_path(settings)  # noqa: SLF001

    assert metrics_path == (tmp_path / "storage" / "results" / "custom" / "ppo_tracking_smoke_metrics.json").resolve(strict=False)


def test_ppo_tracking_dependency_detection_returns_booleans() -> None:
    """Verify dependency detection reports simple boolean availability flags."""
    dependencies = experiments.ppo_tracking.detect_ppo_tracking_dependencies()

    assert isinstance(dependencies["stable_baselines3"], bool)
    assert isinstance(dependencies["gymnasium"], bool)
    assert isinstance(dependencies["torch"], bool)


def test_ppo_runtime_info_reports_cuda_availability() -> None:
    """Verify runtime diagnostics expose torch/CUDA information without requiring GPU."""
    runtime = experiments.ppo_tracking.detect_ppo_runtime_info()

    assert isinstance(runtime["torch_available"], bool)
    assert isinstance(runtime["torch_cuda_available"], bool)
    assert isinstance(runtime["torch_cuda_device_count"], int)


def test_tracking_action_metadata_reports_pid_contract() -> None:
    """Verify action metadata captures the upstream PID action contract."""
    task = experiments.ppo_tracking._load_task(  # noqa: SLF001
        experiments.ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        experiments.ppo_tracking.DEFAULT_TASK_INDEX,
    )
    metadata = experiments.ppo_tracking.describe_tracking_env_action_metadata(task)

    assert metadata["action_space_shape"] == [1, 3]
    assert metadata["base_action_type"] == "pid"
    assert "x/y/z movement" in metadata["base_action_semantics"]


def test_liftoff_diagnostics_report_simple_policy_bounds() -> None:
    """Verify liftoff diagnostics include structured movement summaries."""
    task = experiments.ppo_tracking._load_task(  # noqa: SLF001
        experiments.ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        experiments.ppo_tracking.DEFAULT_TASK_INDEX,
    )
    diagnostics = experiments.ppo_tracking.run_liftoff_diagnostics(task, max_steps=DIAGNOSTIC_STEPS, seed=0)

    assert "zero_action" in diagnostics
    assert "high_action" in diagnostics
    assert diagnostics["high_action"]["z_max"] >= diagnostics["high_action"]["z_min"]
    assert diagnostics["high_action"]["base_action_shape"] == [1, 3]


def test_evaluate_model_metrics_include_movement_bounds() -> None:
    """Verify PPO evaluation metrics include action and position bounds."""

    class HighActionModel:
        """Tiny predict-only model used to avoid SB3 training in unit tests."""

        def predict(self, _observation: np.ndarray, deterministic: bool = True) -> tuple[np.ndarray, None]:
            """Return the bounded high action expected by PID tracking."""
            _ = deterministic
            return np.ones((1, 3), dtype=np.float32), None

    task = experiments.ppo_tracking._load_task(  # noqa: SLF001
        experiments.ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        experiments.ppo_tracking.DEFAULT_TASK_INDEX,
    )
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False, max_steps=DIAGNOSTIC_STEPS)
    try:
        settings = experiments.ppo_tracking.PPOTrackingSmokeSettings(eval_steps=DIAGNOSTIC_STEPS)
        metrics = experiments.ppo_tracking._evaluate_model(HighActionModel(), tracking_env, settings)  # noqa: SLF001
    finally:
        tracking_env.close()

    assert "position_bounds" in metrics
    assert "reference_position_bounds" in metrics
    assert "actual_z_span_m" in metrics
    assert metrics["action_bounds"]["max"] == [1.0, 1.0, 1.0]


def test_cli_train_tracking_parser_accepts_task_shape_and_run_name() -> None:
    """Verify the training parser exposes task-specific run controls."""
    parser = cli_train_tracking.build_parser()
    args = parser.parse_args(["--task-shape", "line", "--run-name", "ppo_tracking_line_smoke"])

    assert args.task_shape == "line"
    assert args.run_name == "ppo_tracking_line_smoke"


def test_cli_train_tracking_help_works() -> None:
    """Verify the PPO tracking CLI exposes help without running training."""
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.cli_train_tracking", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--task-shape" in completed.stdout
    assert "--run-name" in completed.stdout
    assert "--total-timesteps" in completed.stdout
    assert "--eval-steps" in completed.stdout
    assert "--wandb-mode" in completed.stdout
