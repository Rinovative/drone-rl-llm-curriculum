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
CURRICULUM_DIAGNOSTIC_STEPS = 120


def _manual_curriculum_task(stage_name: str) -> dict[str, object]:
    """Return a copied task from the manual line curriculum fixture."""
    config = experiments.config.load_experiment_config("configs/curricula/manual_line_curriculum.yaml")
    for stage in config["stages"]:
        if stage["stage_name"] == stage_name:
            return dict(stage["task"])
    message = f"manual curriculum stage not found: {stage_name}"
    raise AssertionError(message)


def test_ppo_tracking_imports_through_package_alias() -> None:
    """Verify PPO tracking helpers are exposed by the experiments package."""
    assert experiments.ppo_tracking is not None
    assert experiments.ppo_tracking.PPOTrackingSmokeSettings is not None


def test_load_ppo_tracking_smoke_config_returns_valid_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the smoke YAML file loads into validated settings."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = experiments.ppo_tracking.load_ppo_tracking_settings("configs/training/ppo_tracking.yaml")

    assert settings.training_config_path == Path("configs/training/ppo_tracking.yaml")
    assert settings.task_config_path == Path("configs/smoke/trajectory_validation.yaml")
    assert settings.task_index == EXPECTED_SMOKE_TASK_INDEX
    assert settings.task_shape == "hover"
    assert settings.run_name is None
    assert settings.total_timesteps == EXPECTED_SMOKE_TIMESTEPS
    assert settings.eval_steps == EXPECTED_SMOKE_EVAL_STEPS
    assert settings.seed == 0
    assert settings.output_dir is None
    assert settings.model_dir is None
    assert settings.normalize_actions is True
    assert settings.wandb_mode == "auto"
    assert settings.wandb_project == utils.wandb.DEFAULT_WANDB_PROJECT
    assert settings.wandb_entity is None
    assert settings.wandb_group is None
    assert settings.wandb_name is None
    assert settings.wandb_tags == ()
    assert settings.wandb_dir is None
    assert experiments.ppo_tracking.default_output_dir() == tmp_path / "training_runs" / "ppo_hover_4096_seed0"
    assert experiments.ppo_tracking.default_model_dir() == tmp_path / "training_runs" / "ppo_hover_4096_seed0" / "models"
    assert experiments.ppo_tracking.default_manifests_dir() == tmp_path / "training_runs" / "ppo_hover_4096_seed0" / "manifests"
    assert utils.wandb.default_wandb_dir("ppo_hover_4096_seed0") == tmp_path / "training_runs" / "ppo_hover_4096_seed0" / "wandb"


def test_ppo_tracking_settings_reject_invalid_timesteps() -> None:
    """Verify invalid PPO timestep budgets are rejected."""
    with pytest.raises(ValueError, match="total_timesteps must be positive"):
        experiments.ppo_tracking.PPOTrackingSmokeSettings(total_timesteps=0)


def test_ppo_tracking_settings_reject_invalid_eval_steps() -> None:
    """Verify invalid evaluation lengths are rejected."""
    with pytest.raises(ValueError, match="eval_steps must be positive"):
        experiments.ppo_tracking.PPOTrackingSmokeSettings(eval_steps=0)


def test_ppo_tracking_settings_reject_invalid_run_name() -> None:
    """Verify training run names cannot escape storage/training_runs."""
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
        task_shape="hover",
        output_dir=tmp_path / "run",
        model_dir=tmp_path / "run" / "models",
    )

    model_path = experiments.ppo_tracking._resolve_model_path(settings)  # noqa: SLF001
    metrics_path = experiments.ppo_tracking._resolve_metrics_path(settings)  # noqa: SLF001

    assert model_path == (tmp_path / "run" / "models" / "ppo_hover_4096_seed0.zip").resolve(strict=False)
    assert metrics_path == (tmp_path / "run" / "metrics" / "ppo_hover_4096_seed0_metrics.json").resolve(strict=False)


def test_ppo_tracking_auto_run_name_uses_resolved_task_shape() -> None:
    """Verify omitted run names are derived from task shape, timesteps, and seed."""
    settings = experiments.ppo_tracking.PPOTrackingSmokeSettings(task_shape="line", total_timesteps=10000, seed=0)

    assert experiments.ppo_tracking._timesteps_label(4096) == "4096"  # noqa: SLF001
    assert experiments.ppo_tracking._timesteps_label(10000) == "10k"  # noqa: SLF001
    assert experiments.ppo_tracking._timesteps_label(1000000) == "1m"  # noqa: SLF001
    assert experiments.ppo_tracking._run_name(settings) == "ppo_line_10k_seed0"  # noqa: SLF001


def test_ppo_tracking_run_name_controls_default_artifact_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run-name defaults place training artifacts under one run root."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = experiments.ppo_tracking.PPOTrackingSmokeSettings(run_name="ppo_line_smoke")

    assert (
        experiments.ppo_tracking._resolve_model_path(settings) == tmp_path / "training_runs" / "ppo_line_smoke" / "models" / "ppo_line_smoke.zip"  # noqa: SLF001
    )
    assert (
        experiments.ppo_tracking._resolve_metrics_path(settings)  # noqa: SLF001
        == tmp_path / "training_runs" / "ppo_line_smoke" / "metrics" / "ppo_line_smoke_metrics.json"
    )
    assert (
        experiments.ppo_tracking._resolve_manifest_path(settings)  # noqa: SLF001
        == tmp_path / "training_runs" / "ppo_line_smoke" / "manifests" / "ppo_line_smoke_manifest.json"
    )
    wandb_settings = experiments.ppo_tracking._wandb_settings(settings, "line")  # noqa: SLF001
    assert wandb_settings.dir == tmp_path / "training_runs" / "ppo_line_smoke" / "wandb"
    assert wandb_settings.name == "ppo_line_smoke"
    assert wandb_settings.group == "ppo_tracking/line"


def test_ppo_tracking_manifest_includes_failure_diagnostics_paths(tmp_path: Path) -> None:
    """Verify PPO manifests duplicate compact diagnostics summary and artifact paths."""
    settings = experiments.ppo_tracking.PPOTrackingSmokeSettings(run_name="ppo_line_smoke")
    metrics = {
        "training_run_name": "ppo_line_smoke",
        "model_path": str(tmp_path / "models" / "ppo_line_smoke.zip"),
        "metrics_path": str(tmp_path / "metrics" / "ppo_line_smoke_metrics.json"),
        "manifest_path": str(tmp_path / "manifests" / "ppo_line_smoke_manifest.json"),
        "logs_dir": str(tmp_path / "logs"),
        "diagnostics_dir": str(tmp_path / "diagnostics"),
        "evaluation_trace_path": str(tmp_path / "diagnostics" / "evaluation_trace.jsonl"),
        "episode_summaries_path": str(tmp_path / "diagnostics" / "episode_summaries.json"),
        "failure_report_path": str(tmp_path / "diagnostics" / "failure_report.json"),
        "curriculum_feedback_path": str(tmp_path / "diagnostics" / "curriculum_feedback.json"),
        "failure_primary_mode": "hover_lock",
        "failure_modes": ["hover_lock", "insufficient_xy_motion"],
        "failure_overall_status": "failed",
        "curriculum_readiness_level": "line_not_ready",
        "curriculum_recommended_next_tasks": ["short_slow_line"],
        "curriculum_avoid_next_tasks": ["circle"],
    }

    manifest = experiments.ppo_tracking._build_manifest(  # noqa: SLF001
        settings=settings,
        metrics=metrics,
        task_source="config",
        selected_task_index=0,
        task={"shape": "line"},
    )

    assert manifest["diagnostics_dir"] == str(tmp_path / "diagnostics")
    assert manifest["evaluation_trace_path"].endswith("evaluation_trace.jsonl")
    assert manifest["failure_primary_mode"] == "hover_lock"
    assert manifest["curriculum_readiness_level"] == "line_not_ready"


def test_ppo_tracking_diagnostic_artifact_paths_include_feedback_files(tmp_path: Path) -> None:
    """Verify W&B artifact path selection includes final diagnostic files."""
    metrics = {
        "failure_report_path": str(tmp_path / "failure_report.json"),
        "curriculum_feedback_path": str(tmp_path / "curriculum_feedback.json"),
        "episode_summaries_path": str(tmp_path / "episode_summaries.json"),
        "evaluation_trace_path": str(tmp_path / "evaluation_trace.jsonl"),
        "metrics_path": str(tmp_path / "metrics.json"),
    }

    artifact_paths = experiments.ppo_tracking._diagnostic_artifact_paths("ppo_line_smoke", metrics)  # noqa: SLF001

    assert artifact_paths == {
        "ppo_line_smoke_failure_report": tmp_path / "failure_report.json",
        "ppo_line_smoke_curriculum_feedback": tmp_path / "curriculum_feedback.json",
        "ppo_line_smoke_episode_summaries": tmp_path / "episode_summaries.json",
        "ppo_line_smoke_evaluation_trace": tmp_path / "evaluation_trace.jsonl",
    }


def test_ppo_tracking_explicit_output_dir_remains_direct(tmp_path: Path) -> None:
    """Verify explicit output directory overrides preserve direct metrics placement."""
    settings = experiments.ppo_tracking.PPOTrackingSmokeSettings(
        task_shape="hover",
        output_dir=tmp_path / "storage" / "results" / "custom",
    )

    metrics_path = experiments.ppo_tracking._resolve_metrics_path(settings)  # noqa: SLF001

    assert metrics_path == (tmp_path / "storage" / "results" / "custom" / "ppo_hover_4096_seed0_metrics.json").resolve(strict=False)


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
    assert metadata["action_space_low"] == [[-1.0, -1.0, -1.0]]
    assert metadata["action_space_high"] == [[1.0, 1.0, 1.0]]
    assert metadata["actions_normalized"] is True
    assert metadata["real_action_space_low"] != metadata["action_space_low"]
    assert metadata["base_action_type"] == "pid"
    assert "x/y/z movement" in metadata["base_action_semantics"]


def test_normalized_action_wrapper_maps_to_real_pid_bounds() -> None:
    """Verify normalized PPO actions map explicitly to real tracking action bounds."""
    task = experiments.ppo_tracking._load_task(  # noqa: SLF001
        experiments.ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        experiments.ppo_tracking.DEFAULT_TASK_INDEX,
    )
    real_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False, max_steps=DIAGNOSTIC_STEPS)
    wrapped_env = envs.tracking_env.make_normalized_action_env(real_env)
    try:
        low = np.asarray(real_env.action_space.low, dtype=np.float32)
        high = np.asarray(real_env.action_space.high, dtype=np.float32)
        midpoint = (low + high) * 0.5

        assert np.allclose(wrapped_env.action_space.low, -1.0)
        assert np.allclose(wrapped_env.action_space.high, 1.0)
        assert np.allclose(wrapped_env.normalized_to_real_action(np.zeros((1, 3), dtype=np.float32)), midpoint)
        assert np.allclose(wrapped_env.normalized_to_real_action(-np.ones((1, 3), dtype=np.float32)), low)
        assert np.allclose(wrapped_env.normalized_to_real_action(np.ones((1, 3), dtype=np.float32)), high)
        assert np.allclose(wrapped_env.real_to_normalized_action(midpoint), 0.0)
    finally:
        wrapped_env.close()


def test_ppo_training_env_uses_normalized_action_space_when_enabled() -> None:
    """Verify PPO receives the normalized action wrapper when configured."""
    task = experiments.ppo_tracking._load_task(  # noqa: SLF001
        experiments.ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        experiments.ppo_tracking.DEFAULT_TASK_INDEX,
    )
    real_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False, max_steps=DIAGNOSTIC_STEPS)
    training_env = experiments.ppo_tracking._ppo_training_env(real_env, normalize_actions=True)  # noqa: SLF001
    try:
        assert training_env.action_space.shape == (1, 3)
        assert np.allclose(training_env.action_space.low, -1.0)
        assert np.allclose(training_env.action_space.high, 1.0)
        assert training_env.real_action_space is real_env.action_space
    finally:
        training_env.close()


def test_task_with_minimum_reference_samples_extends_hold_move_task_without_raising() -> None:
    """Verify start-hold curriculum diagnostics extend duration instead of failing."""
    task = _manual_curriculum_task("start_hold_then_short_line")

    diagnostic_task, reference_samples, warnings = experiments.ppo_tracking._task_with_minimum_reference_samples(  # noqa: SLF001
        task,
        required_steps=CURRICULUM_DIAGNOSTIC_STEPS,
    )

    assert reference_samples >= CURRICULUM_DIAGNOSTIC_STEPS + 1
    assert diagnostic_task["sample_rate_hz"] == task["sample_rate_hz"]
    assert diagnostic_task["hold_duration_sec"] == task["hold_duration_sec"]
    assert diagnostic_task["move_duration_sec"] > task["move_duration_sec"]
    assert diagnostic_task["start_hold_enabled"] is True
    assert diagnostic_task["start_hold_sec"] == task["start_hold_sec"]
    assert diagnostic_task["exclude_start_hold_from_tracking_metrics"] is True
    assert any("move_duration_sec" in warning for warning in warnings)


def test_task_with_minimum_reference_samples_prefers_duration_over_sample_rate() -> None:
    """Verify diagnostics avoid unsafe sample-rate densification for line tasks."""
    task = _manual_curriculum_task("line")
    unsafe_sample_rate_task = dict(task)
    unsafe_sample_rate_task["sample_rate_hz"] = 41.0
    with pytest.raises(ValueError, match="maximum acceleration"):
        envs.task_adapter.make_task_reference(unsafe_sample_rate_task)

    diagnostic_task, reference_samples, warnings = experiments.ppo_tracking._task_with_minimum_reference_samples(  # noqa: SLF001
        task,
        required_steps=CURRICULUM_DIAGNOSTIC_STEPS,
    )

    assert reference_samples >= CURRICULUM_DIAGNOSTIC_STEPS + 1
    assert diagnostic_task["sample_rate_hz"] == task["sample_rate_hz"]
    assert diagnostic_task["duration_sec"] > task["duration_sec"]
    assert diagnostic_task["start_hold_enabled"] is True
    assert diagnostic_task["start_hold_sec"] == task["start_hold_sec"]
    assert diagnostic_task["exclude_start_hold_from_tracking_metrics"] is True
    assert any("duration_sec" in warning for warning in warnings)
    assert not any("sample_rate_hz from" in warning for warning in warnings)


def test_task_with_minimum_reference_samples_falls_back_when_extension_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a failed diagnostic extension returns the original task with a warning."""
    task = _manual_curriculum_task("line")
    original_reference = envs.task_adapter.make_task_reference(task)
    real_make_task_reference = envs.task_adapter.make_task_reference

    def fake_make_task_reference(reference_task: dict[str, object]) -> object:
        if reference_task.get("duration_sec") != task["duration_sec"]:
            message = "forced extension failure"
            raise ValueError(message)
        return real_make_task_reference(reference_task)

    monkeypatch.setattr(envs.task_adapter, "make_task_reference", fake_make_task_reference)

    diagnostic_task, reference_samples, warnings = experiments.ppo_tracking._task_with_minimum_reference_samples(  # noqa: SLF001
        task,
        required_steps=CURRICULUM_DIAGNOSTIC_STEPS,
    )

    assert diagnostic_task == task
    assert reference_samples == original_reference.positions.shape[0]
    assert any("duration extension failed validation" in warning for warning in warnings)
    assert any("using original diagnostic task" in warning for warning in warnings)


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
    real_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False, max_steps=DIAGNOSTIC_STEPS)
    tracking_env = envs.tracking_env.make_normalized_action_env(real_env)
    try:
        settings = experiments.ppo_tracking.PPOTrackingSmokeSettings(eval_steps=DIAGNOSTIC_STEPS)
        metrics = experiments.ppo_tracking._evaluate_model(HighActionModel(), tracking_env, settings)  # noqa: SLF001
    finally:
        tracking_env.close()

    assert "position_bounds" in metrics
    assert "reference_position_bounds" in metrics
    assert "actual_z_span_m" in metrics
    assert "xy_tracking_ratio" in metrics
    assert metrics["action_bounds"]["max"] == [1.0, 1.0, 1.0]
    assert metrics["action_mean"] == [1.0, 1.0, 1.0]
    assert metrics["action_saturation_fraction"] == [1.0, 1.0, 1.0]
    assert metrics["actions_normalized"] is True
    assert "real_action_mean" in metrics
    assert metrics["real_action_saturation_fraction"] == [1.0, 1.0, 1.0]
    assert "mean_abs_x_error" in metrics
    assert "final_abs_z_error" in metrics


def test_cli_train_tracking_parser_accepts_task_shape_and_run_name() -> None:
    """Verify the training parser exposes task-specific run controls."""
    parser = cli_train_tracking.build_parser()
    args = parser.parse_args(["--task-shape", "line", "--run-name", "ppo_line_smoke"])

    assert args.task_shape == "line"
    assert args.run_name == "ppo_line_smoke"


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
    assert "storage/training_runs" in completed.stdout
