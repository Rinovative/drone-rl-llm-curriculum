"""Tests for tiny PPO trajectory-tracking smoke training helpers."""

# ruff: noqa: S101

from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pytest

from src import envs, utils
from src.experiments import experiments_config
from src.experiments.cli import experiments_cli_train_tracking as cli_train_tracking
from src.experiments.training import experiments_training_ppo_config as ppo_config
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking

EXPECTED_SMOKE_TASK_INDEX = 0
LINE_TASK_INDEX = 0
EXPECTED_SMOKE_TIMESTEPS = 4096
EXPECTED_SMOKE_NUM_ENVS = 1
EXPECTED_SMOKE_EVAL_STEPS = 120
DIAGNOSTIC_STEPS = 6
CURRICULUM_DIAGNOSTIC_STEPS = 120
CONFIGURED_TEST_PPO_N_STEPS = 12
CONFIGURED_TEST_PPO_BATCH_SIZE = 6
CONFIGURED_TEST_NUM_ENVS = 2
VECTOR_TEST_N_STEPS = 8
VECTOR_TEST_BATCH_SIZE = 16
VECTOR_TEST_OVERSIZED_BATCH_SIZE = 17
VECTOR_TEST_EFFECTIVE_ROLLOUT_STEPS = 16
SUBPROC_TEST_NUM_ENVS = 3
SUBPROC_TEST_SEED = 11
CLI_NUM_ENVS_OVERRIDE = 3
BASE_OBSERVATION_DIM = 10
PID_DYNAMICS_PREVIOUS_OBSERVATION_DIM = 22
DIRECT_RPM_DYNAMICS_PREVIOUS_OBSERVATION_DIM = 23
PID_ACTION_DIM = 3
DIRECT_RPM_ACTION_DIM = 4
DIRECT_RPM_DELTA_SCALE = 0.05
POLICY_NET_ARCH = [128, 128]
STRICT_TEST_LIMIT_VIOLATIONS = 2

EXPECTED_PPO_CONFIG = {
    "policy": "MlpPolicy",
    "device": "cpu",
    "learning_rate": 0.0003,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "n_steps": 256,
    "batch_size": 64,
    "n_epochs": 5,
    "clip_range": 0.2,
    "ent_coef": 0.001,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "target_kl": 0.03,
    "policy_kwargs": ppo_config.default_policy_kwargs(),
}


class _FakeTrainingEnv:
    """Minimal environment stand-in for fast PPO training tests."""

    def __init__(self) -> None:
        """Track whether the fake environment was closed."""
        self.closed = False

    def close(self) -> None:
        """Close the fake environment."""
        self.closed = True


class _FakeWandbRun:
    """Tiny W&B run stand-in that records finish calls."""

    def __init__(self) -> None:
        """Create fake W&B metadata and event tracking."""
        self.id = "fake-run-id"
        self.url = "https://wandb.example/fake-run"
        self.finish_count = 0

    def finish(self) -> None:
        """Record that the run was finished."""
        self.finish_count += 1


def _install_fast_ppo_smoke_fakes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    wandb_run: _FakeWandbRun | None,
    learn_error: Exception | None = None,
) -> types.SimpleNamespace:
    """Install fake PPO, env, diagnostics, and W&B hooks for fast lifecycle tests."""

    class FakePPO:
        """Tiny SB3 PPO stand-in that can fail during learning."""

        def __init__(self, _policy: str, _env: _FakeTrainingEnv, **kwargs: Any) -> None:
            """Record device metadata used by downstream metrics."""
            self.device = kwargs["device"]

        def learn(self, **_kwargs: Any) -> None:
            """Succeed or raise the configured learning error."""
            if learn_error is not None:
                raise learn_error

        def save(self, path: str) -> None:
            """Write a tiny model marker file."""
            Path(path).write_text("fake model", encoding="utf-8")

    events: list[str] = []
    fake_training_vec_env = _FakeTrainingEnv()
    fake_eval_env = _FakeTrainingEnv()
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    fake_sb3 = types.ModuleType("stable_baselines3")
    fake_sb3.PPO = FakePPO
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    monkeypatch.setattr(
        ppo_tracking,
        "detect_ppo_tracking_dependencies",
        lambda: {"stable_baselines3": True, "gymnasium": True, "gym_pybullet_drones": True, "torch": True},
    )
    monkeypatch.setattr(ppo_tracking, "detect_ppo_runtime_info", lambda: {"torch_available": True})
    monkeypatch.setattr(ppo_tracking, "_select_task", lambda **_kwargs: ({"shape": "line"}, "config", 0, ()))
    monkeypatch.setattr(ppo_tracking, "run_liftoff_diagnostics", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(ppo_tracking, "_make_ppo_training_vec_env", lambda **_kwargs: fake_training_vec_env)
    monkeypatch.setattr(ppo_tracking, "_make_seeded_ppo_tracking_env", lambda **_kwargs: fake_eval_env)
    monkeypatch.setattr(ppo_tracking, "_tracking_env_action_metadata", lambda _env: {"actions_normalized": False})
    monkeypatch.setattr(ppo_tracking, "_movement_warnings", lambda **_kwargs: ())
    monkeypatch.setattr(ppo_tracking, "_wandb_callback", lambda _run: None)
    monkeypatch.setattr(
        ppo_tracking.evaluation.diagnostics,
        "collect_policy_evaluation_diagnostics",
        lambda **_kwargs: types.SimpleNamespace(metrics={"mean_position_error_m": 0.1}),
    )
    monkeypatch.setattr(ppo_tracking.evaluation.diagnostics, "write_policy_evaluation_diagnostics", lambda *_args, **_kwargs: {})

    def fake_start_wandb_run(*, settings: Any, config: dict[str, Any]) -> _FakeWandbRun | None:
        """Capture W&B startup without importing wandb."""
        _ = settings, config
        events.append("start")
        return wandb_run

    def fake_log_wandb_summary(run: Any | None, metrics: dict[str, Any]) -> None:
        """Record successful summary logging."""
        _ = run, metrics
        events.append("summary")

    def fake_log_wandb_artifacts(run: Any | None, paths: dict[str, Path]) -> None:
        """Record successful artifact logging."""
        _ = run, paths
        events.append("artifacts")

    monkeypatch.setattr(ppo_tracking.utils.wandb, "start_wandb_run", fake_start_wandb_run)
    monkeypatch.setattr(ppo_tracking.utils.wandb, "log_wandb_summary", fake_log_wandb_summary)
    monkeypatch.setattr(ppo_tracking.utils.wandb, "log_wandb_artifacts", fake_log_wandb_artifacts)
    return types.SimpleNamespace(events=events, training_env=fake_training_vec_env, eval_env=fake_eval_env)


def _manual_curriculum_task(stage_name: str) -> dict[str, object]:
    """Return a copied task from the manual line curriculum fixture."""
    config = experiments_config.load_experiment_config("configs/curricula/curriculum_manual_line_smoke.yaml")
    for stage in config["stages"]:
        if stage["stage_name"] == stage_name:
            return dict(stage["task"])
    message = f"manual curriculum stage not found: {stage_name}"
    raise AssertionError(message)


def test_ppo_tracking_canonical_module_imports() -> None:
    """Verify PPO tracking helpers are exposed by the canonical module."""
    assert ppo_tracking is not None
    assert ppo_tracking.PPOTrackingSmokeSettings is not None


def test_load_ppo_tracking_smoke_config_returns_valid_settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the smoke YAML file loads into validated settings."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = ppo_tracking.load_ppo_tracking_settings("configs/training/ppo_tracking_smoke.yaml")

    assert settings.training_config_path == Path("configs/training/ppo_tracking_smoke.yaml")
    assert settings.task_config_path == Path("configs/training/ppo_tracking_tasks.yaml")
    assert settings.task_index == EXPECTED_SMOKE_TASK_INDEX
    assert settings.task_shape is None
    assert settings.run_name == "direct_ppo_line_smoke_seed0"
    assert settings.total_timesteps == EXPECTED_SMOKE_TIMESTEPS
    assert settings.num_envs == EXPECTED_SMOKE_NUM_ENVS
    assert settings.ppo_config.to_dict() == EXPECTED_PPO_CONFIG
    assert settings.eval_steps == EXPECTED_SMOKE_EVAL_STEPS
    assert settings.seed == 0
    assert settings.output_dir is None
    assert settings.model_dir is None
    assert settings.normalize_actions is True
    assert settings.action_interface == "pid_position"
    assert settings.rpm_delta_scale == DIRECT_RPM_DELTA_SCALE
    assert settings.include_dynamics_observation is False
    assert settings.include_previous_action is False
    assert settings.termination_limits.mode == "default"
    assert settings.termination_limits.terminate_on_base_truncation is True
    assert settings.diagnostic_limits.mode == "default"
    assert settings.wandb_mode == "disabled"
    assert settings.wandb_project == utils.wandb.DEFAULT_WANDB_PROJECT
    assert settings.wandb_entity is None
    assert settings.wandb_group == "direct_ppo/line_smoke"
    assert settings.wandb_name == "direct_ppo_line_smoke_seed0"
    assert settings.wandb_tags == ("smoke", "direct_ppo", "line")
    assert settings.wandb_dir is None
    assert ppo_tracking.default_output_dir() == tmp_path / "runs" / "direct_ppo_line_smoke_seed0"
    assert ppo_tracking.default_model_dir() == tmp_path / "runs" / "direct_ppo_line_smoke_seed0" / "training" / "models"
    assert utils.wandb.default_wandb_dir("direct_ppo_line_smoke_seed0") == tmp_path / "runs" / "direct_ppo_line_smoke_seed0" / "training" / "wandb"


@pytest.mark.parametrize(
    ("config_text", "match"),
    [
        (
            "task_shape: hover\ntotal_timesteps: 12\n",
            "ppo config section is required",
        ),
        (
            "task_shape: hover\ntotal_timesteps: 12\nlearning_rate: 0.0003\nn_steps: 12\nbatch_size: 6\n",
            "top-level PPO keys are not supported",
        ),
        (
            "task_shape: hover\ntotal_timesteps: 12\nlearning_rate: 0.0003\nppo:\n  n_steps: 12\n  batch_size: 6\n",
            "top-level PPO keys are not supported",
        ),
    ],
)
def test_load_ppo_tracking_settings_requires_nested_ppo_config(
    tmp_path: Path,
    config_text: str,
    match: str,
) -> None:
    """Verify YAML training settings reject missing, flat, and mixed PPO config forms."""
    config_path = tmp_path / "ppo_tracking.yaml"
    config_path.write_text(config_text, encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        ppo_tracking.load_ppo_tracking_settings(config_path)


def test_ppo_tracking_settings_reject_invalid_timesteps() -> None:
    """Verify invalid PPO timestep budgets are rejected."""
    with pytest.raises(ValueError, match="total_timesteps must be positive"):
        ppo_tracking.PPOTrackingSmokeSettings(total_timesteps=0)


def test_ppo_tracking_settings_reject_invalid_action_interface() -> None:
    """Verify action interfaces must use the canonical config contract."""
    with pytest.raises(ValueError, match="action_interface must be one of"):
        ppo_tracking.PPOTrackingSmokeSettings(action_interface="rpm")


def test_ppo_tracking_direct_rpm_config_loads() -> None:
    """Verify the direct-RPM smoke config resolves explicit action settings."""
    settings = ppo_tracking.load_ppo_tracking_settings("configs/training/ppo_tracking_direct_rpm_smoke.yaml")

    assert settings.action_interface == "direct_rpm"
    assert settings.normalize_actions is True
    assert settings.rpm_delta_scale == DIRECT_RPM_DELTA_SCALE
    assert settings.include_dynamics_observation is True
    assert settings.include_previous_action is True
    assert settings.termination_limits.mode == "default"
    assert settings.diagnostic_limits.mode == "default"
    assert settings.wandb_mode == "disabled"
    assert settings.num_envs == 1


def test_ppo_tracking_dynamics_smoke_config_loads() -> None:
    """Verify the optional PID dynamics/previous-action smoke config resolves."""
    settings = ppo_tracking.load_ppo_tracking_settings("configs/training/ppo_tracking_dynamics_smoke.yaml")

    assert settings.action_interface == "pid_position"
    assert settings.include_dynamics_observation is True
    assert settings.include_previous_action is True
    assert settings.ppo_config.policy_kwargs == ppo_config.default_policy_kwargs()
    assert settings.wandb_mode == "disabled"
    assert settings.num_envs == 1


def test_ppo_tracking_dynamics_medium_net_smoke_config_loads() -> None:
    """Verify the optional larger-network smoke config resolves policy kwargs."""
    settings = ppo_tracking.load_ppo_tracking_settings("configs/training/ppo_tracking_dynamics_medium_net_smoke.yaml")

    assert settings.action_interface == "pid_position"
    assert settings.include_dynamics_observation is True
    assert settings.include_previous_action is True
    assert settings.ppo_config.policy_kwargs == {"net_arch": POLICY_NET_ARCH}
    assert settings.ppo_config.to_sb3_kwargs()["policy_kwargs"] == {"net_arch": POLICY_NET_ARCH}
    assert settings.wandb_mode == "disabled"


@pytest.mark.parametrize("num_envs", [0, -1, 1.5, True])
def test_ppo_tracking_settings_reject_invalid_num_envs(num_envs: object) -> None:
    """Verify parallel training environment counts must be positive integers."""
    with pytest.raises(ValueError, match="num_envs must be a positive integer"):
        ppo_tracking.PPOTrackingSmokeSettings(num_envs=num_envs)  # type: ignore[arg-type]


def test_ppo_tracking_settings_accept_vectorized_rollout_batch_size() -> None:
    """Verify settings validate batch size against n_steps multiplied by num_envs."""
    config = ppo_config.PPOConfig(n_steps=VECTOR_TEST_N_STEPS, batch_size=VECTOR_TEST_BATCH_SIZE)
    settings = ppo_tracking.PPOTrackingSmokeSettings(
        total_timesteps=8,
        num_envs=CONFIGURED_TEST_NUM_ENVS,
        ppo_config=config,
    )

    assert settings.num_envs == CONFIGURED_TEST_NUM_ENVS
    assert settings.ppo_config.effective_rollout_steps(settings.num_envs) == VECTOR_TEST_EFFECTIVE_ROLLOUT_STEPS


def test_ppo_tracking_settings_reject_batch_size_larger_than_effective_rollout() -> None:
    """Verify settings reject PPO minibatches larger than the vectorized rollout."""
    config = ppo_config.PPOConfig(n_steps=VECTOR_TEST_N_STEPS, batch_size=VECTOR_TEST_OVERSIZED_BATCH_SIZE)

    with pytest.raises(ValueError, match=r"ppo\.batch_size must be less than or equal to ppo\.n_steps \* num_envs"):
        ppo_tracking.PPOTrackingSmokeSettings(total_timesteps=8, num_envs=CONFIGURED_TEST_NUM_ENVS, ppo_config=config)


def test_ppo_tracking_settings_reject_invalid_termination_limits() -> None:
    """Verify invalid termination-limit config values are rejected."""
    with pytest.raises(ValueError, match="allow_recovery_steps"):
        ppo_tracking.PPOTrackingSmokeSettings(termination_limits={"mode": "relaxed", "allow_recovery_steps": -1})


def test_ppo_tracking_settings_reject_invalid_eval_steps() -> None:
    """Verify invalid evaluation lengths are rejected."""
    with pytest.raises(ValueError, match="eval_steps must be positive"):
        ppo_tracking.PPOTrackingSmokeSettings(eval_steps=0)


def test_ppo_tracking_settings_reject_invalid_run_name() -> None:
    """Verify training run names cannot escape storage/runs."""
    with pytest.raises(ValueError, match="run_name"):
        ppo_tracking.PPOTrackingSmokeSettings(run_name="../bad")


def test_ppo_tracking_select_task_by_shape_uses_configured_task() -> None:
    """Verify task-shape selection reuses the configured task list."""
    task, task_source, task_index, warnings = ppo_tracking._select_task(  # noqa: SLF001
        task_config_path=Path("configs/training/ppo_tracking_tasks.yaml"),
        default_task_index=0,
        task_shape="line",
    )

    assert task["shape"] == "line"
    assert task_source == "shape_override"
    assert task_index == LINE_TASK_INDEX
    assert warnings


def test_ppo_tracking_paths_resolve_under_run_directories(tmp_path: Path) -> None:
    """Verify caller-provided run roots control generated artifact paths."""
    settings = ppo_tracking.PPOTrackingSmokeSettings(
        task_shape="hover",
        output_dir=tmp_path / "run",
        model_dir=tmp_path / "run" / "models",
    )

    model_path = ppo_tracking._resolve_model_path(settings)  # noqa: SLF001
    metrics_path = ppo_tracking._resolve_metrics_path(settings)  # noqa: SLF001

    assert model_path == (tmp_path / "run" / "models" / "direct_ppo_hover_seed0.zip").resolve(strict=False)
    assert metrics_path == (tmp_path / "run" / "metrics" / "direct_ppo_hover_seed0_metrics.json").resolve(strict=False)


def test_ppo_tracking_auto_run_name_uses_resolved_task_shape() -> None:
    """Verify omitted run names are derived from task shape and seed."""
    settings = ppo_tracking.PPOTrackingSmokeSettings(task_shape="line", total_timesteps=10000, seed=0)

    assert ppo_tracking._timesteps_label(4096) == "4096"  # noqa: SLF001
    assert ppo_tracking._timesteps_label(10000) == "10k"  # noqa: SLF001
    assert ppo_tracking._timesteps_label(1000000) == "1m"  # noqa: SLF001
    assert ppo_tracking._run_name(settings) == "direct_ppo_line_seed0"  # noqa: SLF001


def test_direct_ppo_wandb_naming_uses_run_name_group_and_identity_tags() -> None:
    """Verify direct PPO W&B identity is derived from resolved run settings."""
    settings = ppo_tracking.load_ppo_tracking_settings("configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml")
    assert settings.task_distribution_settings is not None

    wandb_settings = ppo_tracking._wandb_settings(  # noqa: SLF001
        settings,
        "line",
        settings.task_distribution_settings.to_metadata(),
    )

    assert wandb_settings.name == "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    assert wandb_settings.group == "direct_ppo/pid_position/tracking_medium/pid_dynprev_m-taskdist_medium/seed0"
    assert "direct_ppo" in wandb_settings.tags
    assert "training" in wandb_settings.tags
    assert "curriculum" not in wandb_settings.tags
    assert "action_interface:pid_position" in wandb_settings.tags
    assert "observation:dynamics" in wandb_settings.tags
    assert "observation:previous_action" in wandb_settings.tags
    assert "task_distribution:tracking_medium" in wandb_settings.tags
    assert "net:net128_default" in wandb_settings.tags
    assert "ppo_profile:default" in wandb_settings.tags
    assert "seed:0" in wandb_settings.tags


def test_run_name_override_becomes_default_wandb_name(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify config W&B names do not leak into curriculum stage run overrides."""
    captured: dict[str, object] = {}

    def fake_run(settings: ppo_tracking.PPOTrackingSmokeSettings) -> ppo_tracking.PPOTrackingSmokeResult:
        captured["settings"] = settings
        return ppo_tracking.PPOTrackingSmokeResult(model_path="model.zip", metrics_path="metrics.json", manifest_path="manifest.json", metrics={})

    monkeypatch.setattr(ppo_tracking, "run_ppo_tracking_smoke", fake_run)

    ppo_tracking.run_ppo_tracking_smoke_from_config(
        config_path="configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml",
        run_name="llm_curriculum_pid_dynprev_m-taskdist_medium_stage01_line_seed0",
    )

    settings = captured["settings"]
    assert isinstance(settings, ppo_tracking.PPOTrackingSmokeSettings)
    assert settings.run_name == "llm_curriculum_pid_dynprev_m-taskdist_medium_stage01_line_seed0"
    assert settings.wandb_name == "llm_curriculum_pid_dynprev_m-taskdist_medium_stage01_line_seed0"


def test_ppo_tracking_run_name_controls_default_artifact_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify run-name defaults place training artifacts under canonical storage/runs."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = ppo_tracking.PPOTrackingSmokeSettings(run_name="ppo_line_smoke")

    assert (
        ppo_tracking._resolve_model_path(settings)  # noqa: SLF001
        == tmp_path / "runs" / "ppo_line_smoke" / "training" / "models" / "ppo_line_smoke.zip"
    )
    assert (
        ppo_tracking._resolve_metrics_path(settings)  # noqa: SLF001
        == tmp_path / "runs" / "ppo_line_smoke" / "training" / "metrics" / "ppo_line_smoke_metrics.json"
    )
    assert (
        ppo_tracking._resolve_manifest_path(settings)  # noqa: SLF001
        == tmp_path / "runs" / "ppo_line_smoke" / "training" / "manifest.json"
    )
    assert (
        ppo_tracking._resolve_run_manifest_path(settings, "ppo_line_smoke")  # noqa: SLF001
        == tmp_path / "runs" / "ppo_line_smoke" / "run_manifest.json"
    )
    wandb_settings = ppo_tracking._wandb_settings(settings, "line")  # noqa: SLF001
    assert wandb_settings.dir == tmp_path / "runs" / "ppo_line_smoke" / "training" / "wandb"
    assert wandb_settings.name == "ppo_line_smoke"
    assert wandb_settings.group == "direct_ppo/pid_position/fixed/line/seed0"


def test_ppo_tracking_manifest_includes_failure_diagnostics_paths(tmp_path: Path) -> None:
    """Verify PPO manifests duplicate compact diagnostics summary and artifact paths."""
    settings = ppo_tracking.PPOTrackingSmokeSettings(run_name="ppo_line_smoke")
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
        "termination_limits_mode": "relaxed",
        "termination_limits": {"mode": "relaxed"},
        "diagnostic_limits": {"mode": "default"},
        "base_truncation_policy": "diagnose_only",
        "terminate_on_base_truncation": False,
        "evaluation_termination_limits_mode": "default",
        "strict_limit_violation_count": 2,
        "strict_limit_violation_causes": ["pitch_above_limit"],
    }

    manifest = ppo_tracking._build_manifest(  # noqa: SLF001
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
    assert manifest["termination_limits_mode"] == "relaxed"
    assert manifest["base_truncation_policy"] == "diagnose_only"
    assert manifest["diagnostics"]["strict_limit_violation_count"] == STRICT_TEST_LIMIT_VIOLATIONS
    assert manifest["ppo_config"] == settings.ppo_config.to_dict()


def test_curriculum_stage_manifest_and_tags_include_identity_fields(tmp_path: Path) -> None:
    """Verify curriculum stage manifests and W&B tags expose unambiguous identity."""
    run_name = "curriculum_llm_unit_stage01_line_seed0"
    settings = ppo_tracking.PPOTrackingSmokeSettings(
        run_name=run_name,
        artifact_root=tmp_path / "stage01_line" / "training",
        task_shape="line",
        total_timesteps=32,
        ppo_config=ppo_config.PPOConfig(n_steps=16, batch_size=8),
        run_metadata={
            "run_kind": "curriculum_stage",
            "curriculum_kind": "llm",
            "curriculum_run_name": "curriculum_llm_unit_seed0",
            "curriculum_stage_index": 1,
            "curriculum_stage_name": "line",
            "curriculum_stage_count": 2,
            "curriculum_stage_run_name": run_name,
            "stage_budget_profile": "short",
            "stage_total_timesteps": 32,
            "cumulative_llm_budget_timesteps": 32,
            "llm_budget_cap_timesteps": 64,
            "proposal_fallback_used": False,
            "task_distribution_reference": {"task_distribution_id": "tracking_medium"},
            "resolved_task_shape": "line",
            "llm_provider": "mock",
        },
        wandb_tags=("stage_index:1", "stage:line", "llm_provider:mock", "llm_budget_profile:short", "llm_fallback:false"),
    )
    metrics = {
        "run_type": "training",
        "run_kind": "curriculum_stage",
        "curriculum_kind": "llm",
        "training_run_name": run_name,
        "model_path": str(tmp_path / "models" / f"{run_name}.zip"),
        "metrics_path": str(tmp_path / "metrics" / f"{run_name}_metrics.json"),
        "manifest_path": str(tmp_path / "manifest.json"),
        "logs_dir": str(tmp_path / "logs"),
        "diagnostics_dir": str(tmp_path / "diagnostics"),
        "run_metadata": dict(settings.run_metadata),
        **settings.run_metadata,
    }

    manifest = ppo_tracking._build_manifest(  # noqa: SLF001
        settings=settings,
        metrics=metrics,
        task_source="config",
        selected_task_index=0,
        task={"shape": "line"},
    )
    wandb_settings = ppo_tracking._wandb_settings(settings, "line", {"task_distribution_name": "tracking_medium"})  # noqa: SLF001

    assert manifest["run_kind"] == "curriculum_stage"
    assert manifest["curriculum_kind"] == "llm"
    assert manifest["curriculum_run_name"] == "curriculum_llm_unit_seed0"
    assert manifest["curriculum_stage_index"] == 1
    assert manifest["curriculum_stage_name"] == "line"
    assert manifest["curriculum_stage_run_name"] == run_name
    assert manifest["stage_budget_profile"] == "short"
    assert manifest["llm_provider"] == "mock"
    assert manifest["task_distribution_reference"] == {"task_distribution_id": "tracking_medium"}
    assert wandb_settings.name == run_name
    assert wandb_settings.group == "curriculum/llm/curriculum_llm_unit_seed0"
    for tag in (
        "curriculum",
        "llm",
        "training",
        "stage_index:1",
        "stage:line",
        "action_interface:pid_position",
        "task_distribution:tracking_medium",
        "llm_provider:mock",
        "llm_budget_profile:short",
        "llm_fallback:false",
    ):
        assert tag in wandb_settings.tags


def test_ppo_tracking_diagnostic_artifact_paths_include_feedback_files(tmp_path: Path) -> None:
    """Verify W&B artifact path selection includes final diagnostic files."""
    metrics = {
        "failure_report_path": str(tmp_path / "failure_report.json"),
        "curriculum_feedback_path": str(tmp_path / "curriculum_feedback.json"),
        "episode_summaries_path": str(tmp_path / "episode_summaries.json"),
        "evaluation_trace_path": str(tmp_path / "evaluation_trace.jsonl"),
        "metrics_path": str(tmp_path / "metrics.json"),
    }

    artifact_paths = ppo_tracking._diagnostic_artifact_paths("ppo_line_smoke", metrics)  # noqa: SLF001

    assert artifact_paths == {
        "ppo_line_smoke_failure_report": tmp_path / "failure_report.json",
        "ppo_line_smoke_curriculum_feedback": tmp_path / "curriculum_feedback.json",
        "ppo_line_smoke_episode_summaries": tmp_path / "episode_summaries.json",
        "ppo_line_smoke_evaluation_trace": tmp_path / "evaluation_trace.jsonl",
    }


def test_ppo_tracking_explicit_output_dir_remains_direct(tmp_path: Path) -> None:
    """Verify explicit output directory overrides preserve direct metrics placement."""
    settings = ppo_tracking.PPOTrackingSmokeSettings(
        task_shape="hover",
        output_dir=tmp_path / "storage" / "results" / "custom",
    )

    metrics_path = ppo_tracking._resolve_metrics_path(settings)  # noqa: SLF001

    assert metrics_path == (tmp_path / "storage" / "results" / "custom" / "direct_ppo_hover_seed0_metrics.json").resolve(strict=False)


def test_ppo_tracking_dependency_detection_returns_booleans() -> None:
    """Verify dependency detection reports simple boolean availability flags."""
    dependencies = ppo_tracking.detect_ppo_tracking_dependencies()

    assert isinstance(dependencies["stable_baselines3"], bool)
    assert isinstance(dependencies["gymnasium"], bool)
    assert isinstance(dependencies["torch"], bool)


def test_ppo_runtime_info_reports_cuda_availability() -> None:
    """Verify runtime diagnostics expose torch/CUDA information without requiring GPU."""
    runtime = ppo_tracking.detect_ppo_runtime_info()

    assert isinstance(runtime["torch_available"], bool)
    assert isinstance(runtime["torch_cuda_available"], bool)
    assert isinstance(runtime["torch_cuda_device_count"], int)


def test_tracking_action_metadata_reports_pid_contract() -> None:
    """Verify action metadata captures the upstream PID action contract."""
    task = ppo_tracking._load_task(  # noqa: SLF001
        ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        ppo_tracking.DEFAULT_TASK_INDEX,
    )
    metadata = ppo_tracking.describe_tracking_env_action_metadata(task)

    assert metadata["action_space_shape"] == [1, 3]
    assert metadata["action_space_low"] == [[-1.0, -1.0, -1.0]]
    assert metadata["action_space_high"] == [[1.0, 1.0, 1.0]]
    assert metadata["actions_normalized"] is True
    assert metadata["action_interface"] == "pid_position"
    assert metadata["ppo_action_dim"] == PID_ACTION_DIM
    assert metadata["real_action_type"] == "pid_target_position"
    assert metadata["include_dynamics_observation"] is False
    assert metadata["include_previous_action"] is False
    assert metadata["observation_dim"] == BASE_OBSERVATION_DIM
    assert [component["name"] for component in metadata["observation_components"]] == [
        "current_position",
        "reference_position",
        "position_error",
        "trajectory_progress",
    ]
    assert metadata["real_action_space_low"] != metadata["action_space_low"]
    assert metadata["base_action_type"] == "pid"
    assert "x/y/z movement" in metadata["base_action_semantics"]


def test_tracking_action_metadata_reports_direct_rpm_contract() -> None:
    """Verify direct-RPM metadata captures motor-level action details."""
    task = ppo_tracking._load_task(  # noqa: SLF001
        ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        ppo_tracking.DEFAULT_TASK_INDEX,
    )
    metadata = ppo_tracking.describe_tracking_env_action_metadata(
        task,
        action_interface="direct_rpm",
        include_dynamics_observation=True,
        include_previous_action=True,
    )

    assert metadata["action_space_shape"] == [1, 4]
    assert metadata["action_space_low"] == [[-1.0, -1.0, -1.0, -1.0]]
    assert metadata["action_space_high"] == [[1.0, 1.0, 1.0, 1.0]]
    assert metadata["actions_normalized"] is True
    assert metadata["action_interface"] == "direct_rpm"
    assert metadata["ppo_action_dim"] == DIRECT_RPM_ACTION_DIM
    assert metadata["real_action_type"] == "motor_rpm"
    assert metadata["base_action_type"] == "rpm"
    assert metadata["hover_rpm"] > 0.0
    assert metadata["rpm_delta_scale"] == DIRECT_RPM_DELTA_SCALE
    assert metadata["real_action_space_bounds"]["units"] == "rpm"
    assert metadata["include_dynamics_observation"] is True
    assert metadata["include_previous_action"] is True
    assert metadata["observation_dim"] == DIRECT_RPM_DYNAMICS_PREVIOUS_OBSERVATION_DIM
    assert metadata["observation_components"][-1] == {"name": "previous_action", "dim": DIRECT_RPM_ACTION_DIM}
    assert metadata["direct_control_limitations"]


def test_normalized_action_wrapper_maps_to_real_pid_bounds() -> None:
    """Verify normalized PPO actions map explicitly to real tracking action bounds."""
    task = ppo_tracking._load_task(  # noqa: SLF001
        ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        ppo_tracking.DEFAULT_TASK_INDEX,
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


def test_ppo_training_env_uses_direct_rpm_without_pid_normalization_wrapper() -> None:
    """Verify direct RPM is already normalized and does not use the PID wrapper."""
    task = ppo_tracking._load_task(  # noqa: SLF001
        ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        ppo_tracking.DEFAULT_TASK_INDEX,
    )
    training_env = ppo_tracking._make_seeded_ppo_tracking_env(  # noqa: SLF001
        task=task,
        normalize_actions=True,
        seed=0,
        action_interface="direct_rpm",
        include_dynamics_observation=True,
        include_previous_action=True,
    )
    try:
        assert training_env.action_interface == "direct_rpm"
        assert not hasattr(training_env, "real_action_space")
        assert training_env.action_space.shape == (1, 4)
        assert np.allclose(training_env.action_space.low, -1.0)
        assert np.allclose(training_env.action_space.high, 1.0)
        assert training_env.base_env.ACT_TYPE.value == "rpm"
    finally:
        training_env.close()


def test_ppo_training_env_uses_normalized_action_space_when_enabled() -> None:
    """Verify PPO receives the normalized action wrapper when configured."""
    task = ppo_tracking._load_task(  # noqa: SLF001
        ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        ppo_tracking.DEFAULT_TASK_INDEX,
    )
    real_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False, max_steps=DIAGNOSTIC_STEPS)
    training_env = ppo_tracking._ppo_training_env(real_env, normalize_actions=True)  # noqa: SLF001
    try:
        assert training_env.action_space.shape == (1, 3)
        assert np.allclose(training_env.action_space.low, -1.0)
        assert np.allclose(training_env.action_space.high, 1.0)
        assert training_env.real_action_space is real_env.action_space
    finally:
        training_env.close()


def test_ppo_training_vec_env_uses_dummy_vec_env_for_single_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify num_envs=1 uses the same vectorized construction path with DummyVecEnv."""

    class FakeDummyVecEnv:
        """Record DummyVecEnv construction without building child envs eagerly."""

        def __init__(self, env_fns: list[object]) -> None:
            """Store lazy env factories."""
            self.env_fns = list(env_fns)
            self.seed_calls: list[int] = []

        def seed(self, seed: int) -> None:
            """Record the base vector seed."""
            self.seed_calls.append(seed)

    class FakeSubprocVecEnv:
        """Unused stand-in for the subprocess vector env."""

    class FakeVecMonitor:
        """Record VecMonitor wrapping without depending on SB3 internals."""

        def __init__(self, venv: object, filename: str | None = None) -> None:
            """Store the wrapped VecEnv and optional monitor filename."""
            self.venv = venv
            self.filename = filename

    factory_seeds: list[int] = []
    constructed_seeds: list[int] = []

    def fake_factory(
        task: dict[str, Any],
        normalize_actions: bool,
        seed: int,
        action_interface: str = "pid_position",
        rpm_delta_scale: float = 0.05,
        include_dynamics_observation: bool = False,
        include_previous_action: bool = False,
        termination_limits: object | None = None,
        diagnostic_limits: object | None = None,
    ) -> object:
        """Return a lazy factory while recording derived seeds."""
        assert task == {"shape": "line"}
        assert normalize_actions is True
        assert action_interface == "direct_rpm"
        assert rpm_delta_scale == DIRECT_RPM_DELTA_SCALE
        assert include_dynamics_observation is True
        assert include_previous_action is True
        assert termination_limits is None
        assert diagnostic_limits is None
        factory_seeds.append(seed)

        def make_env() -> object:
            constructed_seeds.append(seed)
            return object()

        return make_env

    monkeypatch.setattr(ppo_tracking, "_vec_env_classes", lambda: (FakeDummyVecEnv, FakeSubprocVecEnv))
    monkeypatch.setattr(ppo_tracking, "_vec_monitor_class", lambda: FakeVecMonitor)
    monkeypatch.setattr(ppo_tracking, "_make_ppo_training_env_factory", fake_factory)

    vec_env = ppo_tracking._make_ppo_training_vec_env(  # noqa: SLF001
        task={"shape": "line"},
        num_envs=1,
        normalize_actions=True,
        seed=7,
        action_interface="direct_rpm",
        rpm_delta_scale=DIRECT_RPM_DELTA_SCALE,
        include_dynamics_observation=True,
        include_previous_action=True,
    )

    assert isinstance(vec_env, FakeVecMonitor)
    assert vec_env.filename is None
    assert isinstance(vec_env.venv, FakeDummyVecEnv)
    base_vec_env = vec_env.venv
    assert ppo_tracking._vec_env_type(1) == "DummyVecEnv"  # noqa: SLF001
    assert factory_seeds == [7]
    assert constructed_seeds == []
    assert len(base_vec_env.env_fns) == 1
    assert base_vec_env.seed_calls == [7]


def test_ppo_training_vec_env_uses_subproc_vec_env_lazily(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify num_envs>1 uses SubprocVecEnv with lazy per-rank env factories."""

    class FakeDummyVecEnv:
        """Unused stand-in for the single-env vector env."""

    class FakeSubprocVecEnv:
        """Record SubprocVecEnv construction without calling env factories."""

        def __init__(self, env_fns: list[object], start_method: str) -> None:
            """Store factories and multiprocessing start method."""
            self.env_fns = list(env_fns)
            self.start_method = start_method
            self.seed_calls: list[int] = []

        def seed(self, seed: int) -> None:
            """Record the base vector seed."""
            self.seed_calls.append(seed)

    class FakeVecMonitor:
        """Record VecMonitor wrapping without depending on SB3 internals."""

        def __init__(self, venv: object, filename: str | None = None) -> None:
            """Store the wrapped VecEnv and optional monitor filename."""
            self.venv = venv
            self.filename = filename

    factory_seeds: list[int] = []
    constructed_seeds: list[int] = []

    def fake_factory(
        task: dict[str, Any],
        normalize_actions: bool,
        seed: int,
        action_interface: str = "pid_position",
        rpm_delta_scale: float = 0.05,
        include_dynamics_observation: bool = False,
        include_previous_action: bool = False,
        termination_limits: object | None = None,
        diagnostic_limits: object | None = None,
    ) -> object:
        """Return a lazy factory while recording derived seeds."""
        assert task == {"shape": "line"}
        assert normalize_actions is False
        assert action_interface == "pid_position"
        assert rpm_delta_scale == DIRECT_RPM_DELTA_SCALE
        assert include_dynamics_observation is False
        assert include_previous_action is False
        assert termination_limits is None
        assert diagnostic_limits is None
        factory_seeds.append(seed)

        def make_env() -> object:
            constructed_seeds.append(seed)
            return object()

        return make_env

    monkeypatch.setattr(ppo_tracking, "_vec_env_classes", lambda: (FakeDummyVecEnv, FakeSubprocVecEnv))
    monkeypatch.setattr(ppo_tracking, "_vec_monitor_class", lambda: FakeVecMonitor)
    monkeypatch.setattr(ppo_tracking, "_make_ppo_training_env_factory", fake_factory)

    vec_env = ppo_tracking._make_ppo_training_vec_env(  # noqa: SLF001
        task={"shape": "line"},
        num_envs=SUBPROC_TEST_NUM_ENVS,
        normalize_actions=False,
        seed=SUBPROC_TEST_SEED,
    )

    assert isinstance(vec_env, FakeVecMonitor)
    assert vec_env.filename is None
    assert isinstance(vec_env.venv, FakeSubprocVecEnv)
    base_vec_env = vec_env.venv
    assert ppo_tracking._vec_env_type(3) == "SubprocVecEnv"  # noqa: SLF001
    assert base_vec_env.start_method == "spawn"
    assert factory_seeds == [11, 12, 13]
    assert constructed_seeds == []
    assert len(base_vec_env.env_fns) == SUBPROC_TEST_NUM_ENVS
    assert base_vec_env.seed_calls == [11]


def test_task_with_minimum_reference_samples_extends_hold_move_task_without_raising() -> None:
    """Verify start-hold curriculum diagnostics extend duration instead of failing."""
    task = _manual_curriculum_task("start_hold_then_short_line")

    diagnostic_task, reference_samples, warnings = ppo_tracking._task_with_minimum_reference_samples(  # noqa: SLF001
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

    diagnostic_task, reference_samples, warnings = ppo_tracking._task_with_minimum_reference_samples(  # noqa: SLF001
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

    diagnostic_task, reference_samples, warnings = ppo_tracking._task_with_minimum_reference_samples(  # noqa: SLF001
        task,
        required_steps=CURRICULUM_DIAGNOSTIC_STEPS,
    )

    assert diagnostic_task == task
    assert reference_samples == original_reference.positions.shape[0]
    assert any("duration extension failed validation" in warning for warning in warnings)
    assert any("using original diagnostic task" in warning for warning in warnings)


def test_liftoff_diagnostics_report_simple_policy_bounds() -> None:
    """Verify liftoff diagnostics include structured movement summaries."""
    task = ppo_tracking._load_task(  # noqa: SLF001
        ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        ppo_tracking.DEFAULT_TASK_INDEX,
    )
    diagnostics = ppo_tracking.run_liftoff_diagnostics(task, max_steps=DIAGNOSTIC_STEPS, seed=0)

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

    task = ppo_tracking._load_task(  # noqa: SLF001
        ppo_tracking.DEFAULT_TASK_CONFIG_PATH,
        ppo_tracking.DEFAULT_TASK_INDEX,
    )
    real_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False, max_steps=DIAGNOSTIC_STEPS)
    tracking_env = envs.tracking_env.make_normalized_action_env(real_env)
    try:
        settings = ppo_tracking.PPOTrackingSmokeSettings(eval_steps=DIAGNOSTIC_STEPS)
        metrics = ppo_tracking._evaluate_model(HighActionModel(), tracking_env, settings)  # noqa: SLF001
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


def test_run_ppo_tracking_smoke_finishes_wandb_run_on_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify successful PPO training logs W&B outputs and finishes exactly once."""
    fake_wandb_run = _FakeWandbRun()
    harness = _install_fast_ppo_smoke_fakes(tmp_path, monkeypatch, wandb_run=fake_wandb_run)
    settings = ppo_tracking.PPOTrackingSmokeSettings(
        run_name="wandb_success",
        total_timesteps=CONFIGURED_TEST_PPO_N_STEPS,
        ppo_config=ppo_config.PPOConfig(n_steps=CONFIGURED_TEST_PPO_N_STEPS, batch_size=CONFIGURED_TEST_PPO_BATCH_SIZE),
        eval_steps=4,
        check_env=False,
        wandb_mode=utils.wandb.WANDB_MODE_OFFLINE,
    )

    result = ppo_tracking.run_ppo_tracking_smoke(settings)

    assert result.metrics["wandb"]["enabled"] is True
    assert result.metrics["wandb"]["run_id"] == "fake-run-id"
    assert harness.events == ["start", "summary", "artifacts"]
    assert fake_wandb_run.finish_count == 1
    assert harness.training_env.closed is True
    assert harness.eval_env.closed is True


def test_run_ppo_tracking_smoke_finishes_wandb_run_on_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify W&B is finished and the original post-start exception propagates."""
    fake_wandb_run = _FakeWandbRun()
    learn_error = RuntimeError("training failed after wandb start")
    harness = _install_fast_ppo_smoke_fakes(
        tmp_path,
        monkeypatch,
        wandb_run=fake_wandb_run,
        learn_error=learn_error,
    )
    settings = ppo_tracking.PPOTrackingSmokeSettings(
        run_name="wandb_failure",
        total_timesteps=CONFIGURED_TEST_PPO_N_STEPS,
        ppo_config=ppo_config.PPOConfig(n_steps=CONFIGURED_TEST_PPO_N_STEPS, batch_size=CONFIGURED_TEST_PPO_BATCH_SIZE),
        eval_steps=4,
        check_env=False,
        wandb_mode=utils.wandb.WANDB_MODE_OFFLINE,
    )

    with pytest.raises(RuntimeError, match="training failed after wandb start") as exc_info:
        ppo_tracking.run_ppo_tracking_smoke(settings)

    assert exc_info.value is learn_error
    assert harness.events == ["start"]
    assert fake_wandb_run.finish_count == 1
    assert harness.training_env.closed is True
    assert harness.eval_env.closed is False


def test_run_ppo_tracking_smoke_passes_resolved_ppo_config(  # noqa: PLR0915
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify PPO, metrics, manifests, and W&B all receive resolved PPO config."""

    class FakeTrainingEnv:
        """Minimal environment stand-in for the PPO training and eval paths."""

        def __init__(self) -> None:
            """Track whether the fake environment was closed."""
            self.closed = False

        def close(self) -> None:
            """Close the fake environment."""
            self.closed = True

    class FakePPO:
        """Tiny SB3 PPO stand-in that records constructor kwargs."""

        init_calls: ClassVar[list[dict[str, Any]]] = []

        def __init__(self, policy: str, env: FakeTrainingEnv, **kwargs: Any) -> None:
            """Record PPO constructor inputs."""
            self.policy = policy
            self.env = env
            self.kwargs = dict(kwargs)
            self.device = kwargs["device"]
            FakePPO.init_calls.append({"policy": policy, "env": env, "kwargs": dict(kwargs)})

        def learn(self, **kwargs: Any) -> None:
            """Record learn kwargs without training."""
            self.learn_kwargs = dict(kwargs)

        def save(self, path: str) -> None:
            """Write a tiny model marker file."""
            Path(path).write_text("fake model", encoding="utf-8")

    configured_ppo = ppo_config.PPOConfig(
        policy="MlpPolicy",
        device="cpu",
        learning_rate=0.0007,
        gamma=0.91,
        gae_lambda=0.82,
        n_steps=CONFIGURED_TEST_PPO_N_STEPS,
        batch_size=CONFIGURED_TEST_PPO_BATCH_SIZE,
        n_epochs=3,
        clip_range=0.17,
        ent_coef=0.004,
        vf_coef=0.42,
        max_grad_norm=0.9,
        target_kl=0.07,
        policy_kwargs={"net_arch": POLICY_NET_ARCH},
    )
    captured_wandb_config: dict[str, Any] = {}
    captured_eval: dict[str, Any] = {}
    fake_training_vec_env = FakeTrainingEnv()
    fake_eval_env = FakeTrainingEnv()
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    fake_sb3 = types.ModuleType("stable_baselines3")
    fake_sb3.PPO = FakePPO
    monkeypatch.setitem(sys.modules, "stable_baselines3", fake_sb3)
    monkeypatch.setattr(
        ppo_tracking,
        "detect_ppo_tracking_dependencies",
        lambda: {"stable_baselines3": True, "gymnasium": True, "gym_pybullet_drones": True, "torch": True},
    )
    monkeypatch.setattr(ppo_tracking, "detect_ppo_runtime_info", lambda: {"torch_available": True})
    monkeypatch.setattr(
        ppo_tracking,
        "_select_task",
        lambda **_kwargs: ({"shape": "line"}, "config", 0, ()),
    )
    monkeypatch.setattr(
        ppo_tracking,
        "run_liftoff_diagnostics",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        ppo_tracking,
        "_make_ppo_training_vec_env",
        lambda **_kwargs: fake_training_vec_env,
    )
    monkeypatch.setattr(
        ppo_tracking,
        "_make_seeded_ppo_tracking_env",
        lambda **_kwargs: fake_eval_env,
    )
    monkeypatch.setattr(
        ppo_tracking.envs.tracking_env,
        "make_trajectory_tracking_env",
        lambda *_args, **_kwargs: FakeTrainingEnv(),
    )
    monkeypatch.setattr(
        ppo_tracking,
        "_tracking_env_action_metadata",
        lambda _env: {
            "actions_normalized": False,
            "action_interface": "pid_position",
            "ppo_action_dim": PID_ACTION_DIM,
            "real_action_type": "pid_target_position",
            "real_action_space_bounds": {"low": [[-1.0, -1.0, -1.0]], "high": [[1.0, 1.0, 1.0]], "units": "meters"},
            "include_dynamics_observation": False,
            "include_previous_action": False,
            "observation_dim": BASE_OBSERVATION_DIM,
            "observation_components": [
                {"name": "current_position", "dim": 3},
                {"name": "reference_position", "dim": 3},
                {"name": "position_error", "dim": 3},
                {"name": "trajectory_progress", "dim": 1},
            ],
            "direct_control_limitations": [],
        },
    )
    monkeypatch.setattr(ppo_tracking, "_movement_warnings", lambda **_kwargs: ())

    def fake_collect_policy_evaluation_diagnostics(**kwargs: Any) -> object:
        """Capture the eval env used for deterministic diagnostics."""
        captured_eval.update(kwargs)
        return types.SimpleNamespace(metrics={"mean_position_error_m": 0.1})

    monkeypatch.setattr(
        ppo_tracking.evaluation.diagnostics,
        "collect_policy_evaluation_diagnostics",
        fake_collect_policy_evaluation_diagnostics,
    )
    monkeypatch.setattr(
        ppo_tracking.evaluation.diagnostics,
        "write_policy_evaluation_diagnostics",
        lambda *_args, **_kwargs: {},
    )

    def fake_start_wandb_run(*, settings: Any, config: dict[str, Any]) -> None:
        """Capture W&B config without creating a run."""
        _ = settings
        captured_wandb_config.update(config)

    monkeypatch.setattr(ppo_tracking.utils.wandb, "start_wandb_run", fake_start_wandb_run)
    settings = ppo_tracking.PPOTrackingSmokeSettings(
        run_name="configured_ppo",
        total_timesteps=20,
        num_envs=CONFIGURED_TEST_NUM_ENVS,
        ppo_config=configured_ppo,
        eval_steps=4,
        check_env=False,
        normalize_actions=False,
        wandb_mode=utils.wandb.WANDB_MODE_DISABLED,
    )

    result = ppo_tracking.run_ppo_tracking_smoke(settings)

    assert len(FakePPO.init_calls) == 1
    constructor_call = FakePPO.init_calls[0]
    constructor_kwargs = constructor_call["kwargs"]
    assert constructor_call["policy"] == configured_ppo.policy
    for key, value in configured_ppo.to_dict().items():
        if key != "policy":
            assert constructor_kwargs[key] == value
    assert constructor_kwargs["n_steps"] == CONFIGURED_TEST_PPO_N_STEPS
    assert constructor_kwargs["batch_size"] == CONFIGURED_TEST_PPO_BATCH_SIZE
    assert constructor_kwargs["policy_kwargs"] == {"net_arch": POLICY_NET_ARCH}
    assert result.metrics["ppo_config"] == configured_ppo.to_dict()
    assert result.metrics["policy_kwargs"] == {"net_arch": POLICY_NET_ARCH}
    assert result.metrics["num_envs"] == CONFIGURED_TEST_NUM_ENVS
    assert result.metrics["action_interface"] == "pid_position"
    assert result.metrics["ppo_action_dim"] == PID_ACTION_DIM
    assert result.metrics["real_action_type"] == "pid_target_position"
    assert result.metrics["include_dynamics_observation"] is False
    assert result.metrics["include_previous_action"] is False
    assert result.metrics["observation_dim"] == BASE_OBSERVATION_DIM
    assert result.metrics["observation_components"][-1] == {"name": "trajectory_progress", "dim": 1}
    assert result.metrics["direct_control_limitations"] == []
    assert result.metrics["vec_env_type"] == "SubprocVecEnv"
    assert result.metrics["vec_monitor_enabled"] is True
    assert result.metrics["effective_rollout_steps"] == CONFIGURED_TEST_PPO_N_STEPS * CONFIGURED_TEST_NUM_ENVS
    assert result.metrics["wandb"]["enabled"] is False
    assert captured_eval["tracking_env"] is fake_eval_env
    assert captured_eval["tracking_env"] is not fake_training_vec_env
    assert fake_training_vec_env.closed is True
    assert fake_eval_env.closed is True
    assert result.metrics["run_kind"] == "direct_ppo"
    assert result.metrics["curriculum_kind"] is None
    assert "artifact_layout" not in result.metrics
    assert Path(result.model_path) == tmp_path / "runs" / "configured_ppo" / "training" / "models" / "configured_ppo.zip"
    assert Path(result.last_model_path or "") == Path(result.model_path)
    assert result.best_model_path is None
    assert result.best_model_metric is None
    assert result.best_model_step is None
    assert result.best_model_source == "not_selected_no_eval_callback"
    assert result.metrics["last_model_path"] == result.model_path
    assert result.metrics["best_model_path"] is None
    assert result.metrics["best_model_source"] == "not_selected_no_eval_callback"
    assert Path(result.metrics_path) == tmp_path / "runs" / "configured_ppo" / "training" / "metrics" / "configured_ppo_metrics.json"
    assert Path(result.manifest_path) == tmp_path / "runs" / "configured_ppo" / "training" / "manifest.json"
    assert Path(result.metrics["run_manifest_path"]) == tmp_path / "runs" / "configured_ppo" / "run_manifest.json"
    assert captured_wandb_config["ppo"] == configured_ppo.to_dict()
    assert captured_wandb_config["num_envs"] == CONFIGURED_TEST_NUM_ENVS
    assert captured_wandb_config["action_interface"] == "pid_position"
    assert captured_wandb_config["ppo_action_dim"] == PID_ACTION_DIM
    assert captured_wandb_config["real_action_type"] == "pid_target_position"
    assert captured_wandb_config["include_dynamics_observation"] is False
    assert captured_wandb_config["include_previous_action"] is False
    assert captured_wandb_config["observation_dim"] == BASE_OBSERVATION_DIM
    assert captured_wandb_config["policy_kwargs"] == {"net_arch": POLICY_NET_ARCH}
    assert captured_wandb_config["vec_env_type"] == "SubprocVecEnv"
    assert captured_wandb_config["vec_monitor_enabled"] is True
    assert captured_wandb_config["effective_rollout_steps"] == CONFIGURED_TEST_PPO_N_STEPS * CONFIGURED_TEST_NUM_ENVS
    assert "artifact_layout" not in captured_wandb_config
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    assert manifest["ppo_config"] == configured_ppo.to_dict()
    assert manifest["num_envs"] == CONFIGURED_TEST_NUM_ENVS
    assert manifest["action_interface"] == "pid_position"
    assert manifest["ppo_action_dim"] == PID_ACTION_DIM
    assert manifest["real_action_type"] == "pid_target_position"
    assert manifest["include_dynamics_observation"] is False
    assert manifest["include_previous_action"] is False
    assert manifest["observation_dim"] == BASE_OBSERVATION_DIM
    assert manifest["policy_kwargs"] == {"net_arch": POLICY_NET_ARCH}
    assert manifest["vec_env_type"] == "SubprocVecEnv"
    assert manifest["vec_monitor_enabled"] is True
    assert manifest["effective_rollout_steps"] == CONFIGURED_TEST_PPO_N_STEPS * CONFIGURED_TEST_NUM_ENVS
    assert manifest["run_kind"] == "direct_ppo"
    assert manifest["curriculum_kind"] is None
    assert manifest["last_model_path"] == result.model_path
    assert manifest["last_model_path_relative"] == "training/models/configured_ppo.zip"
    assert manifest["best_model_path"] is None
    assert manifest["best_model_path_relative"] is None
    assert manifest["best_model_metric"] is None
    assert manifest["best_model_step"] is None
    assert manifest["best_model_source"] == "not_selected_no_eval_callback"
    assert "artifact_layout" not in manifest
    run_root = tmp_path / "runs" / "configured_ppo"
    training_snapshot = run_root / "config" / "training_config.yaml"
    task_snapshot = run_root / "config" / "task_config.yaml"
    assert (run_root / "config").is_dir()
    assert (run_root / "config" / "evaluation_suites").is_dir()
    assert training_snapshot.exists()
    assert task_snapshot.exists()
    assert result.metrics["training_config_snapshot_path"] == str(training_snapshot)
    assert result.metrics["training_config_snapshot_path_relative"] == "config/training_config.yaml"
    assert result.metrics["task_config_snapshot_path"] == str(task_snapshot)
    assert result.metrics["task_config_snapshot_path_relative"] == "config/task_config.yaml"
    assert (run_root / "training" / "wandb").is_dir()
    assert manifest["model_path_relative"] == "training/models/configured_ppo.zip"
    assert manifest["metrics_path_relative"] == "training/metrics/configured_ppo_metrics.json"
    assert manifest["manifest_path_relative"] == "training/manifest.json"
    assert manifest["training_config_snapshot_path_relative"] == "config/training_config.yaml"
    assert manifest["task_config_snapshot_path_relative"] == "config/task_config.yaml"
    run_manifest = json.loads(Path(result.metrics["run_manifest_path"]).read_text(encoding="utf-8"))
    assert run_manifest["run_kind"] == "direct_ppo"
    assert run_manifest["curriculum_kind"] is None
    assert run_manifest["num_envs"] == CONFIGURED_TEST_NUM_ENVS
    assert run_manifest["action_interface"] == "pid_position"
    assert run_manifest["ppo_action_dim"] == PID_ACTION_DIM
    assert run_manifest["real_action_type"] == "pid_target_position"
    assert run_manifest["include_dynamics_observation"] is False
    assert run_manifest["include_previous_action"] is False
    assert run_manifest["observation_dim"] == BASE_OBSERVATION_DIM
    assert run_manifest["policy_kwargs"] == {"net_arch": POLICY_NET_ARCH}
    assert run_manifest["vec_env_type"] == "SubprocVecEnv"
    assert run_manifest["vec_monitor_enabled"] is True
    assert run_manifest["effective_rollout_steps"] == CONFIGURED_TEST_PPO_N_STEPS * CONFIGURED_TEST_NUM_ENVS
    assert run_manifest["config"]["num_envs"] == CONFIGURED_TEST_NUM_ENVS
    assert run_manifest["config"]["action_interface"] == "pid_position"
    assert run_manifest["config"]["ppo_action_dim"] == PID_ACTION_DIM
    assert run_manifest["config"]["real_action_type"] == "pid_target_position"
    assert run_manifest["config"]["include_dynamics_observation"] is False
    assert run_manifest["config"]["include_previous_action"] is False
    assert run_manifest["config"]["observation_dim"] == BASE_OBSERVATION_DIM
    assert run_manifest["config"]["policy_kwargs"] == {"net_arch": POLICY_NET_ARCH}
    assert run_manifest["config"]["vec_env_type"] == "SubprocVecEnv"
    assert run_manifest["config"]["vec_monitor_enabled"] is True
    assert run_manifest["config"]["effective_rollout_steps"] == CONFIGURED_TEST_PPO_N_STEPS * CONFIGURED_TEST_NUM_ENVS
    assert "artifact_layout" not in run_manifest
    assert run_manifest["training"]["manifest_path"] == result.manifest_path
    assert run_manifest["last_model_path"] == result.model_path
    assert run_manifest["last_model_path_relative"] == "training/models/configured_ppo.zip"
    assert run_manifest["best_model_path"] is None
    assert run_manifest["best_model_source"] == "not_selected_no_eval_callback"
    assert run_manifest["training"]["manifest_path_relative"] == "training/manifest.json"
    assert run_manifest["training"]["model_path_relative"] == "training/models/configured_ppo.zip"
    assert run_manifest["training"]["last_model_path"] == result.model_path
    assert run_manifest["training"]["last_model_path_relative"] == "training/models/configured_ppo.zip"
    assert run_manifest["training"]["best_model_path"] is None
    assert run_manifest["training"]["best_model_source"] == "not_selected_no_eval_callback"
    assert run_manifest["training"]["metrics_path"] == result.metrics_path
    assert run_manifest["training"]["metrics_path_relative"] == "training/metrics/configured_ppo_metrics.json"
    assert run_manifest["config"]["training_config_snapshot_path"] == str(training_snapshot)
    assert run_manifest["config"]["training_config_snapshot_path_relative"] == "config/training_config.yaml"
    assert run_manifest["config"]["task_config_snapshot_path"] == str(task_snapshot)
    assert run_manifest["config"]["task_config_snapshot_path_relative"] == "config/task_config.yaml"
    assert run_manifest["evaluation_index"]["path_relative"] == "evaluation_index.json"
    assert run_manifest["evaluation_index"]["entry_count"] == 0


def test_cli_train_tracking_parser_accepts_task_shape_and_run_name() -> None:
    """Verify the training parser exposes task-specific run controls."""
    parser = cli_train_tracking.build_parser()
    args = parser.parse_args(
        [
            "--task-shape",
            "line",
            "--run-name",
            "ppo_line_smoke",
            "--num-envs",
            str(CLI_NUM_ENVS_OVERRIDE),
            "--action-interface",
            "direct_rpm",
        ]
    )

    assert args.task_shape == "line"
    assert args.run_name == "ppo_line_smoke"
    assert args.num_envs == CLI_NUM_ENVS_OVERRIDE
    assert args.action_interface == "direct_rpm"


def test_cli_train_tracking_passes_num_envs_and_action_interface_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the CLI forwards --num-envs and --action-interface to the training helper."""
    captured: dict[str, Any] = {}

    def fake_run_from_config(**kwargs: Any) -> ppo_tracking.PPOTrackingSmokeResult:
        """Capture CLI overrides without running PPO."""
        captured.update(kwargs)
        return ppo_tracking.PPOTrackingSmokeResult(
            model_path="model.zip",
            metrics_path="metrics.json",
            manifest_path="manifest.json",
            metrics={"num_envs": kwargs["num_envs"]},
        )

    monkeypatch.setattr(cli_train_tracking.ppo_tracking, "run_ppo_tracking_smoke_from_config", fake_run_from_config)

    status = cli_train_tracking.main(
        [
            "--config",
            "configs/training/ppo_tracking_smoke.yaml",
            "--num-envs",
            str(CLI_NUM_ENVS_OVERRIDE),
            "--action-interface",
            "direct_rpm",
        ]
    )

    assert status == 0
    assert captured["num_envs"] == CLI_NUM_ENVS_OVERRIDE
    assert captured["action_interface"] == "direct_rpm"


def test_cli_train_tracking_help_works() -> None:
    """Verify the PPO tracking CLI exposes help without running training."""
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.cli.experiments_cli_train_tracking", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--task-shape" in completed.stdout
    assert "--run-name" in completed.stdout
    assert "--total-timesteps" in completed.stdout
    assert "--num-envs" in completed.stdout
    assert "--action-interface" in completed.stdout
    assert "--eval-steps" in completed.stdout
    assert "--artifact-layout" not in completed.stdout
    assert "--wandb-mode" in completed.stdout
    assert "storage/runs" in completed.stdout
