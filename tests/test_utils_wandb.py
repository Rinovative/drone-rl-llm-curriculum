"""Tests for optional W&B tracking utilities."""

# ruff: noqa: S101

from __future__ import annotations

import builtins
from typing import TYPE_CHECKING, Any

import pytest

from src import utils

if TYPE_CHECKING:
    from pathlib import Path

MEAN_POSITION_ERROR_M = 0.4
MEAN_ABS_Z_ERROR_M = 0.3
ACTION_MEAN_0 = 0.1
REAL_ACTION_SATURATION_0 = 0.5
REAL_ACTION_MEAN_2 = 0.75
EVAL_STEPS = 120


def test_wandb_defaults_are_auto_and_training_scoped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify W&B defaults are auto mode and scoped under the training run."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = utils.wandb.WandbTrackingSettings()

    assert settings.mode == "auto"
    assert settings.project == "drone-rl-llm-curriculum"
    assert utils.wandb.default_wandb_dir("ppo_hover_4096_seed0") == tmp_path / "training_runs" / "ppo_hover_4096_seed0" / "wandb"


def test_disabled_wandb_does_not_import_wandb(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify disabled mode is a no-op even when wandb cannot be imported."""
    original_import = builtins.__import__

    def fail_wandb_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "wandb":
            message = "wandb should not be imported when disabled"
            raise AssertionError(message)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fail_wandb_import)

    run = utils.wandb.start_wandb_run(
        utils.wandb.WandbTrackingSettings(mode="disabled"),
        config={"total_timesteps": 1},
    )

    assert run is None


def test_wandb_tags_parse_comma_separated_values() -> None:
    """Verify CLI tag strings are normalized before W&B init."""
    assert utils.wandb.parse_wandb_tags(" smoke, docker ,,offline ") == ("smoke", "docker", "offline")
    assert utils.wandb.parse_wandb_tags(None) == ()
    assert utils.wandb.parse_wandb_tags([" smoke ", "", "docker"]) == ("smoke", "docker")


def test_auto_wandb_mode_resolves_from_credentials(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify auto mode uses offline without credentials and online when a key is present."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "missing-home"))

    assert utils.wandb.resolve_wandb_mode("auto") == "offline"

    monkeypatch.setenv("WANDB_API_KEY", "secret-test-key")

    assert utils.wandb.resolve_wandb_mode("auto") == "online"


def test_online_wandb_without_key_fails_before_import(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify online mode cannot hang on login when credentials are absent."""
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path / "missing-home"))

    with pytest.raises(RuntimeError, match="WANDB_API_KEY"):
        utils.wandb.start_wandb_run(
            utils.wandb.WandbTrackingSettings(mode="online"),
            config={},
        )


def test_wandb_key_can_be_loaded_from_home_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the optional home key file populates the environment without printing it."""
    key_path = tmp_path / "wandb_key.txt"
    key_path.write_text("secret-test-key\n", encoding="utf-8")
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))

    utils.wandb._load_wandb_api_key_from_home_file()  # noqa: SLF001

    assert "WANDB_API_KEY" in __import__("os").environ
    assert __import__("os").environ["WANDB_API_KEY"] == "secret-test-key"


def test_wandb_summary_metrics_group_final_diagnostics() -> None:
    """Verify final PPO diagnostics are grouped into W&B summary namespaces."""
    metrics = {
        "training_run_name": "ppo_line_smoke",
        "task_shape": "line",
        "task_index": 2,
        "seed": 7,
        "total_timesteps": 4096,
        "normalize_actions": True,
        "eval_steps": EVAL_STEPS,
        "actual_eval_steps": 118,
        "eval_terminated_count": 1,
        "eval_truncated_count": 0,
        "eval_reset_count": 1,
        "episode_count": 2,
        "mean_eval_reward": -1.5,
        "final_eval_reward": -0.5,
        "mean_position_error_m": MEAN_POSITION_ERROR_M,
        "final_position_error_m": 0.3,
        "max_position_error_m": 0.9,
        "mean_abs_x_error": 0.1,
        "mean_abs_y_error": 0.2,
        "mean_abs_z_error": MEAN_ABS_Z_ERROR_M,
        "final_abs_x_error": 0.4,
        "final_abs_y_error": 0.5,
        "final_abs_z_error": 0.6,
        "reference_xy_span_m": 1.2,
        "actual_xy_span_m": 0.8,
        "xy_tracking_ratio": 0.67,
        "action_mean": [ACTION_MEAN_0, 0.2, MEAN_ABS_Z_ERROR_M],
        "action_std": [0.01, 0.02, 0.03],
        "action_min": [-1.0, -0.5, 0.0],
        "action_max": [0.9, 0.8, 1.0],
        "action_saturation_fraction": [0.0, 0.25, 1.0],
        "real_action_mean": [0.4, 0.0, REAL_ACTION_MEAN_2],
        "real_action_std": [0.1, 0.0, 0.05],
        "real_action_min": [-0.2, -0.2, 0.5],
        "real_action_max": [1.0, 0.2, 1.0],
        "real_action_saturation_fraction": [REAL_ACTION_SATURATION_0, 0.0, 1.0],
        "failure_modes": ["hover_lock", "action_saturation"],
        "failure_primary_mode": "hover_lock",
        "curriculum_readiness_level": "line_not_ready",
        "curriculum_recommended_next_tasks": ["short_slow_line"],
        "curriculum_avoid_next_tasks": ["circle"],
        "position_bounds": {"min": [0, 0, 0], "max": [1, 1, 1]},
        "liftoff_diagnostics": {"zero_action": {"z_max": 0.1}},
        "evaluation_trace_path": "storage/tmp/evaluation_trace.jsonl",
    }

    summary = utils.wandb.build_wandb_summary_metrics(metrics)

    assert summary["tracking/mean_position_error_m"] == MEAN_POSITION_ERROR_M
    assert summary["tracking/mean_abs_z_error_m"] == MEAN_ABS_Z_ERROR_M
    assert summary["actions/saturation_fraction_2"] == 1.0
    assert summary["actions/real_saturation_fraction_0"] == REAL_ACTION_SATURATION_0
    assert summary["actions/real_mean_2"] == REAL_ACTION_MEAN_2
    assert summary["actions/mean_0"] == ACTION_MEAN_0
    assert summary["evaluation/eval_steps"] == EVAL_STEPS
    assert summary["evaluation/terminated_count"] == 1
    assert summary["failure/hover_lock"] == 1
    assert summary["failure/action_saturation"] == 1
    assert summary["failure/no_failure_detected"] == 0
    assert summary["curriculum/readiness_level"] == "line_not_ready"
    assert summary["curriculum/recommended_next_tasks"] == ["short_slow_line"]
    assert summary["run/training_run_name"] == "ppo_line_smoke"
    assert summary["run/normalize_actions"] is True
    assert "position_bounds" not in summary
    assert "liftoff_diagnostics" not in summary
    assert "evaluation_trace_path" not in summary
    assert "eval_steps" not in summary
    assert "xy_tracking_ratio" not in summary


def test_wandb_summary_writer_uses_summary_not_history() -> None:
    """Verify final diagnostics do not get logged as a one-point W&B history row."""

    class FakeRun:
        """Tiny W&B run stand-in for summary writes."""

        def __init__(self) -> None:
            """Create empty summary and history containers."""
            self.summary: dict[str, Any] = {}
            self.history: list[dict[str, Any]] = []

        def log(self, payload: dict[str, Any]) -> None:
            """Record unexpected history logs for assertions."""
            self.history.append(payload)

    run = FakeRun()

    utils.wandb.log_wandb_summary(run, {"mean_position_error_m": 0.25, "eval_steps": 10})

    assert run.summary == {
        "tracking/mean_position_error_m": 0.25,
        "evaluation/eval_steps": 10,
        "failure/hover_lock": 0,
        "failure/insufficient_xy_motion": 0,
        "failure/action_saturation": 0,
        "failure/overshoot": 0,
        "failure/z_instability": 0,
        "failure/attitude_instability": 0,
        "failure/early_termination": 0,
        "failure/repeated_truncation": 0,
        "failure/reference_too_fast_or_too_hard": 0,
        "failure/no_failure_detected": 0,
    }
    assert run.history == []
