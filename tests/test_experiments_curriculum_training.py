"""Tests for manual PPO curriculum training orchestration."""

# ruff: noqa: S101

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from src.experiments.cli import experiments_cli_train_curriculum as cli_train_curriculum
from src.experiments.curriculum import experiments_curriculum_training as curriculum_training
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking

EXPECTED_CURRICULUM_STAGE_COUNT = 2
CLI_SEED_OVERRIDE = 3
EXPECTED_CURRICULUM_NUM_ENVS = 1
EXPECTED_CURRICULUM_EFFECTIVE_ROLLOUT_STEPS = 256

EXPECTED_STAGE_COUNT = 5


def test_manual_curriculum_config_loads_and_validates() -> None:
    """Verify the manual line curriculum config exposes the expected stages."""
    settings = curriculum_training.load_manual_curriculum_settings("configs/curricula/curriculum_manual_line_smoke.yaml")

    assert settings.curriculum_name == "curriculum_manual_line_smoke"
    assert settings.base_training_config == Path("configs/training/ppo_tracking_smoke.yaml")
    assert settings.seed == 0
    assert settings.normalize_actions is True
    assert len(settings.stages) == EXPECTED_STAGE_COUNT
    assert [stage.stage_name for stage in settings.stages] == [
        "hover_stabilization",
        "nearby_target_hover",
        "start_hold_then_short_line",
        "short_slow_line",
        "line",
    ]
    curriculum_training.validate_manual_curriculum(settings)


def test_manual_curriculum_stage_run_name_derivation() -> None:
    """Verify stage run names match the documented manual curriculum contract."""
    run_name = curriculum_training.derive_stage_run_name(
        curriculum_name="curriculum_manual_line_smoke",
        stage_index=3,
        stage_name="start_hold_then_short_line",
        seed=0,
    )

    assert run_name == "curriculum_manual_line_smoke_stage03_start_hold_then_short_line_seed0"


def test_manual_curriculum_invalid_stage_fails_clearly() -> None:
    """Verify invalid configured tasks fail before training."""
    settings = curriculum_training.manual_curriculum_settings_from_mapping(
        {
            "curriculum_name": "curriculum_manual_line_smoke",
            "base_training_config": "configs/training/ppo_tracking_smoke.yaml",
            "seed": 0,
            "wandb_mode": "disabled",
            "normalize_actions": True,
            "stages": [
                {
                    "stage_name": "bad_line",
                    "task_shape": "short_slow_line",
                    "total_timesteps": 8,
                    "eval_steps": 4,
                    "task": {
                        "task_type": "trajectory",
                        "shape": "short_slow_line",
                        "duration_sec": 2.0,
                        "sample_rate_hz": 10.0,
                        "start": [0.0, 0.0, 1.0],
                        "end": [10.0, 0.0, 1.0],
                    },
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="invalid curriculum stage 'bad_line'"):
        curriculum_training.validate_manual_curriculum(settings)


def test_manual_curriculum_summary_writing_includes_diagnostics_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify curriculum summary artifacts contain compact per-stage diagnostics metadata."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = curriculum_training.manual_curriculum_settings_from_mapping(
        {
            "curriculum_name": "curriculum_manual_line_smoke",
            "base_training_config": "configs/training/ppo_tracking_smoke.yaml",
            "seed": 0,
            "wandb_mode": "disabled",
            "normalize_actions": True,
            "stages": [
                {
                    "stage_name": "hover_stabilization",
                    "task_shape": "hover_stabilization",
                    "total_timesteps": 8,
                    "eval_steps": 4,
                    "task": {
                        "task_type": "trajectory",
                        "shape": "hover_stabilization",
                        "duration_sec": 2.0,
                        "sample_rate_hz": 5.0,
                        "position": [0.0, 0.0, 1.0],
                    },
                },
                {
                    "stage_name": "nearby_target_hover",
                    "task_shape": "nearby_target_hover",
                    "total_timesteps": 8,
                    "eval_steps": 4,
                    "task": {
                        "task_type": "trajectory",
                        "shape": "nearby_target_hover",
                        "duration_sec": 2.0,
                        "sample_rate_hz": 5.0,
                        "position": [0.1, 0.0, 1.0],
                    },
                },
            ],
        }
    )
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> ppo_tracking.PPOTrackingSmokeResult:
        run_name = str(kwargs["run_name"])
        initial_model_path = kwargs.get("initial_model_path")
        calls.append(dict(kwargs))
        metrics = {
            "seed": kwargs["seed"],
            "diagnostics_dir": str(tmp_path / run_name / "diagnostics"),
            "num_envs": EXPECTED_CURRICULUM_NUM_ENVS,
            "vec_env_type": "DummyVecEnv",
            "effective_rollout_steps": EXPECTED_CURRICULUM_EFFECTIVE_ROLLOUT_STEPS,
            "mean_position_error_m": 0.1,
            "final_position_error_m": 0.2,
            "max_position_error_m": 0.3,
            "actual_xy_span_m": 0.0,
            "reference_xy_span_m": 0.0,
            "xy_tracking_ratio": None,
            "action_saturation_fraction": [0.0, 0.0, 0.0],
            "real_action_saturation_fraction": [0.0, 0.0, 0.0],
            "failure_overall_status": "passed",
            "failure_primary_mode": "none",
            "failure_modes": [],
            "curriculum_readiness_level": "ready",
            "curriculum_recommended_next_tasks": [],
            "curriculum_avoid_next_tasks": [],
            "initial_model_path": initial_model_path,
            "model_transfer_enabled": initial_model_path is not None,
            "model_transfer_source": initial_model_path,
        }
        return ppo_tracking.PPOTrackingSmokeResult(
            model_path=str(tmp_path / f"{run_name}.zip"),
            metrics_path=str(tmp_path / f"{run_name}_metrics.json"),
            manifest_path=str(tmp_path / f"{run_name}_manifest.json"),
            metrics=metrics,
        )

    monkeypatch.setattr(ppo_tracking, "run_ppo_tracking_smoke_from_config", fake_run)

    result = curriculum_training.run_manual_curriculum_training(settings)
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))

    expected_root = tmp_path / "runs" / "curriculum_manual_line_smoke_seed0"
    assert Path(result.summary_path) == expected_root / "run_manifest.json"
    assert Path(result.manifest_path) == expected_root / "run_manifest.json"
    assert Path(result.summary_path).exists()
    assert Path(result.manifest_path).exists()
    assert summary["run_name"] == "curriculum_manual_line_smoke_seed0"
    assert summary["artifact_root"] == str(expected_root)
    assert summary["artifact_root_relative"] == "."
    assert summary["summary_path"] == str(expected_root / "run_manifest.json")
    assert summary["summary_path_relative"] == "run_manifest.json"
    assert summary["manifest_path_relative"] == "run_manifest.json"
    assert summary["curriculum_config_snapshot_path"] == str(expected_root / "config" / "curriculum_config.yaml")
    assert summary["curriculum_config_snapshot_path_relative"] == "config/curriculum_config.yaml"
    assert (expected_root / "config" / "curriculum_config.yaml").exists()
    assert summary["config"]["curriculum_config_snapshot_path_relative"] == "config/curriculum_config.yaml"
    assert summary["evaluation_index"]["path_relative"] == "evaluation_index.json"
    assert summary["evaluation_index"]["entry_count"] == 0
    assert summary["run_kind"] == "curriculum"
    assert summary["curriculum_kind"] == "manual"
    assert summary["curriculum_name"] == "curriculum_manual_line_smoke"
    assert summary["stage_count"] == EXPECTED_CURRICULUM_STAGE_COUNT
    assert summary["model_transfer_enabled"] is True
    assert summary["final_stage_run_name"] == "curriculum_manual_line_smoke_stage02_nearby_target_hover_seed0"
    assert summary["final_stage"]["stage_index"] == EXPECTED_CURRICULUM_STAGE_COUNT
    assert summary["final_stage"]["stage_name"] == "nearby_target_hover"
    assert summary["final_stage"]["run_name"] == "curriculum_manual_line_smoke_stage02_nearby_target_hover_seed0"
    assert summary["final_stage"]["model_path"] == summary["final_model_path"]
    assert summary["final_stage"]["model_path_relative"] == summary["stages"][1]["model_path_relative"]
    assert summary["final_stage"]["manifest_path_relative"] == summary["stages"][1]["manifest_path_relative"]
    assert summary["stages"][0]["stage_dir"] == str(expected_root / "stages" / "stage01_hover_stabilization")
    assert summary["stages"][0]["stage_dir_relative"] == "stages/stage01_hover_stabilization"
    assert summary["stages"][0]["training_dir"] == str(expected_root / "stages" / "stage01_hover_stabilization" / "training")
    assert summary["stages"][0]["training_dir_relative"] == "stages/stage01_hover_stabilization/training"
    assert summary["stages"][0]["manifest_path_relative"].endswith("_manifest.json")
    assert summary["stages"][0]["diagnostics_dir"].endswith("diagnostics")
    assert summary["stages"][0]["num_envs"] == EXPECTED_CURRICULUM_NUM_ENVS
    assert summary["stages"][0]["vec_env_type"] == "DummyVecEnv"
    assert summary["stages"][0]["effective_rollout_steps"] == EXPECTED_CURRICULUM_EFFECTIVE_ROLLOUT_STEPS
    assert summary["stages"][0]["model_transfer_enabled"] is False
    assert summary["stages"][1]["model_transfer_enabled"] is True
    assert summary["stages"][1]["previous_model_path"] == summary["stages"][0]["model_path"]
    assert calls[0]["initial_model_path"] is None
    assert calls[1]["initial_model_path"] == summary["stages"][0]["model_path"]
    assert calls[0]["artifact_root"] == expected_root / "stages" / "stage01_hover_stabilization" / "training"
    assert calls[1]["artifact_root"] == expected_root / "stages" / "stage02_nearby_target_hover" / "training"
    assert calls[0]["normalize_actions"] is True
    assert "num_envs" not in calls[0]
    assert calls[0]["wandb_group"] == "curriculum/curriculum_manual_line_smoke"
    assert Path(calls[0]["task_config_path"]).parent == expected_root / "config"
    assert Path(calls[0]["task_config_path"]).name == "stage01_hover_stabilization_task.yaml"
    assert Path(calls[1]["task_config_path"]).name == "stage02_nearby_target_hover_task.yaml"


def test_manual_curriculum_cli_parser_accepts_expected_options() -> None:
    """Verify the curriculum parser exposes config, seed, and W&B controls."""
    parser = cli_train_curriculum.build_parser()
    args = parser.parse_args(["--config", "configs/curricula/curriculum_manual_line_smoke.yaml", "--seed", "3", "--wandb-mode", "offline"])

    assert args.config == Path("configs/curricula/curriculum_manual_line_smoke.yaml")
    assert args.seed == CLI_SEED_OVERRIDE
    assert args.wandb_mode == "offline"


def test_manual_curriculum_cli_help_works() -> None:
    """Verify the manual curriculum CLI exposes help without running training."""
    completed = subprocess.run(
        [sys.executable, "-m", "src.experiments.cli.experiments_cli_train_curriculum", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "--config" in completed.stdout
    assert "--seed" in completed.stdout
    assert "--wandb-mode" in completed.stdout
