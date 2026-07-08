"""Tests for LLM-guided curriculum training orchestration."""

# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from src.experiments.cli import experiments_cli_train_llm_curriculum as cli_train_llm_curriculum
from src.experiments.curriculum import experiments_curriculum_llm_training as llm_curriculum_training
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking

if TYPE_CHECKING:
    import pytest

CONFIG_STAGE_COUNT = 3
DRY_RUN_STAGE_COUNT = 2
TRAINING_STAGE_COUNT = 2
CLI_SEED_OVERRIDE = 3
CLI_MAX_STAGES = 2
CLI_MAX_REPAIR_ATTEMPTS = 1


def test_llm_curriculum_config_loads_and_validates() -> None:
    """Verify the mock LLM curriculum config exposes bootstrap and provider settings."""
    settings = llm_curriculum_training.load_llm_curriculum_settings("configs/curricula/curriculum_llm_smoke.yaml")

    assert settings.curriculum_name == "curriculum_llm_smoke"
    assert settings.base_training_config == Path("configs/training/ppo_tracking_smoke.yaml")
    assert settings.seed == 0
    assert settings.max_stages == CONFIG_STAGE_COUNT
    assert settings.llm_provider == "mock"
    assert settings.bootstrap_stage is not None
    assert settings.bootstrap_stage.task_shape == "hover_stabilization"
    llm_curriculum_training.validate_llm_curriculum(settings)


def test_llm_curriculum_dry_run_writes_manifest_and_proposal_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify dry-run mode exercises proposals without launching PPO training."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))

    result = llm_curriculum_training.run_llm_curriculum_training_from_config(
        config_path="configs/curricula/curriculum_llm_smoke.yaml",
        max_stages=2,
        dry_run_proposals=True,
    )
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    events = [json.loads(line) for line in Path(result.proposal_log_path).read_text(encoding="utf-8").splitlines() if line]

    expected_root = tmp_path / "runs" / "curriculum_llm_smoke_seed0"
    assert Path(result.summary_path) == expected_root / "run_manifest.json"
    assert Path(result.proposal_log_path) == expected_root / "llm_logs" / "proposals.jsonl"
    assert summary["run_kind"] == "curriculum"
    assert summary["curriculum_kind"] == "llm"
    assert summary["mode"] == "llm_curriculum"
    assert summary["dry_run_proposals"] is True
    assert summary["stage_count"] == DRY_RUN_STAGE_COUNT
    assert summary["final_model_path"] is None
    assert summary["llm_provider"] == "mock"
    assert summary["proposal_stats"]["total_proposals"] == 1
    assert summary["proposal_stats"]["final_accepted_tasks"] == 1
    assert summary["stages"][0]["task_shape"] == "hover_stabilization"
    assert summary["stages"][1]["task_shape"] == "start_hold_then_short_line"
    assert summary["stages"][1]["task_reason"]
    assert events[0]["status"] == "accepted"
    assert not (expected_root / "training").exists()


def test_llm_curriculum_training_uses_ppo_stage_helper_and_model_transfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify non-dry orchestration reuses the PPO helper and transfers models between stages."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = llm_curriculum_training.llm_curriculum_settings_from_mapping(
        {
            "curriculum_name": "curriculum_llm_unit",
            "base_training_config": "configs/training/ppo_tracking_smoke.yaml",
            "seed": 7,
            "wandb_mode": "disabled",
            "normalize_actions": True,
            "max_stages": 2,
            "stage_defaults": {"total_timesteps": 8, "eval_steps": 4},
            "bootstrap": {
                "enabled": True,
                "stage_name": "hover_stabilization",
                "task_shape": "hover_stabilization",
                "total_timesteps": 8,
                "eval_steps": 4,
                "task": {
                    "task_type": "trajectory",
                    "shape": "hover_stabilization",
                    "duration_sec": 2.0,
                    "sample_rate_hz": 10.0,
                    "position": [0.0, 0.0, 1.0],
                },
            },
            "llm": {
                "provider": "mock",
                "model": "mock",
                "max_repair_attempts": 1,
                "mock_responses": [
                    (
                        '{"task_type":"trajectory","shape":"nearby_target_hover","duration_sec":2.5,'
                        '"sample_rate_hz":10.0,"position":[0.1,0.0,1.0],"reason":"Small offset."}'
                    )
                ],
            },
        }
    )
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> ppo_tracking.PPOTrackingSmokeResult:
        run_name = str(kwargs["run_name"])
        calls.append(dict(kwargs))
        metrics = {
            "seed": kwargs["seed"],
            "diagnostics_dir": str(tmp_path / run_name / "diagnostics"),
            "mean_position_error_m": 0.1,
            "mean_position_error_tracking_m": 0.1,
            "final_position_error_m": 0.2,
            "max_position_error_m": 0.3,
            "xy_tracking_ratio": None,
            "failure_overall_status": "passed",
            "failure_primary_mode": "none",
            "failure_modes": [],
            "curriculum_readiness_level": "ready",
            "curriculum_recommended_next_tasks": [],
            "curriculum_avoid_next_tasks": [],
        }
        return ppo_tracking.PPOTrackingSmokeResult(
            model_path=str(tmp_path / f"{run_name}.zip"),
            metrics_path=str(tmp_path / f"{run_name}_metrics.json"),
            manifest_path=str(tmp_path / f"{run_name}_manifest.json"),
            metrics=metrics,
        )

    monkeypatch.setattr(ppo_tracking, "run_ppo_tracking_smoke_from_config", fake_run)

    result = llm_curriculum_training.run_llm_curriculum_training(settings)
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))

    expected_root = tmp_path / "runs" / "curriculum_llm_unit_seed7"
    assert Path(result.summary_path) == expected_root / "run_manifest.json"
    assert len(calls) == TRAINING_STAGE_COUNT
    assert calls[0]["initial_model_path"] is None
    assert calls[1]["initial_model_path"] == summary["stages"][0]["model_path"]
    assert calls[0]["artifact_root"] == expected_root / "stages" / "stage01_hover_stabilization" / "training"
    assert calls[1]["artifact_root"] == expected_root / "stages" / "stage02_nearby_target_hover" / "training"
    assert calls[0]["wandb_group"] == "curriculum/curriculum_llm_unit"
    assert summary["model_transfer_enabled"] is True
    assert summary["final_stage_run_name"] == "curriculum_llm_unit_stage02_nearby_target_hover_seed7"
    assert summary["final_model_path"] == summary["stages"][1]["model_path"]
    assert summary["stages"][1]["task"] == {
        "task_type": "trajectory",
        "shape": "nearby_target_hover",
        "duration_sec": 2.5,
        "sample_rate_hz": 10.0,
        "position": [0.1, 0.0, 1.0],
    }
    assert summary["stages"][1]["task_reason"] == "Small offset."
    assert summary["proposal_log_path_relative"] == "llm_logs/proposals.jsonl"


def test_llm_curriculum_cli_parser_accepts_expected_options() -> None:
    """Verify the LLM curriculum parser exposes provider, stage, repair, and dry-run controls."""
    parser = cli_train_llm_curriculum.build_parser()
    args = parser.parse_args(
        [
            "--config",
            "configs/curricula/curriculum_llm_smoke.yaml",
            "--seed",
            "3",
            "--wandb-mode",
            "offline",
            "--provider",
            "mock",
            "--max-stages",
            "2",
            "--max-repair-attempts",
            "1",
            "--dry-run-proposals",
        ]
    )

    assert args.config == Path("configs/curricula/curriculum_llm_smoke.yaml")
    assert args.seed == CLI_SEED_OVERRIDE
    assert args.wandb_mode == "offline"
    assert args.provider == "mock"
    assert args.max_stages == CLI_MAX_STAGES
    assert args.max_repair_attempts == CLI_MAX_REPAIR_ATTEMPTS
    assert args.dry_run_proposals is True
