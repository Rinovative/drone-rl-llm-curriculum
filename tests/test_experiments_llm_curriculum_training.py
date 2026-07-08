"""Tests for LLM-guided curriculum training orchestration."""

# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.cli import experiments_cli_train_llm_curriculum as cli_train_llm_curriculum
from src.experiments.curriculum import experiments_curriculum_llm_training as llm_curriculum_training
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking

CONFIG_STAGE_COUNT = 3
DRY_RUN_STAGE_COUNT = 2
TRAINING_STAGE_COUNT = 2
CLI_SEED_OVERRIDE = 3
CLI_MAX_STAGES = 2
CLI_MAX_REPAIR_ATTEMPTS = 1
EXPECTED_PID_ACTION_DIM = 3
EXPECTED_BASE_OBSERVATION_DIM = 10
EXPECTED_LOCAL_LLM_MAX_STAGES = 10
EXPECTED_LLM_BUDGET_CAP = 350000
EXPECTED_LLM_BUDGET_PROFILES = {"short": 20000, "normal": 30000, "recovery": 40000, "extend": 50000}
BUDGET_TEST_STAGE_COUNT = 4
BUDGET_TEST_CAP = 110
BUDGET_TEST_STAGE_BUDGETS = [30, 20, 40, 20]


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
    proposal_events = [event for event in events if event["event_type"] == "llm_proposal_attempt"]
    assert summary["stages"][1]["task_shape"] == "start_hold_then_short_line"
    assert summary["stages"][1]["task_reason"]
    assert proposal_events[0]["status"] == "accepted"
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
            "action_interface": "pid_position",
            "ppo_action_dim": EXPECTED_PID_ACTION_DIM,
            "real_action_type": "pid_target_position",
            "real_action_space_bounds": {"low": [[-1.0, -1.0, -1.0]], "high": [[1.0, 1.0, 1.0]], "units": "meters"},
            "rpm_delta_scale": None,
            "include_dynamics_observation": False,
            "include_previous_action": False,
            "observation_dim": EXPECTED_BASE_OBSERVATION_DIM,
            "observation_components": [
                {"name": "current_position", "dim": 3},
                {"name": "reference_position", "dim": 3},
                {"name": "position_error", "dim": 3},
                {"name": "trajectory_progress", "dim": 1},
            ],
            "policy_kwargs": None,
            "direct_control_limitations": [],
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
    assert calls[0]["config_path"] == Path("configs/training/ppo_tracking_smoke.yaml")
    assert "action_interface" not in calls[0]
    assert summary["model_transfer_enabled"] is True
    assert summary["action_interface"] == "pid_position"
    assert summary["ppo_action_dim"] == EXPECTED_PID_ACTION_DIM
    assert summary["real_action_type"] == "pid_target_position"
    assert summary["include_dynamics_observation"] is False
    assert summary["include_previous_action"] is False
    assert summary["observation_dim"] == EXPECTED_BASE_OBSERVATION_DIM
    assert summary["policy_kwargs"] is None
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


def test_llm_taskdist_curriculum_configs_load() -> None:
    """Verify local LLM task-distribution curriculum configs resolve with taskdist bases."""
    for config_path in (
        "configs/curricula/curriculum_llm_local_pid_dynprev_taskdist_medium_medium.yaml",
        "configs/curricula/curriculum_llm_local_directrpm_dynprev_taskdist_medium_medium.yaml",
    ):
        settings = llm_curriculum_training.load_llm_curriculum_settings(config_path)
        base_settings = ppo_tracking.load_ppo_tracking_settings(settings.base_training_config)

        assert settings.llm_provider == "openai_compatible"
        assert settings.max_stages == EXPECTED_LOCAL_LLM_MAX_STAGES
        assert settings.llm_stage_budget.enabled is True
        assert settings.llm_stage_budget.total_budget_cap_timesteps == EXPECTED_LLM_BUDGET_CAP
        assert settings.llm_stage_budget.profiles == EXPECTED_LLM_BUDGET_PROFILES
        assert base_settings.task_distribution_settings is not None
        assert base_settings.include_dynamics_observation is True
        assert base_settings.include_previous_action is True


def _budget_test_settings() -> llm_curriculum_training.LLMCurriculumSettings:
    """Return tiny dry-run settings that exercise adaptive budget resolution."""
    return llm_curriculum_training.llm_curriculum_settings_from_mapping(
        {
            "curriculum_name": "curriculum_llm_budget_unit",
            "base_training_config": "configs/training/ppo_tracking_smoke.yaml",
            "seed": 5,
            "wandb_mode": "disabled",
            "normalize_actions": True,
            "max_stages": 4,
            "stage_defaults": {"total_timesteps": 30, "eval_steps": 4},
            "llm_stage_budget": {
                "enabled": True,
                "total_budget_cap_timesteps": 110,
                "default_profile": "normal",
                "min_stage_timesteps": 20,
                "max_stage_timesteps": 50,
                "profiles": {
                    "short": {"total_timesteps": 20},
                    "normal": {"total_timesteps": 30},
                    "recovery": {"total_timesteps": 40},
                    "extend": {"total_timesteps": 50},
                },
            },
            "bootstrap": {
                "enabled": True,
                "stage_name": "hover_stabilization",
                "task_shape": "hover_stabilization",
                "stage_budget_profile": "normal",
                "budget_rationale": "Bootstrap with normal budget.",
                "total_timesteps": 30,
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
                        '{"proposal_kind":"task_distribution","task_distribution_id":"tracking_medium",'
                        '"stage_budget_profile":"extend","budget_rationale":"Probe medium distribution."}'
                    ),
                    (
                        '{"task_type":"trajectory","shape":"nearby_target_hover","duration_sec":2.5,'
                        '"sample_rate_hz":10.0,"position":[0.1,0.0,1.0],'
                        '"stage_budget_profile":"recovery","budget_rationale":"Recover after mixed tracking."}'
                    ),
                    (
                        '{"proposal_kind":"task_distribution","task_distribution_id":"tracking_medium",'
                        '"stage_budget_profile":"extend","budget_rationale":"Try to extend if cap allows."}'
                    ),
                ],
            },
        }
    )


def test_llm_adaptive_budget_dry_run_logs_metadata_and_enforces_cap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify adaptive budget decisions appear in logs and never exceed the configured cap."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))

    result = llm_curriculum_training.run_llm_curriculum_training(_budget_test_settings(), dry_run_proposals=True)
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    events = [json.loads(line) for line in Path(result.proposal_log_path).read_text(encoding="utf-8").splitlines() if line]
    budget_events = [event for event in events if event["event_type"] == "llm_stage_budget_decision"]

    assert summary["stage_count"] == BUDGET_TEST_STAGE_COUNT
    assert summary["llm_budget_cap_timesteps"] == BUDGET_TEST_CAP
    assert summary["cumulative_llm_budget_timesteps"] == BUDGET_TEST_CAP
    assert [stage["stage_total_timesteps"] for stage in summary["stages"]] == BUDGET_TEST_STAGE_BUDGETS
    assert summary["stages"][1]["requested_stage_budget_profile"] == "extend"
    assert summary["stages"][1]["selected_stage_budget_profile"] == "short"
    assert summary["stages"][1]["budget_was_clipped"] is True
    assert summary["stages"][1]["budget_rationale"] == "Probe medium distribution."
    assert len(budget_events) == BUDGET_TEST_STAGE_COUNT
    assert budget_events[-1]["cumulative_llm_budget_timesteps"] == BUDGET_TEST_CAP
    assert any(event.get("stage_budget_profile") == "extend" for event in events if event["event_type"] == "llm_proposal_attempt")


def test_llm_budget_cap_must_reserve_minimum_budget_for_all_stages() -> None:
    """Verify impossible total caps are rejected at config load time."""
    config = {
        "curriculum_name": "curriculum_llm_bad_budget",
        "base_training_config": "configs/training/ppo_tracking_smoke.yaml",
        "max_stages": 4,
        "stage_defaults": {"total_timesteps": 30, "eval_steps": 4},
        "llm_stage_budget": {
            "enabled": True,
            "total_budget_cap_timesteps": 70,
            "default_profile": "normal",
            "min_stage_timesteps": 20,
            "max_stage_timesteps": 50,
            "profiles": {
                "short": {"total_timesteps": 20},
                "normal": {"total_timesteps": 30},
                "recovery": {"total_timesteps": 40},
                "extend": {"total_timesteps": 50},
            },
        },
        "bootstrap": {
            "enabled": True,
            "stage_name": "hover_stabilization",
            "task_shape": "hover_stabilization",
            "task": {
                "task_type": "trajectory",
                "shape": "hover_stabilization",
                "duration_sec": 2.0,
                "sample_rate_hz": 10.0,
                "position": [0.0, 0.0, 1.0],
            },
        },
        "llm": {"provider": "mock", "mock_responses": []},
    }

    with pytest.raises(ValueError, match="total cap"):
        llm_curriculum_training.llm_curriculum_settings_from_mapping(config)


def test_old_local_llm_smoke_config_keeps_adaptive_budget_disabled() -> None:
    """Verify legacy local LLM smoke config loads without adaptive budget settings."""
    settings = llm_curriculum_training.load_llm_curriculum_settings("configs/curricula/curriculum_llm_local_smoke.yaml")

    assert settings.llm_stage_budget.enabled is False
    assert settings.llm_stage_budget.profiles == {"normal": settings.stage_total_timesteps}
