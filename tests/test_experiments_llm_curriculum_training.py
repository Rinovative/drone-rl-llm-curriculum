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
EXPECTED_TRAINING_TOTAL_TIMESTEPS = 16
EXPECTED_LOCAL_LLM_MAX_STAGES = 10
EXPECTED_REFERENCE_MEDIUM_TIMESTEPS = 500000
EXPECTED_LLM_BUDGET_CAP = 2500000
EXPECTED_LLM_BUDGET_PROFILES = {
    "bootstrap": 500000,
    "short": 175000,
    "normal": 250000,
    "recovery": 325000,
    "extend": 400000,
}
EXPECTED_LLM_BUDGET_MULTIPLIERS = {"bootstrap": 1.0, "short": 0.35, "normal": 0.5, "recovery": 0.65, "extend": 0.8}
EXPECTED_BOOTSTRAP_DISTRIBUTION_CONFIG = Path("configs/tasks/task_distribution_hover_bootstrap_medium.yaml")
EXPECTED_BOOTSTRAP_BOUNDS = {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.7, 1.4]}
BUDGET_TEST_STAGE_COUNT = 4
BUDGET_TEST_CAP = 110
BUDGET_TEST_STAGE_BUDGETS = [30, 40, 20, 20]
TASKDIST_RESOLUTION_EFFECTIVE_SEED = 12
FALLBACK_FAILED_PROPOSAL_COUNT = 2


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
        best_model_path = str(tmp_path / f"{run_name}_best.zip") if "stage01" in run_name else None
        return ppo_tracking.PPOTrackingSmokeResult(
            model_path=str(tmp_path / f"{run_name}.zip"),
            metrics_path=str(tmp_path / f"{run_name}_metrics.json"),
            manifest_path=str(tmp_path / f"{run_name}_manifest.json"),
            metrics=metrics,
            last_model_path=str(tmp_path / f"{run_name}.zip"),
            best_model_path=best_model_path,
            best_model_metric="mean_position_error_m" if best_model_path is not None else None,
            best_model_step=8 if best_model_path is not None else None,
            best_model_source="unit_test_best" if best_model_path is not None else None,
        )

    monkeypatch.setattr(ppo_tracking, "run_ppo_tracking_smoke_from_config", fake_run)

    result = llm_curriculum_training.run_llm_curriculum_training(settings)
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))

    expected_root = tmp_path / "runs" / "curriculum_llm_unit_seed7"
    assert Path(result.summary_path) == expected_root / "run_manifest.json"
    assert len(calls) == TRAINING_STAGE_COUNT
    assert calls[0]["initial_model_path"] is None
    assert calls[1]["initial_model_path"] == summary["stages"][0]["best_model_path"]
    assert calls[0]["artifact_root"] == expected_root / "stages" / "stage01_hover_stabilization" / "training"
    assert calls[1]["artifact_root"] == expected_root / "stages" / "stage02_nearby_target_hover" / "training"
    assert calls[0]["wandb_group"] == "curriculum/llm/curriculum_llm_unit_seed7"
    assert calls[0]["wandb_tags"] == (
        "stage_index:1",
        "stage:hover_stabilization",
        "task:hover_stabilization",
        "llm_provider:mock",
        "llm_fallback:false",
        "llm_budget_profile:normal",
    )
    assert calls[0]["run_metadata"]["run_kind"] == "curriculum_stage"
    assert calls[0]["run_metadata"]["curriculum_kind"] == "llm"
    assert calls[0]["run_metadata"]["curriculum_run_name"] == "curriculum_llm_unit_seed7"
    assert calls[0]["run_metadata"]["curriculum_stage_index"] == 1
    assert calls[0]["run_metadata"]["curriculum_stage_name"] == "hover_stabilization"
    assert calls[0]["run_metadata"]["curriculum_stage_run_name"] == "curriculum_llm_unit_stage01_hover_stabilization_seed7"
    assert calls[0]["run_metadata"]["llm_provider"] == "mock"
    assert calls[0]["run_metadata"]["stage_budget_profile"] == "normal"
    assert calls[0]["config_path"] == Path("configs/training/ppo_tracking_smoke.yaml")
    assert "action_interface" not in calls[0]
    assert summary["model_transfer_enabled"] is True
    assert summary["stage_run_names"] == [
        "curriculum_llm_unit_stage01_hover_stabilization_seed7",
        "curriculum_llm_unit_stage02_nearby_target_hover_seed7",
    ]
    assert summary["total_configured_timesteps"] == EXPECTED_TRAINING_TOTAL_TIMESTEPS
    assert summary["total_actual_timesteps"] == EXPECTED_TRAINING_TOTAL_TIMESTEPS
    assert summary["budget_profile_counts"] == {"normal": 2}
    assert summary["action_interface"] == "pid_position"
    assert summary["ppo_action_dim"] == EXPECTED_PID_ACTION_DIM
    assert summary["real_action_type"] == "pid_target_position"
    assert summary["include_dynamics_observation"] is False
    assert summary["include_previous_action"] is False
    assert summary["observation_dim"] == EXPECTED_BASE_OBSERVATION_DIM
    assert summary["policy_kwargs"] is None
    assert summary["stages"][0]["selected_transfer_model_path"] == summary["stages"][0]["best_model_path"]
    assert summary["stages"][0]["selected_transfer_model_source"] == "best"
    assert summary["stages"][1]["previous_model_path"] == summary["stages"][0]["best_model_path"]
    assert summary["final_stage_run_name"] == "curriculum_llm_unit_stage02_nearby_target_hover_seed7"
    assert summary["final_model_path"] == summary["stages"][1]["model_path"]
    assert summary["final_model_source"] == "last"
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
        "configs/curricula/llm_curriculum_pid_dynprev_m-taskdist_medium.yaml",
        "configs/curricula/llm_curriculum_directrpm_dynprev_m-taskdist_medium.yaml",
    ):
        settings = llm_curriculum_training.load_llm_curriculum_settings(config_path)
        base_settings = ppo_tracking.load_ppo_tracking_settings(settings.base_training_config)

        assert settings.llm_provider == "openai_compatible"
        assert settings.curriculum_name.startswith("llm_curriculum_")
        assert settings.max_stages == EXPECTED_LOCAL_LLM_MAX_STAGES
        assert settings.reference_medium_config_path == settings.base_training_config
        assert settings.reference_medium_timesteps == EXPECTED_REFERENCE_MEDIUM_TIMESTEPS
        assert settings.stage_budget_multipliers == EXPECTED_LLM_BUDGET_MULTIPLIERS
        assert base_settings.total_timesteps == EXPECTED_REFERENCE_MEDIUM_TIMESTEPS
        assert settings.llm_stage_budget.enabled is True
        assert settings.llm_stage_budget.total_budget_cap_timesteps == EXPECTED_LLM_BUDGET_CAP
        assert settings.llm_stage_budget.profiles == EXPECTED_LLM_BUDGET_PROFILES
        assert settings.llm_stage_budget.min_stage_timesteps == EXPECTED_LLM_BUDGET_PROFILES["short"]
        assert settings.llm_stage_budget.max_stage_timesteps == EXPECTED_LLM_BUDGET_PROFILES["bootstrap"]
        assert settings.bootstrap_stage is not None
        assert settings.bootstrap_stage.stage_name == "hover_stabilization"
        assert settings.bootstrap_stage.task_shape == "hover_stabilization"
        assert settings.bootstrap_stage.total_timesteps == EXPECTED_REFERENCE_MEDIUM_TIMESTEPS
        assert settings.bootstrap_stage.requested_stage_budget_profile == "bootstrap"
        assert settings.bootstrap_stage.task_distribution_id == "bootstrap_randomized_hover_target"
        assert settings.bootstrap_stage.task_distribution_config_path == EXPECTED_BOOTSTRAP_DISTRIBUTION_CONFIG
        assert settings.bootstrap_stage.bootstrap_stage_source == "deterministic_config"
        assert settings.bootstrap_stage.bootstrap_task_shape == "hover_stabilization"
        assert settings.bootstrap_stage.bootstrap_target_sampling_bounds == EXPECTED_BOOTSTRAP_BOUNDS
        run_name = llm_curriculum_training._curriculum_artifact_run_name(settings.curriculum_name, settings.seed)  # noqa: SLF001
        assert run_name == f"{settings.curriculum_name}_seed{settings.seed}"
        assert llm_curriculum_training._curriculum_wandb_group(run_name) == run_name  # noqa: SLF001
        assert settings.proposal_fallback.enabled is True
        assert settings.proposal_fallback.task_distribution_id == "tracking_medium"
        assert settings.proposal_fallback.default_stage_budget_profile == "short"
        assert settings.proposal_fallback.ready_stage_budget_profile == "normal"
        assert base_settings.task_distribution_settings is not None
        assert base_settings.include_dynamics_observation is True
        assert base_settings.include_previous_action is True


def test_llm_taskdist_reference_resolves_to_concrete_stage_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify distribution proposals resolve to concrete validated stage tasks."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = llm_curriculum_training.llm_curriculum_settings_from_mapping(
        {
            "curriculum_name": "curriculum_llm_taskdist_resolve_unit",
            "base_training_config": "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml",
            "seed": 11,
            "wandb_mode": "disabled",
            "normalize_actions": True,
            "max_stages": 1,
            "stage_defaults": {"total_timesteps": 8, "eval_steps": 4},
            "bootstrap": False,
            "llm": {
                "provider": "mock",
                "model": "mock",
                "max_repair_attempts": 0,
                "mock_responses": [
                    (
                        '{"task_distribution_id":"tracking_medium","stage_budget_profile":"normal",'
                        '"budget_rationale":"Use the bounded medium task distribution."}'
                    )
                ],
            },
        }
    )

    result = llm_curriculum_training.run_llm_curriculum_training(settings, dry_run_proposals=True)
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    events = [json.loads(line) for line in Path(result.proposal_log_path).read_text(encoding="utf-8").splitlines() if line]
    stage = summary["stages"][0]
    proposal_events = [event for event in events if event["event_type"] == "llm_proposal_attempt"]
    budget_events = [event for event in events if event["event_type"] == "llm_stage_budget_decision"]

    assert stage["proposal_type"] == "task_distribution"
    assert stage["original_proposal"]["task_distribution_id"] == "tracking_medium"
    assert stage["task_distribution_reference"] == {
        "task_distribution_id": "tracking_medium",
        "task_distribution_config_path": "configs/tasks/task_distribution_tracking_medium.yaml",
    }
    assert stage["task_distribution_config_path"] == "configs/tasks/task_distribution_tracking_medium.yaml"
    assert stage["task_distribution_id"] == "tracking_medium"
    assert stage["task"]["task_type"] == "trajectory"
    assert stage["task"].get("shape")
    assert stage["resolved_task"] == stage["task"]
    assert stage["resolved_task_shape"] == stage["task"]["shape"]
    assert stage["stage_name"] == stage["resolved_task_shape"]
    assert "tracking_medium" not in stage["run_name"]
    assert stage["resolved_task_sample_metadata"]["task_distribution_env_rank"] == 0
    assert stage["resolved_task_sample_metadata"]["task_distribution_effective_seed"] == TASKDIST_RESOLUTION_EFFECTIVE_SEED
    assert stage["proposal_fallback_used"] is False
    assert proposal_events[0]["proposal_type"] == "task_distribution"
    assert budget_events[0]["resolved_task_shape"] == stage["task"]["shape"]


def test_llm_taskdist_fallback_logs_and_resolves_concrete_task(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify enabled fallback prevents overnight taskdist curricula from crashing on bad proposals."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = llm_curriculum_training.llm_curriculum_settings_from_mapping(
        {
            "curriculum_name": "curriculum_llm_taskdist_fallback_unit",
            "base_training_config": "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml",
            "seed": 0,
            "wandb_mode": "disabled",
            "normalize_actions": True,
            "max_stages": 1,
            "stage_defaults": {"total_timesteps": 8, "eval_steps": 4},
            "llm_stage_budget": {
                "enabled": True,
                "total_budget_cap_timesteps": 10,
                "default_profile": "normal",
                "min_stage_timesteps": 4,
                "max_stage_timesteps": 10,
                "profiles": {
                    "bootstrap": {"total_timesteps": 10},
                    "short": {"total_timesteps": 6},
                    "normal": {"total_timesteps": 8},
                    "recovery": {"total_timesteps": 9},
                    "extend": {"total_timesteps": 10},
                },
            },
            "proposal_fallback": {
                "enabled": True,
                "task_distribution_id": "tracking_medium",
                "default_stage_budget_profile": "short",
                "ready_stage_budget_profile": "normal",
            },
            "bootstrap": False,
            "llm": {
                "provider": "mock",
                "model": "mock",
                "max_repair_attempts": 1,
                "skip_invalid_proposals": False,
                "mock_responses": ["{}", "{}"],
            },
        }
    )

    result = llm_curriculum_training.run_llm_curriculum_training(settings, dry_run_proposals=True)
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    events = [json.loads(line) for line in Path(result.proposal_log_path).read_text(encoding="utf-8").splitlines() if line]
    stage = summary["stages"][0]
    fallback_events = [event for event in events if event["event_type"] == "llm_proposal_fallback"]

    assert summary["proposal_fallback"]["enabled"] is True
    assert summary["proposal_fallback_used"] is True
    assert summary["proposal_stats"]["total_proposals"] == FALLBACK_FAILED_PROPOSAL_COUNT
    assert summary["proposal_stats"]["invalid_proposals"] == FALLBACK_FAILED_PROPOSAL_COUNT
    assert summary["proposal_stats"]["fallback_proposals"] == 1
    assert summary["fallback_count"] == 1
    assert summary["repair_count"] == FALLBACK_FAILED_PROPOSAL_COUNT - 1
    assert summary["budget_profile_counts"] == {"short": 1}
    assert stage["proposal_fallback_used"] is True
    assert stage["proposal_type"] == "task_distribution"
    assert stage["selected_stage_budget_profile"] == "short"
    assert stage["stage_budget_profile"] == "short"
    assert stage["task_distribution_reference"]["task_distribution_id"] == "tracking_medium"
    assert stage["task"]["task_type"] == "trajectory"
    assert stage["task"].get("shape")
    assert stage["resolved_task"] == stage["task"]
    assert "missing required keys" in stage["proposal_failure_reason"]
    assert stage["original_proposal"]["proposal_fallback_used"] is True
    assert len(fallback_events) == 1
    assert fallback_events[0]["proposal_fallback_used"] is True
    assert "missing required keys" in fallback_events[0]["proposal_failure_reason"]


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
                    "bootstrap": {"total_timesteps": 30},
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
                "budget_rationale": "Bootstrap with default warmup budget.",
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
    assert summary["budget_profile_counts"] == {"bootstrap": 1, "recovery": 1, "short": 2}
    assert summary["stages"][0]["requested_stage_budget_profile"] == "bootstrap"
    assert summary["stages"][0]["selected_stage_budget_profile"] == "bootstrap"
    assert summary["stages"][1]["requested_stage_budget_profile"] == "extend"
    assert summary["stages"][1]["selected_stage_budget_profile"] == "recovery"
    assert summary["stages"][1]["budget_was_clipped"] is True
    assert "fell back to 'recovery'" in summary["stages"][1]["budget_fallback_reason"]
    assert summary["stages"][1]["budget_rationale"] == "Probe medium distribution."
    assert summary["stages"][2]["requested_stage_budget_profile"] == "recovery"
    assert summary["stages"][2]["selected_stage_budget_profile"] == "short"
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
                "bootstrap": {"total_timesteps": 30},
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


def test_llm_taskdist_stage_names_use_resolved_concrete_shapes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify task-distribution stage names use sampled task shapes instead of the distribution id."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    response = '{"proposal_kind":"task_distribution","task_distribution_id":"tracking_medium"}'
    settings = llm_curriculum_training.llm_curriculum_settings_from_mapping(
        {
            "curriculum_name": "curriculum_llm_stage_name_unit",
            "base_training_config": "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml",
            "seed": 0,
            "wandb_mode": "disabled",
            "normalize_actions": True,
            "max_stages": 4,
            "stage_defaults": {"total_timesteps": 8, "eval_steps": 4},
            "bootstrap": False,
            "llm": {
                "provider": "mock",
                "model": "mock",
                "max_repair_attempts": 0,
                "mock_responses": [response, response, response, response],
            },
        }
    )

    result = llm_curriculum_training.run_llm_curriculum_training(settings, dry_run_proposals=True)
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))
    events = [json.loads(line) for line in Path(result.proposal_log_path).read_text(encoding="utf-8").splitlines() if line]
    stage_names = [stage["stage_name"] for stage in summary["stages"]]

    assert stage_names == ["line", "hover_stabilization", "figure_eight", "vertical"]
    assert all("tracking_medium" not in stage["run_name"] for stage in summary["stages"])
    assert summary["stage_run_names"][2] == "curriculum_llm_stage_name_unit_stage03_figure_eight_seed0"
    assert summary["stages"][2]["resolved_task_shape"] == "figure_eight"
    assert summary["stages"][3]["resolved_task_shape"] == "vertical"
    budget_events = [event for event in events if event["event_type"] == "llm_stage_budget_decision"]
    assert budget_events[1]["stage_name"] == "hover_stabilization"
    assert budget_events[2]["stage_name"] == "figure_eight"
    assert budget_events[2]["accepted_task"]["shape"] == "figure_eight"


def test_llm_taskdist_wandb_identity_uses_resolved_stage_shape(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify stage run names and W&B tags use the resolved task-distribution shape."""
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path))
    settings = llm_curriculum_training.llm_curriculum_settings_from_mapping(
        {
            "curriculum_name": "curriculum_llm_wandb_name_unit",
            "base_training_config": "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml",
            "seed": 2,
            "wandb_mode": "disabled",
            "normalize_actions": True,
            "max_stages": 1,
            "stage_defaults": {"total_timesteps": 8, "eval_steps": 4},
            "bootstrap": False,
            "llm": {
                "provider": "mock",
                "model": "mock",
                "max_repair_attempts": 0,
                "mock_responses": ['{"proposal_kind":"task_distribution","task_distribution_id":"tracking_medium"}'],
            },
        }
    )
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> ppo_tracking.PPOTrackingSmokeResult:
        calls.append(dict(kwargs))
        run_name = str(kwargs["run_name"])
        return ppo_tracking.PPOTrackingSmokeResult(
            model_path=str(tmp_path / f"{run_name}.zip"),
            metrics_path=str(tmp_path / f"{run_name}_metrics.json"),
            manifest_path=str(tmp_path / f"{run_name}_manifest.json"),
            metrics={"seed": kwargs["seed"], "diagnostics_dir": str(tmp_path / run_name / "diagnostics")},
        )

    monkeypatch.setattr(ppo_tracking, "run_ppo_tracking_smoke_from_config", fake_run)

    result = llm_curriculum_training.run_llm_curriculum_training(settings)
    summary = json.loads(Path(result.summary_path).read_text(encoding="utf-8"))

    assert len(calls) == 1
    assert calls[0]["run_name"] == "curriculum_llm_wandb_name_unit_stage01_figure_eight_seed2"
    assert "stage:figure_eight" in calls[0]["wandb_tags"]
    assert "task:figure_eight" in calls[0]["wandb_tags"]
    assert calls[0]["run_metadata"]["curriculum_stage_name"] == "figure_eight"
    assert calls[0]["run_metadata"]["curriculum_stage_run_name"] == calls[0]["run_name"]
    assert calls[0]["run_metadata"]["accepted_task"]["shape"] == "figure_eight"
    assert summary["stages"][0]["stage_name"] == "figure_eight"
    assert summary["stages"][0]["run_name"] == calls[0]["run_name"]


def test_old_local_llm_smoke_config_keeps_adaptive_budget_disabled() -> None:
    """Verify legacy local LLM smoke config loads without adaptive budget settings."""
    settings = llm_curriculum_training.load_llm_curriculum_settings("configs/curricula/curriculum_llm_local_smoke.yaml")

    assert settings.llm_stage_budget.enabled is False
    assert settings.llm_stage_budget.profiles == {"normal": settings.stage_total_timesteps}
    assert settings.proposal_fallback.enabled is False
    llm_curriculum_training.validate_llm_curriculum(settings)
