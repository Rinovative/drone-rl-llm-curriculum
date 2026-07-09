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
DIRECT_RPM_DELTA_SCALE = 0.05
MAX_DIRECT_RPM_SMOKE_TIMESTEPS = 4096
POLICY_NET_ARCH = [256, 256]
EXPECTED_MEDIUM_TIMESTEPS = 500000
EXPECTED_MEDIUM_NUM_ENVS = 4
EXPECTED_MANUAL_STAGE_COUNT = 5


def test_smoke_config_loads_and_contains_valid_tasks() -> None:
    """Verify the smoke config loads and its tasks pass deterministic validation."""
    config = experiments_config.load_experiment_config("tests/fixtures/configs/smoke/trajectory_validation.yaml")

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


def test_active_direct_ppo_training_configs_use_representative_tasks_and_nested_ppo() -> None:
    """Verify active Direct-PPO configs use representative tasks plus explicit training distributions."""
    expected = {
        "configs/training/ppo_tracking_pid_dynprev_basic_show.yaml": {
            "run_name": "direct_ppo_pid_dynprev_basic_show_seed0",
            "task_index": 3,
            "task_distribution_config_path": "configs/tasks/task_distribution_basic_training_show.yaml",
            "action_interface": "pid_position",
            "include_dynamics_observation": True,
            "include_previous_action": True,
        },
        "configs/training/ppo_tracking_directrpm_dynprev_basic_show.yaml": {
            "run_name": "direct_ppo_directrpm_dynprev_basic_show_seed0",
            "task_index": 3,
            "task_distribution_config_path": "configs/tasks/task_distribution_basic_training_show.yaml",
            "action_interface": "direct_rpm",
            "include_dynamics_observation": True,
            "include_previous_action": True,
        },
        "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml": {
            "run_name": "direct_ppo_pid_dynprev_m-taskdist_medium_seed0",
            "task_index": 0,
            "task_distribution_config_path": "configs/tasks/task_distribution_tracking_medium.yaml",
            "action_interface": "pid_position",
            "include_dynamics_observation": True,
            "include_previous_action": True,
        },
        "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium.yaml": {
            "run_name": "direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0",
            "task_index": 0,
            "task_distribution_config_path": "configs/tasks/task_distribution_tracking_medium.yaml",
            "action_interface": "direct_rpm",
            "include_dynamics_observation": True,
            "include_previous_action": True,
        },
    }

    for config_path, values in expected.items():
        config = experiments_config.load_experiment_config(config_path)
        flat_ppo_keys = set(config) & set(ppo_config.PPO_CONFIG_KEYS)

        assert flat_ppo_keys == set()
        assert config["task_config_path"] == "configs/training/ppo_tracking_representative_tasks.yaml"
        assert config["task_distribution_config_path"] == values["task_distribution_config_path"]
        assert config["run_name"] == values["run_name"]
        assert config["total_timesteps"] == EXPECTED_MEDIUM_TIMESTEPS
        assert config["num_envs"] == EXPECTED_MEDIUM_NUM_ENVS
        assert config["task_index"] == values["task_index"]
        assert config["action_interface"] == values["action_interface"]
        assert config["include_dynamics_observation"] is values["include_dynamics_observation"]
        assert config["include_previous_action"] is values["include_previous_action"]
        assert config["wandb_mode"] == "auto"

        settings = ppo_tracking.load_ppo_tracking_settings(config_path)
        assert settings.run_name == values["run_name"]
        assert settings.num_envs == EXPECTED_MEDIUM_NUM_ENVS
        assert settings.action_interface == values["action_interface"]
        assert settings.task_config_path == Path("configs/training/ppo_tracking_representative_tasks.yaml")
        assert settings.task_distribution_config_path == Path(values["task_distribution_config_path"])
        assert settings.include_dynamics_observation is values["include_dynamics_observation"]
        assert settings.include_previous_action is values["include_previous_action"]
        assert settings.ppo_config.to_dict() == {**config["ppo"], "policy_kwargs": ppo_config.default_policy_kwargs()}


def test_direct_rpm_smoke_config_is_explicit_and_safe() -> None:
    """Verify the direct-RPM smoke config is opt-in, tiny, and W&B-disabled."""
    config_path = "tests/fixtures/configs/training/ppo_tracking_direct_rpm_smoke.yaml"
    config = experiments_config.load_experiment_config(config_path)
    settings = ppo_tracking.load_ppo_tracking_settings(config_path)

    assert config["task_config_path"] == "configs/training/ppo_tracking_representative_tasks.yaml"
    assert config["action_interface"] == "direct_rpm"
    assert config["normalize_actions"] is True
    assert config["include_dynamics_observation"] is True
    assert config["include_previous_action"] is True
    assert config["rpm_delta_scale"] == DIRECT_RPM_DELTA_SCALE
    assert config["wandb_mode"] == "disabled"
    assert config["num_envs"] == 1
    assert config["total_timesteps"] <= MAX_DIRECT_RPM_SMOKE_TIMESTEPS
    assert settings.action_interface == "direct_rpm"
    assert settings.include_dynamics_observation is True
    assert settings.include_previous_action is True
    assert settings.rpm_delta_scale == DIRECT_RPM_DELTA_SCALE


def test_dynamics_smoke_config_is_explicit_and_safe() -> None:
    """Verify the optional PID dynamics smoke config is tiny and W&B-disabled."""
    config_path = "tests/fixtures/configs/training/ppo_tracking_dynamics_smoke.yaml"
    config = experiments_config.load_experiment_config(config_path)
    settings = ppo_tracking.load_ppo_tracking_settings(config_path)

    assert config["action_interface"] == "pid_position"
    assert config["include_dynamics_observation"] is True
    assert config["include_previous_action"] is True
    assert config["wandb_mode"] == "disabled"
    assert config["num_envs"] == 1
    assert config["total_timesteps"] <= MAX_DIRECT_RPM_SMOKE_TIMESTEPS
    assert settings.action_interface == "pid_position"
    assert settings.include_dynamics_observation is True
    assert settings.include_previous_action is True
    assert settings.ppo_config.policy_kwargs == ppo_config.default_policy_kwargs()


def test_dynamics_medium_net_smoke_config_is_explicit_and_safe() -> None:
    """Verify the optional larger-network config is still smoke-sized."""
    config_path = "tests/fixtures/configs/training/ppo_tracking_dynamics_medium_net_smoke.yaml"
    config = experiments_config.load_experiment_config(config_path)
    settings = ppo_tracking.load_ppo_tracking_settings(config_path)

    assert config["action_interface"] == "pid_position"
    assert config["include_dynamics_observation"] is True
    assert config["include_previous_action"] is True
    assert config["wandb_mode"] == "disabled"
    assert config["num_envs"] == 1
    assert config["total_timesteps"] <= MAX_DIRECT_RPM_SMOKE_TIMESTEPS
    assert config["ppo"]["policy_kwargs"] == {"net_arch": POLICY_NET_ARCH}
    assert settings.ppo_config.policy_kwargs == {"net_arch": POLICY_NET_ARCH}


def test_representative_task_source_is_labelled_and_not_training_like() -> None:
    """Verify representative tasks validate and do not masquerade as training distributions."""
    task_config = experiments_config.load_experiment_config("configs/training/ppo_tracking_representative_tasks.yaml")
    limits = validation.tasks.ValidationLimits(**task_config["validation_limits"])

    assert task_config["task_config_role"] == "representative_eval_only"
    assert task_config["representative_task_config_path"] == "configs/training/ppo_tracking_representative_tasks.yaml"
    assert task_config["training_task_distribution_config_paths"] == {
        "basic_training_show": "configs/tasks/task_distribution_basic_training_show.yaml",
        "tracking_medium": "configs/tasks/task_distribution_tracking_medium.yaml",
    }
    assert [task["task_name"] for task in task_config["tasks"]] == [
        "line_basic",
        "line_long_final",
        "line_diagonal_validation",
        "basic_training_show",
    ]
    for task in task_config["tasks"]:
        result = validation.tasks.validate_task(task, limits=limits)
        assert result.is_valid, result.messages
        assert result.trajectory is not None

    basic_show = task_config["tasks"][3]
    assert basic_show["task_is_show"] is True
    assert basic_show["show_name"] == "basic_training_show"
    for removed_key in (
        "task_is_distribution",
        "sampled_per_episode",
        "variation_enabled",
        "variation_mode",
        "requested_task_family",
        "accepted_task_family",
        "repair_was_applied",
    ):
        assert removed_key not in basic_show


def test_active_manual_curriculum_configs_are_canonical_and_valid() -> None:
    """Verify active manual curricula are the current five-stage task-distribution curricula."""
    expected = {
        "configs/curricula/curriculum_pid_dynprev_m-taskdist_medium.yaml": {
            "name": "curriculum_manual_pid_dynprev_m-taskdist_medium",
            "base": "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml",
        },
        "configs/curricula/curriculum_directrpm_dynprev_m-taskdist_medium.yaml": {
            "name": "curriculum_manual_directrpm_dynprev_m-taskdist_medium",
            "base": "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium.yaml",
        },
    }
    stage_distribution_paths = [
        Path("configs/tasks/task_distribution_hover_bootstrap_medium.yaml"),
        Path("configs/tasks/task_distribution_vertical_bootstrap_medium.yaml"),
        Path("configs/tasks/task_distribution_short_line_bootstrap_medium.yaml"),
        Path("configs/tasks/task_distribution_polyline_bootstrap_medium.yaml"),
        Path("configs/tasks/task_distribution_tracking_medium.yaml"),
    ]

    for config_path, values in expected.items():
        settings = curriculum_training.load_manual_curriculum_settings(config_path)

        assert settings.curriculum_name == values["name"]
        assert settings.base_training_config == Path(values["base"])
        assert settings.wandb_mode == "auto"
        assert settings.manual_stage_count == EXPECTED_MANUAL_STAGE_COUNT
        assert [stage.total_timesteps for stage in settings.stages] == [EXPECTED_MEDIUM_TIMESTEPS] * EXPECTED_MANUAL_STAGE_COUNT
        assert [stage.training_task_distribution_config_path for stage in settings.stages] == stage_distribution_paths
        assert [stage.stage_name for stage in settings.stages] == [
            "hover_stabilization",
            "vertical_low_high",
            "start_hold_then_short_line",
            "polyline_l_shape",
            "medium_tracking",
        ]
        curriculum_training.validate_manual_curriculum(settings)


def test_stale_legacy_configs_and_half_smoke_folder_are_absent() -> None:
    """Verify deleted legacy configs do not remain in the active config tree."""
    absent_paths = (
        "configs/training/ppo_tracking_final.yaml",
        "configs/training/ppo_tracking_medium.yaml",
        "configs/training/ppo_tracking_smoke.yaml",
        "configs/training/ppo_tracking_tasks.yaml",
        "configs/training/ppo_tracking_pid_baseline_medium.yaml",
        "configs/training/ppo_tracking_pid_dynprev_medium.yaml",
        "configs/training/ppo_tracking_directrpm_dynprev_medium.yaml",
        "configs/training/ppo_tracking_pid_dynprev_net128_small_m-taskdist_medium.yaml",
        "configs/training/ppo_tracking_pid_dynprev_net512_large_m-taskdist_medium.yaml",
        "configs/training/ppo_tracking_directrpm_dynprev_net512_large_m-taskdist_medium.yaml",
        "configs/curricula/curriculum_manual_line_final.yaml",
        "configs/curricula/curriculum_manual_line_medium.yaml",
        "configs/curricula/curriculum_manual_line_smoke.yaml",
        "configs/curricula/curriculum_llm_smoke.yaml",
        "configs/curricula/curriculum_llm_local_smoke.yaml",
        "configs/smoke",
        "configs/scenarios",
    )

    for path in absent_paths:
        assert not Path(path).exists(), path


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
