"""Tests for fixed and randomized task-distribution sampling."""

# ruff: noqa: S101, PLR2004

from __future__ import annotations

from pathlib import Path

import pytest

from src import envs, validation
from src.experiments.evaluation import experiments_evaluation_suites as evaluation_suites
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking


def _base_hover_task() -> dict[str, object]:
    """Return a valid hover base task."""
    return {
        "task_type": "trajectory",
        "shape": "hover_stabilization",
        "duration_sec": 3.0,
        "sample_rate_hz": 10.0,
        "position": [0.0, 0.0, 1.0],
    }


def _base_line_task() -> dict[str, object]:
    """Return a valid line base task."""
    return {
        "task_type": "trajectory",
        "shape": "line",
        "duration_sec": 4.0,
        "sample_rate_hz": 10.0,
        "start_hold_enabled": True,
        "start_hold_sec": 1.0,
        "exclude_start_hold_from_tracking_metrics": True,
        "start": [0.0, 0.0, 1.0],
        "end": [0.5, 0.0, 1.0],
    }


def _settings(**overrides: object) -> envs.task_distribution.TaskDistributionSettings:
    """Build valid randomized settings with caller overrides."""
    values: dict[str, object] = {
        "name": "unit",
        "enabled": True,
        "mode": "randomized",
        "seed": 5,
        "strength": 0.4,
        "sample_on_reset": True,
        "base_task": _base_line_task(),
        "family_weights": {"line": 1.0},
        "variations": {
            "line": {
                "start_xy_radius_m": 0.1,
                "heading_jitter_deg": 20,
                "length_range_m": [0.25, 0.7],
                "z_range_m": [0.9, 1.1],
                "duration_range_sec": [4.0, 7.0],
                "start_hold_range_sec": [0.5, 1.5],
            }
        },
    }
    values.update(overrides)
    return envs.task_distribution.TaskDistributionSettings(**values)  # type: ignore[arg-type]


def test_fixed_distribution_returns_base_task() -> None:
    """Verify fixed-mode sampling is exactly the base task."""
    task = _base_hover_task()
    settings = envs.task_distribution.normalize_fixed_task_to_distribution(task, seed=3)

    sampled = envs.task_distribution.sample_task(settings)

    assert sampled == task
    assert settings.mode == envs.task_distribution.MODE_FIXED
    assert settings.strength == 0.0


def test_strength_zero_returns_base_task() -> None:
    """Verify randomized settings with zero strength keep fixed behavior."""
    settings = _settings(strength=0.0)
    sampled = envs.task_distribution.sample_task(settings)

    assert sampled == _base_line_task()


@pytest.mark.parametrize("strength", [-0.1, 1.1])
def test_invalid_strength_rejected(strength: float) -> None:
    """Verify strength must remain in [0, 1]."""
    with pytest.raises(ValueError, match="strength"):
        _settings(strength=strength)


def test_invalid_family_weights_rejected() -> None:
    """Verify unsupported, negative, and all-zero family weights are rejected."""
    with pytest.raises(ValueError, match="unsupported families"):
        _settings(family_weights={"spiral": 1.0})
    with pytest.raises(ValueError, match="negative"):
        _settings(family_weights={"line": -1.0})
    with pytest.raises(ValueError, match="all zero"):
        _settings(family_weights={"line": 0.0})


def test_weights_normalized_internally() -> None:
    """Verify family weights may be unnormalized in configs."""
    settings = _settings(family_weights={"line": 2.0, "hover_stabilization": 1.0})

    assert settings.family_weights["line"] == pytest.approx(2.0 / 3.0)
    assert settings.family_weights["hover_stabilization"] == pytest.approx(1.0 / 3.0)


def test_hover_variation_stays_within_bounds() -> None:
    """Verify hover randomization respects configured conservative bounds."""
    settings = _settings(
        base_task=_base_hover_task(),
        family_weights={"hover_stabilization": 1.0},
        variations={"hover_stabilization": {"xy_radius_m": 0.2, "z_range_m": [0.8, 1.2], "duration_range_sec": [2.0, 5.0]}},
    )
    sampled = envs.task_distribution.sample_task(settings)

    assert sampled["shape"] == "hover_stabilization"
    assert 0.8 <= sampled["position"][2] <= 1.2  # type: ignore[index]
    assert 2.0 <= sampled["duration_sec"] <= 5.0  # type: ignore[operator]
    assert validation.tasks.validate_task(sampled).is_valid


def test_line_variation_stays_within_bounds() -> None:
    """Verify line randomization emits valid line tasks."""
    sampled = envs.task_distribution.sample_task(_settings())

    assert sampled["shape"] == "line"
    assert validation.tasks.validate_task(sampled).is_valid
    assert sampled["start_hold_enabled"] is True


@pytest.mark.parametrize("family", envs.task_distribution.supported_task_families())
def test_supported_generated_family_passes_existing_validation(family: str) -> None:
    """Verify every supported family can emit an existing valid task schema."""
    settings = envs.task_distribution.load_task_distribution_settings("configs/tasks/task_distribution_tracking_broad.yaml")
    settings = envs.task_distribution.TaskDistributionSettings(
        name=settings.name,
        enabled=True,
        mode="randomized",
        seed=7,
        strength=0.5,
        sample_on_reset=True,
        base_task=settings.base_task,
        family_weights={family: 1.0},
        variations=settings.variations,
        validation_limits=settings.validation_limits,
        config_path=settings.config_path,
    )

    sampled = envs.task_distribution.sample_task(settings)

    assert validation.tasks.validate_task(sampled, limits=settings.validation_limits).is_valid


def test_seeded_sampling_is_reproducible() -> None:
    """Verify identical settings and rank produce identical sample sequences."""
    first = envs.task_distribution.TaskDistributionSampler(_settings(), env_rank=0)
    second = envs.task_distribution.TaskDistributionSampler(_settings(), env_rank=0)

    assert [first.sample_task() for _ in range(3)] == [second.sample_task() for _ in range(3)]


def test_different_env_ranks_get_different_deterministic_sequences() -> None:
    """Verify rank-derived RNG changes sampled task sequence."""
    first = envs.task_distribution.TaskDistributionSampler(_settings(), env_rank=0)
    second = envs.task_distribution.TaskDistributionSampler(_settings(), env_rank=1)

    assert [first.sample_task() for _ in range(3)] != [second.sample_task() for _ in range(3)]


def test_sample_on_reset_false_keeps_stable_task() -> None:
    """Verify run-level sampling is stable when sample_on_reset is false."""
    sampler = envs.task_distribution.TaskDistributionSampler(_settings(sample_on_reset=False), env_rank=0)

    assert sampler.sample_task() == sampler.sample_task()


def test_task_distribution_training_configs_load() -> None:
    """Verify new task-distribution training configs resolve through the PPO loader."""
    for config_path in (
        "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml",
        "configs/training/ppo_tracking_pid_dynprev_net128_m-taskdist_medium.yaml",
        "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium.yaml",
    ):
        settings = ppo_tracking.load_ppo_tracking_settings(config_path)
        assert settings.task_distribution_settings is not None
        assert settings.task_distribution_settings.mode == "randomized"
        assert settings.include_dynamics_observation is True
        assert settings.include_previous_action is True


def test_new_evaluation_suites_load() -> None:
    """Verify variation and broad suites validate every fixed task."""
    variation = evaluation_suites.load_evaluation_suite("configs/evaluation/evaluation_task_suite_variation.yaml")
    broad = evaluation_suites.load_evaluation_suite("configs/evaluation/evaluation_task_suite_broad.yaml")

    assert "line_slow" in variation.task_names
    assert "circle_slow" in broad.task_names
    assert "ellipse_slow" in broad.task_names
    assert "figure_eight_slow" in broad.task_names
    assert all(validation.tasks.validate_task(task.task).is_valid for task in broad.tasks)


def test_runner_scripts_exist() -> None:
    """Verify final runner script paths are present for Bash syntax checks."""
    assert Path("scripts/evaluate_variation_suite.sh").is_file()
    assert Path("scripts/render_run_gifs.sh").is_file()
    assert Path("scripts/experiment_runner_common.sh").is_file()
    assert Path("scripts/experiment_matrix.sh").is_file()
    for lane in range(1, 5):
        assert Path(f"scripts/run_lane_{lane}.sh").is_file()
    assert not Path("scripts/run_overnight_experiments.sh").exists()
    assert not Path("scripts/run_overnight_experiments_parallel.sh").exists()
