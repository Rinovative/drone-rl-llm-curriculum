"""Tests for fixed and randomized task-distribution sampling."""

# ruff: noqa: S101, PLR2004

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

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
                "start_hold_range_sec": [1.0, 1.0],
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


def test_polyline_bootstrap_distribution_samples_l_shape_height_offset() -> None:
    """Verify the manual polyline bootstrap can vary final height gently."""
    settings = envs.task_distribution.load_task_distribution_settings("configs/tasks/task_distribution_polyline_bootstrap_medium.yaml")

    sampled = envs.task_distribution.sample_task(settings)

    assert sampled["shape"] == "polyline"
    points = sampled["points"]
    final_height_offset = points[-1][2] - points[0][2]
    assert final_height_offset != pytest.approx(0.0)
    assert -0.1 <= final_height_offset <= 0.1
    assert validation.tasks.validate_task(sampled, limits=settings.validation_limits).is_valid


def test_basic_training_show_distribution_samples_bounded_episode_variation() -> None:
    """Verify the Direct-PPO basic training show varies per episode but keeps its identity."""
    settings = envs.task_distribution.load_task_distribution_settings("configs/tasks/task_distribution_basic_training_show.yaml")
    first_sampler = envs.task_distribution.TaskDistributionSampler(settings, env_rank=0)
    second_sampler = envs.task_distribution.TaskDistributionSampler(settings, env_rank=0)

    first = first_sampler.sample_task()
    second = first_sampler.sample_task()
    repeated_first = second_sampler.sample_task()

    assert first["shape"] == validation.contracts.SHAPE_BASIC_TRAINING_SHOW
    assert first["training_task_kind"] == "basic_training_show"
    assert first["task_is_show"] is True
    assert first["show_name"] == "basic_training_show"
    assert first["sampled_per_episode"] is True
    assert first["constant_within_episode"] is True
    assert first["variation_enabled"] is True
    assert first["variation_mode"] == "bounded_per_episode"
    assert first["segment_shapes"] == [
        "start_hold",
        "hover_stabilization",
        "horizontal_line",
        "diagonal_line",
        "vertical",
        "ellipse",
        "l_shape",
        "zigzag",
        "final_hold",
    ]
    assert first["meaningful_figure_count"] == 8
    assert first["start_hold_enabled"] is True
    assert first["start_hold_sec"] == pytest.approx(1.0)
    assert first["exclude_start_hold_from_tracking_metrics"] is True
    assert first["final_hold_enabled"] is True
    assert 0.8 <= first["final_hold_sec"] <= 1.2
    assert first["duration_range_sec"][0] < 22.0
    assert "ellipse" in first["segment_shapes"]
    assert "zigzag" in first["segment_shapes"]
    assert first != second
    assert first == repeated_first
    assert validation.tasks.validate_task(first, limits=settings.validation_limits).is_valid
    assert validation.tasks.validate_task(second, limits=settings.validation_limits).is_valid


def _assert_start_hold_policy(task: dict[str, object], *, expected_sec: float = 1.0) -> None:
    """Assert a task uses the active uniform start-hold metric policy."""
    assert task["start_hold_enabled"] is True
    assert float(task["start_hold_sec"]) == pytest.approx(expected_sec)
    assert task["exclude_start_hold_from_tracking_metrics"] is True
    if task.get("shape") == validation.contracts.SHAPE_START_HOLD_THEN_SHORT_LINE:
        assert float(task["hold_duration_sec"]) == pytest.approx(expected_sec)


def _initial_task_z(task: dict[str, object]) -> float | None:
    """Return the encoded initial z coordinate for supported task dictionaries."""
    if isinstance(task.get("position"), list):
        return float(task["position"][2])  # type: ignore[index]
    if isinstance(task.get("start"), list):
        return float(task["start"][2])  # type: ignore[index]
    if isinstance(task.get("points"), list):
        return float(task["points"][0][2])  # type: ignore[index]
    if isinstance(task.get("start_height"), (int, float)):
        return float(task["start_height"])
    if isinstance(task.get("height"), (int, float)):
        return float(task["height"])
    if isinstance(task.get("segments"), list):
        first = task["segments"][0]  # type: ignore[index]
        if isinstance(first, dict) and isinstance(first.get("segment_start"), list):
            return float(first["segment_start"][2])
    return None


def _assert_standard_height_policy(task: dict[str, object]) -> None:
    """Assert a task exposes and follows the standard reference-height metadata."""
    assert "lower_start_height_enabled" not in task
    assert task.get("standard_reference_height_enabled") is True
    assert task.get("start_height_policy") == "standard_reference_1p0m"
    assert task.get("start_hold_reward_policy") == "full_tracking_reward_active_during_uniform_reference_start_hold"
    assert task.get("tracking_reward_starts_after_start_hold") is False
    start_z = _initial_task_z(task)
    assert start_z is not None
    assert 0.9 <= start_z <= 1.1


def test_active_training_distributions_use_standard_height_and_uniform_start_hold_policy() -> None:
    """Verify active sampled training distributions use standard height and uniform hold."""
    for path in sorted(Path("configs/tasks").glob("task_distribution_*.yaml")):
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        settings = envs.task_distribution.load_task_distribution_settings(path)
        _assert_start_hold_policy(settings.base_task)
        _assert_standard_height_policy(settings.base_task)
        representative = payload.get("own_task_representative") if isinstance(payload, dict) else None
        if isinstance(representative, dict):
            _assert_start_hold_policy(representative)
        for family in settings.family_weights:
            family_settings = envs.task_distribution.TaskDistributionSettings(
                name=settings.name,
                enabled=True,
                mode=envs.task_distribution.MODE_RANDOMIZED,
                seed=17,
                strength=settings.strength,
                sample_on_reset=True,
                base_task=settings.base_task,
                family_weights={family: 1.0},
                variations=settings.variations,
                validation_limits=settings.validation_limits,
                config_path=settings.config_path,
            )
            sampled = envs.task_distribution.sample_task(family_settings)
            _assert_start_hold_policy(sampled)
            _assert_standard_height_policy(sampled)
            assert validation.tasks.validate_task(sampled, limits=settings.validation_limits).is_valid


def test_active_curriculum_configs_use_uniform_stage_start_hold_policy() -> None:
    """Verify manual and LLM curriculum stage tasks expose the uniform hold policy."""
    config_paths = [
        *sorted(Path("configs/curricula").glob("curriculum_*_m-taskdist_medium.yaml")),
        *sorted(Path("configs/curricula").glob("llm_curriculum_*_m-taskdist_medium.yaml")),
    ]
    for path in config_paths:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        stages = payload.get("stages") or []
        if isinstance(payload.get("bootstrap"), dict):
            stages = [payload["bootstrap"], *stages]
        for stage in stages:
            for key in ("task", "evaluation_task"):
                task = stage.get(key) if isinstance(stage, dict) else None
                if isinstance(task, dict):
                    _assert_start_hold_policy(task)
            bounds = stage.get("stage_sampling_bounds") if isinstance(stage, dict) else None
            if isinstance(bounds, dict):
                for key in ("start_hold_sec", "hold_duration_sec"):
                    value = bounds.get(key)
                    if isinstance(value, list):
                        assert value == [1.0, 1.0]


def test_altitude_control_families_are_supported_and_registered() -> None:
    """Verify named altitude-control families remain in the sampler catalog."""
    expected = {
        "vertical_up_down",
        "angled_vertical",
        "delayed_altitude_polyline",
        "multi_height_polyline",
    }
    assert expected.issubset(set(envs.task_distribution.supported_task_families()))
    medium = envs.task_distribution.load_task_distribution_settings("configs/tasks/task_distribution_tracking_medium.yaml")
    assert expected.issubset(set(medium.family_weights))


def test_vertical_up_down_distribution_samples_climbs_and_descents() -> None:
    """Verify vertical-up/down tasks include descent as well as climb cases."""
    settings = envs.task_distribution.load_task_distribution_settings("configs/tasks/task_distribution_vertical_up_down_bootstrap_medium.yaml")
    deltas = []
    for seed in range(12):
        seeded = envs.task_distribution.TaskDistributionSettings(
            name=settings.name,
            enabled=True,
            mode=envs.task_distribution.MODE_RANDOMIZED,
            seed=seed,
            strength=settings.strength,
            sample_on_reset=True,
            base_task=settings.base_task,
            family_weights=settings.family_weights,
            variations=settings.variations,
            validation_limits=settings.validation_limits,
            config_path=settings.config_path,
        )
        task = envs.task_distribution.sample_task(seeded)
        deltas.append(float(task["end_height"]) - float(task["start_height"]))
        _assert_standard_height_policy(task)
        assert validation.tasks.validate_task(task, limits=settings.validation_limits).is_valid
    assert any(delta > 0.0 for delta in deltas)
    assert any(delta < 0.0 for delta in deltas)


def test_angled_and_delayed_altitude_distributions_keep_altitude_change_later() -> None:
    """Verify angled and delayed-altitude families validate and preserve their intended z structure."""
    angled_settings = envs.task_distribution.load_task_distribution_settings("configs/tasks/task_distribution_angled_vertical_bootstrap_medium.yaml")
    angled = envs.task_distribution.sample_task(angled_settings)
    assert angled["shape"] == "line"
    assert angled["start"][2] != pytest.approx(angled["end"][2])
    assert angled["start"][:2] != angled["end"][:2]
    _assert_standard_height_policy(angled)
    assert validation.tasks.validate_task(angled, limits=angled_settings.validation_limits).is_valid

    delayed_settings = envs.task_distribution.load_task_distribution_settings(
        "configs/tasks/task_distribution_delayed_altitude_polyline_bootstrap_medium.yaml"
    )
    delayed = envs.task_distribution.sample_task(delayed_settings)
    points = delayed["points"]
    assert points[0][2] == pytest.approx(points[1][2])
    assert any(point[2] != pytest.approx(points[0][2]) for point in points[2:])
    _assert_standard_height_policy(delayed)
    assert validation.tasks.validate_task(delayed, limits=delayed_settings.validation_limits).is_valid


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

    assert sampled["start_hold_enabled"] is True
    assert sampled["start_hold_sec"] > 0.0
    assert sampled["exclude_start_hold_from_tracking_metrics"] is True
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
        "configs/training/ppo_tracking_pid_dynprev_net256_m-taskdist_medium.yaml",
        "configs/training/ppo_tracking_directrpm_dynprev_m-taskdist_medium.yaml",
        "configs/training/ppo_tracking_directrpm_dynprev_net256_m-taskdist_medium.yaml",
    ):
        settings = ppo_tracking.load_ppo_tracking_settings(config_path)
        assert settings.task_distribution_settings is not None
        assert settings.task_distribution_settings.mode == "randomized"
        assert settings.include_dynamics_observation is True
        assert settings.include_previous_action is True


def test_legacy_evaluation_suite_fixtures_load() -> None:
    """Verify legacy variation and broad suite fixtures validate every fixed task."""
    variation = evaluation_suites.load_evaluation_suite("tests/fixtures/configs/evaluation/evaluation_task_suite_variation.yaml")
    broad = evaluation_suites.load_evaluation_suite("tests/fixtures/configs/evaluation/evaluation_task_suite_broad.yaml")

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
