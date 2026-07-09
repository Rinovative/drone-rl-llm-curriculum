"""Tests for the medium-screening overnight lane matrix."""

# ruff: noqa: S101

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import yaml

from src import envs, validation
from src.experiments.curriculum import experiments_curriculum_llm_training as llm_training
from src.experiments.curriculum import experiments_curriculum_training as manual_training
from src.experiments.evaluation import experiments_evaluation_suites as evaluation_suites
from src.experiments.rendering import experiments_rendering_scenario as scenario_render
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking

LANE_ASSIGNMENT = Path("docs/experiments/overnight_lane_assignment.tsv")
MANUAL_CURRICULUM_UNIT_COUNT = 5
LLM_CURRICULUM_STAGE_COUNT = 10
LLM_CURRICULUM_UNIT_COUNT = 5
REFERENCE_MEDIUM_TIMESTEPS = 500000
DIRECT_RPM_MIN_RELAXED_RECOVERY_STEPS = 20
MANUAL_TOTAL_BUDGET_TIMESTEPS = MANUAL_CURRICULUM_UNIT_COUNT * REFERENCE_MEDIUM_TIMESTEPS
LLM_TOTAL_BUDGET_TIMESTEPS = MANUAL_TOTAL_BUDGET_TIMESTEPS
BOOTSTRAP_HOVER_DISTRIBUTION_CONFIG = Path("configs/tasks/task_distribution_hover_bootstrap_medium.yaml")
VERTICAL_BOOTSTRAP_DISTRIBUTION_CONFIG = Path("configs/tasks/task_distribution_vertical_bootstrap_medium.yaml")
SHORT_LINE_BOOTSTRAP_DISTRIBUTION_CONFIG = Path("configs/tasks/task_distribution_short_line_bootstrap_medium.yaml")
POLYLINE_BOOTSTRAP_DISTRIBUTION_CONFIG = Path("configs/tasks/task_distribution_polyline_bootstrap_medium.yaml")
TRACKING_MEDIUM_DISTRIBUTION_CONFIG = Path("configs/tasks/task_distribution_tracking_medium.yaml")
BASIC_TRAINING_SHOW_DISTRIBUTION_CONFIG = Path("configs/tasks/task_distribution_basic_training_show.yaml")
BASIC_TRAINING_SHOW_TASK_INDEX = 3
BOOTSTRAP_HOVER_TARGET_BOUNDS = {"x": [-0.5, 0.5], "y": [-0.5, 0.5], "z": [0.7, 1.4]}
MANUAL_STAGE_DISTRIBUTION_CONFIGS = (
    BOOTSTRAP_HOVER_DISTRIBUTION_CONFIG,
    VERTICAL_BOOTSTRAP_DISTRIBUTION_CONFIG,
    SHORT_LINE_BOOTSTRAP_DISTRIBUTION_CONFIG,
    POLYLINE_BOOTSTRAP_DISTRIBUTION_CONFIG,
    TRACKING_MEDIUM_DISTRIBUTION_CONFIG,
)
MANUAL_STAGE_SAMPLED_FAMILIES = (
    "hover_stabilization",
    "takeoff_stabilization",
    "start_hold_then_line",
    "l_shape",
    "tracking_medium",
)
MANUAL_STAGE_SAMPLED_SHAPES = ("hover_stabilization", "vertical", "start_hold_then_short_line", "polyline", "mixed")
LLM_BUDGET_PROFILE_TIMESTEPS = {
    "bootstrap": 500000,
    "short": 175000,
    "normal": 250000,
    "recovery": 325000,
    "extend": 400000,
}
LLM_BUDGET_MULTIPLIERS = {"bootstrap": 1.0, "short": 0.35, "normal": 0.5, "recovery": 0.65, "extend": 0.8}
BASIC_TRAINING_SHOW_DIRECT_PPO_IDS = {
    "direct_ppo_pid_baseline_medium_seed0",
    "direct_ppo_pid_dynprev_medium_seed0",
    "direct_ppo_pid_dynprev_net256_medium_seed0",
    "direct_ppo_directrpm_dynprev_medium_seed0",
}

EXPECTED_EXPERIMENT_IDS = {
    "direct_ppo_pid_baseline_medium_seed0",
    "direct_ppo_pid_dynprev_medium_seed0",
    "direct_ppo_pid_dynprev_net256_medium_seed0",
    "direct_ppo_directrpm_dynprev_medium_seed0",
    "direct_ppo_pid_dynprev_m-taskdist_medium_seed0",
    "direct_ppo_pid_dynprev_net256_m-taskdist_medium_seed0",
    "direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0",
    "direct_ppo_directrpm_dynprev_net256_m-taskdist_medium_seed0",
    "direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0",
    "direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0",
    "direct_ppo_pid_dynprev_net256_m-taskdist_medium_low_lr_seed0",
    "direct_ppo_pid_dynprev_net256_m-taskdist_medium_ent005_seed0",
    "direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0",
    "direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0",
    "curriculum_manual_pid_dynprev_m-taskdist_medium_seed0",
    "curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0",
    "llm_curriculum_pid_dynprev_m-taskdist_medium_seed0",
    "llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0",
}


def _assignment_rows() -> list[dict[str, str]]:
    """Return parsed lane assignment rows."""
    return list(csv.DictReader(LANE_ASSIGNMENT.read_text(encoding="utf-8").splitlines(), delimiter="	"))


def _assert_randomized_distribution_config(config_path: Path) -> envs.task_distribution.TaskDistributionSettings:
    """Verify a distribution config loads, samples per reset, and emits valid tasks."""
    settings = envs.task_distribution.load_task_distribution_settings(config_path)
    assert settings.mode == envs.task_distribution.MODE_RANDOMIZED
    assert settings.sample_on_reset is True
    assert settings.strength > 0.0
    sampled = envs.task_distribution.sample_task(settings)
    assert validation.tasks.validate_task(sampled, limits=settings.validation_limits).is_valid
    return settings


def _assert_manual_medium_curriculum_stage_distributions(settings: manual_training.ManualCurriculumSettings) -> None:
    """Verify the scheduled manual medium curriculum trains on bounded distributions."""
    assert [stage.training_task_distribution_config_path for stage in settings.stages] == list(MANUAL_STAGE_DISTRIBUTION_CONFIGS)
    assert [stage.stage_task_distribution_config_path for stage in settings.stages] == list(MANUAL_STAGE_DISTRIBUTION_CONFIGS)
    assert [stage.task_distribution_config_path for stage in settings.stages] == list(MANUAL_STAGE_DISTRIBUTION_CONFIGS)
    assert [stage.sampled_task_family for stage in settings.stages] == list(MANUAL_STAGE_SAMPLED_FAMILIES)
    assert [stage.sampled_task_shape for stage in settings.stages] == list(MANUAL_STAGE_SAMPLED_SHAPES)
    assert all(stage.evaluation_task == stage.task for stage in settings.stages)

    for _stage, config_path, sampled_shape in zip(settings.stages, MANUAL_STAGE_DISTRIBUTION_CONFIGS, MANUAL_STAGE_SAMPLED_SHAPES, strict=True):
        distribution_settings = _assert_randomized_distribution_config(config_path)
        if sampled_shape != "mixed":
            assert envs.task_distribution.sample_task(distribution_settings)["shape"] == sampled_shape

    stage1, stage2, stage3, stage4, stage5 = settings.stages
    assert stage1.bootstrap_stage_source == "deterministic_config"
    assert stage1.bootstrap_task_shape == "hover_stabilization"
    assert stage1.bootstrap_target_sampling_bounds == BOOTSTRAP_HOVER_TARGET_BOUNDS
    assert stage1.stage_sampling_bounds == {"target_position": BOOTSTRAP_HOVER_TARGET_BOUNDS}
    assert stage2.stage_sampling_bounds["start_height"] == [0.65, 0.9]
    assert stage2.stage_sampling_bounds["end_height"] == [1.1, 1.4]
    assert stage3.stage_sampling_bounds["line_length_m"] == [0.25, 0.55]
    assert stage3.stage_sampling_bounds["direction_angle_deg"] == [-45.0, 45.0]
    assert stage4.stage_sampling_bounds["first_segment_length_m"] == [0.3, 0.6]
    assert stage4.stage_sampling_bounds["second_segment_length_m"] == [0.25, 0.5]
    assert stage4.stage_sampling_bounds["final_height_offset_m"] == [-0.1, 0.1]

    medium_settings = envs.task_distribution.load_task_distribution_settings(stage5.training_task_distribution_config_path)
    assert medium_settings.base_task_shape == "line"
    assert len(medium_settings.family_weights) > 1
    assert set(medium_settings.family_weights).issubset(set(envs.task_distribution.supported_task_families()))
    assert "basic_training_show" not in medium_settings.family_weights


def test_lane_assignment_contains_exact_approved_matrix() -> None:
    """Verify the lane TSV includes exactly the approved 18 experiments."""
    rows = _assignment_rows()

    assert LANE_ASSIGNMENT.is_file()
    assert {row["experiment_id"] for row in rows} == EXPECTED_EXPERIMENT_IDS
    assert {row["lane"] for row in rows} == {"1", "2", "3", "4", "5", "6"}
    assert all("final" not in row["config_path"] for row in rows)
    assert all("task_distribution_tracking_small" not in row["config_path"] for row in rows)


def test_lane_unit_counts_are_balanced() -> None:
    """Verify each lane has the same inferred unit count."""
    totals: dict[str, int] = {}
    for row in _assignment_rows():
        totals[row["lane"]] = totals.get(row["lane"], 0) + int(row["unit_count"])

    assert totals == {"1": 6, "2": 6, "3": 6, "4": 6, "5": 5, "6": 5}


def test_all_listed_configs_exist_and_match_run_names() -> None:
    """Verify matrix configs load and expose the expected run name where applicable."""
    for row in _assignment_rows():
        config_path = Path(row["config_path"])
        assert config_path.is_file()
        if row["kind"] == "direct_ppo":
            settings = ppo_tracking.load_ppo_tracking_settings(config_path)
            assert settings.run_name == row["expected_run_name"]
            assert settings.total_timesteps == REFERENCE_MEDIUM_TIMESTEPS
            assert settings.termination_limits.mode == "relaxed"
            assert settings.diagnostic_limits.mode == "default"
            if settings.action_interface == "direct_rpm":
                assert settings.termination_limits.profile == "direct_rpm_relaxed"
                assert settings.termination_limits.allow_recovery_steps >= DIRECT_RPM_MIN_RELAXED_RECOVERY_STEPS
            else:
                assert settings.termination_limits.profile == "pid_relaxed"
                assert 0 < settings.termination_limits.allow_recovery_steps < DIRECT_RPM_MIN_RELAXED_RECOVERY_STEPS
            assert settings.termination_limits.terminate_on_base_truncation is False
        elif row["kind"] == "manual_curriculum":
            settings = manual_training.load_manual_curriculum_settings(config_path)
            manual_training.validate_manual_curriculum(settings)
            reference_settings = ppo_tracking.load_ppo_tracking_settings(settings.reference_medium_config_path)
            assert reference_settings.termination_limits.mode == "relaxed"
            if reference_settings.action_interface == "direct_rpm":
                assert reference_settings.termination_limits.profile == "direct_rpm_relaxed"
            else:
                assert reference_settings.termination_limits.profile == "pid_relaxed"
            assert len(settings.stages) == MANUAL_CURRICULUM_UNIT_COUNT
            assert int(row["unit_count"]) == MANUAL_CURRICULUM_UNIT_COUNT
            assert settings.reference_medium_timesteps == REFERENCE_MEDIUM_TIMESTEPS
            assert reference_settings.total_timesteps == settings.reference_medium_timesteps
            assert settings.stage_budget_multiplier == 1.0
            assert settings.stage_total_timesteps == REFERENCE_MEDIUM_TIMESTEPS
            assert settings.manual_stage_count == MANUAL_CURRICULUM_UNIT_COUNT
            assert settings.manual_total_budget_timesteps == MANUAL_TOTAL_BUDGET_TIMESTEPS
            assert [stage.total_timesteps for stage in settings.stages] == [REFERENCE_MEDIUM_TIMESTEPS] * MANUAL_CURRICULUM_UNIT_COUNT
            _assert_manual_medium_curriculum_stage_distributions(settings)
            assert row["expected_run_name"] == f"curriculum_manual_{settings.curriculum_name.removeprefix('curriculum_manual_')}_seed{settings.seed}"
        elif row["kind"] == "llm_curriculum":
            settings = llm_training.load_llm_curriculum_settings(config_path)
            llm_training.validate_llm_curriculum(settings)
            reference_settings = ppo_tracking.load_ppo_tracking_settings(settings.reference_medium_config_path)
            assert reference_settings.termination_limits.mode == "relaxed"
            if reference_settings.action_interface == "direct_rpm":
                assert reference_settings.termination_limits.profile == "direct_rpm_relaxed"
            else:
                assert reference_settings.termination_limits.profile == "pid_relaxed"
            assert settings.max_stages == LLM_CURRICULUM_STAGE_COUNT
            assert int(row["unit_count"]) == LLM_CURRICULUM_UNIT_COUNT
            assert settings.reference_medium_timesteps == REFERENCE_MEDIUM_TIMESTEPS
            assert reference_settings.total_timesteps == settings.reference_medium_timesteps
            assert settings.curriculum_name.startswith("llm_curriculum_")
            assert settings.stage_budget_multipliers == LLM_BUDGET_MULTIPLIERS
            assert settings.llm_stage_budget.enabled is True
            assert settings.llm_stage_budget.profiles == LLM_BUDGET_PROFILE_TIMESTEPS
            assert settings.llm_stage_budget.total_budget_cap_timesteps == LLM_TOTAL_BUDGET_TIMESTEPS
            assert settings.llm_stage_budget.min_stage_timesteps == LLM_BUDGET_PROFILE_TIMESTEPS["short"]
            assert settings.llm_stage_budget.max_stage_timesteps == LLM_BUDGET_PROFILE_TIMESTEPS["bootstrap"]
            assert settings.bootstrap_stage is not None
            assert settings.bootstrap_stage.stage_name == "hover_stabilization"
            assert settings.bootstrap_stage.task_shape == "hover_stabilization"
            assert settings.bootstrap_stage.total_timesteps == REFERENCE_MEDIUM_TIMESTEPS
            assert settings.bootstrap_stage.requested_stage_budget_profile == "bootstrap"
            assert settings.bootstrap_stage.task_distribution_id == "bootstrap_randomized_hover_target"
            assert settings.bootstrap_stage.task_distribution_config_path == BOOTSTRAP_HOVER_DISTRIBUTION_CONFIG
            assert settings.bootstrap_stage.bootstrap_stage_source == "deterministic_config"
            assert settings.bootstrap_stage.bootstrap_task_shape == "hover_stabilization"
            assert settings.bootstrap_stage.bootstrap_target_sampling_bounds == BOOTSTRAP_HOVER_TARGET_BOUNDS
            assert row["expected_run_name"] == f"{settings.curriculum_name}_seed{settings.seed}"
        else:
            raise AssertionError(row["kind"])


def test_medium_matrix_future_names_do_not_contain_duplicate_medium() -> None:
    """Verify future config-derived names and W&B config tags avoid duplicated medium labels."""
    rows = _assignment_rows()
    scheduled_paths = [Path(row["config_path"]) for row in rows]

    assert all("medium_medium" not in row["experiment_id"] for row in rows)
    assert all("medium_medium" not in row["expected_run_name"] for row in rows)
    assert all("medium_medium" not in path.as_posix() for path in scheduled_paths)
    assert all("medium_medium" not in f"config:{path.stem}" for path in scheduled_paths)

    renamed_config_files = [*Path("configs/training").glob("*m-taskdist_medium*.yaml"), *Path("configs/curricula").glob("*m-taskdist_medium*.yaml")]
    assert renamed_config_files
    assert all("medium_medium" not in path.name for path in renamed_config_files)
    assert "medium_medium" not in Path("scripts/experiment_matrix.sh").read_text(encoding="utf-8")
    assert "medium_medium" not in LANE_ASSIGNMENT.read_text(encoding="utf-8")


def test_direct_ppo_training_configs_use_intended_task_distribution() -> None:
    """Verify Direct-PPO single-runs use basic_training_show and taskdist variants stay medium."""
    for row in _assignment_rows():
        if row["kind"] != "direct_ppo":
            continue
        settings = ppo_tracking.load_ppo_tracking_settings(row["config_path"])
        if "taskdist" in row["experiment_id"]:
            assert settings.task_distribution_config_path == TRACKING_MEDIUM_DISTRIBUTION_CONFIG
            assert settings.task_index == 0
        elif row["experiment_id"] in BASIC_TRAINING_SHOW_DIRECT_PPO_IDS:
            assert settings.task_distribution_config_path == BASIC_TRAINING_SHOW_DISTRIBUTION_CONFIG
            assert settings.task_index == BASIC_TRAINING_SHOW_TASK_INDEX
            task_config = yaml.safe_load(settings.task_config_path.read_text(encoding="utf-8"))
            task = task_config["tasks"][settings.task_index]
            assert task["shape"] == validation.contracts.SHAPE_BASIC_TRAINING_SHOW
            assert task["show_name"] == "basic_training_show"
            assert task["task_is_show"] is True
        else:
            assert settings.task_distribution_config_path is None


def test_task_distribution_configs_and_families_validate() -> None:
    """Verify task-distribution configs load and every supported family generates a valid task."""
    for config_path in (
        "configs/tasks/task_distribution_hover_small.yaml",
        "configs/tasks/task_distribution_line_small.yaml",
        "configs/tasks/task_distribution_tracking_small.yaml",
        "configs/tasks/task_distribution_tracking_medium.yaml",
        "configs/tasks/task_distribution_hover_bootstrap_medium.yaml",
        "configs/tasks/task_distribution_vertical_bootstrap_medium.yaml",
        "configs/tasks/task_distribution_short_line_bootstrap_medium.yaml",
        "configs/tasks/task_distribution_polyline_bootstrap_medium.yaml",
        "configs/tasks/task_distribution_basic_training_show.yaml",
        "configs/tasks/task_distribution_tracking_broad.yaml",
    ):
        settings = envs.task_distribution.load_task_distribution_settings(config_path)
        assert settings.base_task

    settings = envs.task_distribution.load_task_distribution_settings("configs/tasks/task_distribution_tracking_broad.yaml")
    for family in envs.task_distribution.supported_task_families():
        single_family = envs.task_distribution.TaskDistributionSettings(
            name=settings.name,
            enabled=True,
            mode="randomized",
            seed=13,
            strength=0.5,
            sample_on_reset=True,
            base_task=settings.base_task,
            family_weights={family: 1.0},
            variations=settings.variations,
            validation_limits=settings.validation_limits,
            config_path=settings.config_path,
        )
        task = envs.task_distribution.sample_task(single_family)
        assert validation.tasks.validate_task(task, limits=single_family.validation_limits).is_valid
    assert envs.task_distribution.unsupported_requested_task_families() == ()


def test_bootstrap_hover_distribution_uses_fixed_medium_bounds() -> None:
    """Verify LLM Stage 1 uses a randomized hover-target bootstrap, not broad tracking."""
    settings = envs.task_distribution.load_task_distribution_settings(BOOTSTRAP_HOVER_DISTRIBUTION_CONFIG)

    assert settings.name == "bootstrap_randomized_hover_target"
    assert settings.strength == 1.0
    assert settings.family_weights == {"hover_stabilization": 1.0}
    hover_variation = settings.variations["hover_stabilization"]
    assert hover_variation["x_range_m"] == BOOTSTRAP_HOVER_TARGET_BOUNDS["x"]
    assert hover_variation["y_range_m"] == BOOTSTRAP_HOVER_TARGET_BOUNDS["y"]
    assert hover_variation["z_range_m"] == BOOTSTRAP_HOVER_TARGET_BOUNDS["z"]


def test_manual_and_llm_stage_one_share_randomized_hover_bootstrap() -> None:
    """Verify manual and LLM curricula share the same randomized hover bootstrap."""
    manual_configs = (
        "configs/curricula/curriculum_pid_dynprev_m-taskdist_medium.yaml",
        "configs/curricula/curriculum_directrpm_dynprev_m-taskdist_medium.yaml",
    )
    llm_configs = (
        "configs/curricula/llm_curriculum_pid_dynprev_m-taskdist_medium.yaml",
        "configs/curricula/llm_curriculum_directrpm_dynprev_m-taskdist_medium.yaml",
    )

    manual_stage1_paths = {
        manual_training.load_manual_curriculum_settings(config).stages[0].training_task_distribution_config_path for config in manual_configs
    }
    llm_stage1_paths = {llm_training.load_llm_curriculum_settings(config).bootstrap_stage.task_distribution_config_path for config in llm_configs}

    assert manual_stage1_paths == {BOOTSTRAP_HOVER_DISTRIBUTION_CONFIG}
    assert llm_stage1_paths == {BOOTSTRAP_HOVER_DISTRIBUTION_CONFIG}
    for config in manual_configs:
        stage1 = manual_training.load_manual_curriculum_settings(config).stages[0]
        assert stage1.bootstrap_target_sampling_bounds == BOOTSTRAP_HOVER_TARGET_BOUNDS
        assert stage1.total_timesteps == REFERENCE_MEDIUM_TIMESTEPS
    for config in llm_configs:
        bootstrap = llm_training.load_llm_curriculum_settings(config).bootstrap_stage
        assert bootstrap.bootstrap_target_sampling_bounds == BOOTSTRAP_HOVER_TARGET_BOUNDS
        assert bootstrap.total_timesteps == REFERENCE_MEDIUM_TIMESTEPS


def test_generalization_suite_and_standard_scenarios_validate() -> None:
    """Verify simplified deterministic evaluation assets load through their parsers."""
    generalization = evaluation_suites.load_evaluation_suite("configs/evaluation/generalization_eval_suite.yaml")

    assert generalization.task_names == [
        "hover_center",
        "vertical_basic",
        "line_basic",
        "diagonal_line_basic",
        "short_line_start_hold",
        "polyline_l_basic",
        "circle_basic",
        "ellipse_basic",
        "figure_eight_basic",
    ]
    durations = []
    for config_path in (
        "configs/evaluation/scenarios/show_easy.yaml",
        "configs/evaluation/scenarios/show_medium.yaml",
        "configs/evaluation/scenarios/show_hard.yaml",
    ):
        composition = scenario_render.compose_scenario_reference(scenario_render.load_scenario_render_settings(config_path))
        durations.append(composition.scenario_duration_sec)
        assert composition.final_hold_steps > 0
    assert durations == sorted(durations)


def test_runner_and_helper_scripts_have_valid_bash_syntax() -> None:
    """Verify final lane and helper scripts pass Bash syntax checks."""
    scripts = [
        "scripts/run_lane_1.sh",
        "scripts/run_lane_2.sh",
        "scripts/run_lane_3.sh",
        "scripts/run_lane_4.sh",
        "scripts/run_lane_5.sh",
        "scripts/run_lane_6.sh",
        "scripts/experiment_runner_common.sh",
        "scripts/experiment_matrix.sh",
        "scripts/evaluate_variation_suite.sh",
        "scripts/render_run_gifs.sh",
    ]
    bash_path = shutil.which("bash")
    assert bash_path is not None
    for script in scripts:
        subprocess.run([bash_path, "-n", script], check=True)  # noqa: S603 - scripts are static repo paths.


def test_local_llm_smoke_config_still_loads() -> None:
    """Verify the old local LLM smoke config remains valid but unscheduled."""
    settings = llm_training.load_llm_curriculum_settings("configs/curricula/curriculum_llm_local_smoke.yaml")

    assert settings.curriculum_name == "curriculum_llm_local_smoke"
    assert settings.llm_provider == "openai_compatible"
    llm_training.validate_llm_curriculum(settings)
    assert "curriculum_llm_local_smoke" not in {row["experiment_id"] for row in _assignment_rows()}


def test_no_source_controlled_output_dirs_in_matrix_configs() -> None:
    """Verify matrix configs do not redirect generated outputs into tracked config/docs paths."""
    forbidden_prefixes = ("docs/", "configs/", "src/", "tests/")
    for row in _assignment_rows():
        payload = yaml.safe_load(Path(row["config_path"]).read_text(encoding="utf-8"))
        for key in ("output_dir", "model_dir", "artifact_root", "wandb_dir"):
            value = payload.get(key) if isinstance(payload, dict) else None
            if isinstance(value, str):
                assert not value.startswith(forbidden_prefixes)


def test_runner_preserves_wandb_and_simplified_evaluation_phase_order() -> None:
    """Verify runners do not force W&B off and run only standard evaluation plus render status."""
    runner = Path("scripts/experiment_runner_common.sh").read_text(encoding="utf-8")

    assert "WANDB_MODE_OVERRIDE=disabled" not in runner
    assert '--wandb-mode "$WANDB_MODE_OVERRIDE"' in runner
    assert "experiments_cli_evaluate_policy" in runner
    assert "experiments_cli_evaluate_curriculum" in runner
    assert "--model-scope final-stage" in runner
    assert "evaluate_variation_suite.sh" not in runner
    assert "variation_eval" not in runner
    assert "broad_eval" not in runner
    assert "render_run_gifs.sh" in runner
    assert runner.index(".eval.log") < runner.index(".render.log")


def test_lane_assignment_uses_updated_curriculum_unit_counts() -> None:
    """Verify documented lane assignment uses 5-stage manual and 5-unit LLM curricula."""
    rows = _assignment_rows()

    assert {row["unit_count"] for row in rows if row["kind"] == "manual_curriculum"} == {"5"}
    assert {row["unit_count"] for row in rows if row["kind"] == "llm_curriculum"} == {"5"}
    assert {row["unit_count"] for row in rows if row["kind"] == "direct_ppo"} == {"1"}
    assert not any(row["experiment_id"].startswith("curriculum_llm_") for row in rows)
