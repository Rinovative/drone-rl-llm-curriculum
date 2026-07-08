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
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking

LANE_ASSIGNMENT = Path("docs/experiments/overnight_lane_assignment.tsv")
MANUAL_CURRICULUM_UNIT_COUNT = 5
LLM_CURRICULUM_UNIT_COUNT = 10
REFERENCE_MEDIUM_TIMESTEPS = 500000
MANUAL_TOTAL_BUDGET_TIMESTEPS = MANUAL_CURRICULUM_UNIT_COUNT * REFERENCE_MEDIUM_TIMESTEPS
LLM_BUDGET_PROFILE_TIMESTEPS = {
    "bootstrap": 750000,
    "short": 375000,
    "normal": 500000,
    "recovery": 625000,
    "extend": 750000,
}
EXPECTED_EXPERIMENT_IDS = {
    "direct_ppo_pid_baseline_medium_seed0",
    "direct_ppo_pid_dynprev_medium_seed0",
    "direct_ppo_pid_dynprev_net128_medium_seed0",
    "direct_ppo_directrpm_dynprev_medium_seed0",
    "direct_ppo_pid_dynprev_m-taskdist_medium_seed0",
    "direct_ppo_pid_dynprev_net128_m-taskdist_medium_seed0",
    "direct_ppo_directrpm_dynprev_m-taskdist_medium_seed0",
    "direct_ppo_directrpm_dynprev_net128_m-taskdist_medium_seed0",
    "direct_ppo_pid_dynprev_m-taskdist_medium_low_lr_seed0",
    "direct_ppo_pid_dynprev_m-taskdist_medium_ent005_seed0",
    "direct_ppo_pid_dynprev_net128_m-taskdist_medium_low_lr_seed0",
    "direct_ppo_pid_dynprev_net128_m-taskdist_medium_ent005_seed0",
    "direct_ppo_directrpm_dynprev_m-taskdist_medium_low_lr_seed0",
    "direct_ppo_directrpm_dynprev_m-taskdist_medium_ent005_seed0",
    "curriculum_manual_pid_dynprev_m-taskdist_medium_seed0",
    "curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0",
    "curriculum_llm_pid_dynprev_m-taskdist_medium_seed0",
    "curriculum_llm_directrpm_dynprev_m-taskdist_medium_seed0",
}


def _assignment_rows() -> list[dict[str, str]]:
    """Return parsed lane assignment rows."""
    return list(csv.DictReader(LANE_ASSIGNMENT.read_text(encoding="utf-8").splitlines(), delimiter="	"))


def test_lane_assignment_contains_exact_approved_matrix() -> None:
    """Verify the lane TSV includes exactly the approved 18 experiments."""
    rows = _assignment_rows()

    assert LANE_ASSIGNMENT.is_file()
    assert {row["experiment_id"] for row in rows} == EXPECTED_EXPERIMENT_IDS
    assert {row["lane"] for row in rows} == {"1", "2", "3", "4"}
    assert all("final" not in row["config_path"] for row in rows)
    assert all("task_distribution_tracking_small" not in row["config_path"] for row in rows)


def test_lane_unit_counts_are_balanced() -> None:
    """Verify each lane has the same inferred unit count."""
    totals: dict[str, int] = {}
    for row in _assignment_rows():
        totals[row["lane"]] = totals.get(row["lane"], 0) + int(row["unit_count"])

    assert totals == {"1": 11, "2": 11, "3": 11, "4": 11}


def test_all_listed_configs_exist_and_match_run_names() -> None:
    """Verify matrix configs load and expose the expected run name where applicable."""
    for row in _assignment_rows():
        config_path = Path(row["config_path"])
        assert config_path.is_file()
        if row["kind"] == "direct_ppo":
            settings = ppo_tracking.load_ppo_tracking_settings(config_path)
            assert settings.run_name == row["expected_run_name"]
            assert settings.total_timesteps == REFERENCE_MEDIUM_TIMESTEPS
        elif row["kind"] == "manual_curriculum":
            settings = manual_training.load_manual_curriculum_settings(config_path)
            manual_training.validate_manual_curriculum(settings)
            reference_settings = ppo_tracking.load_ppo_tracking_settings(settings.reference_medium_config_path)
            assert len(settings.stages) == MANUAL_CURRICULUM_UNIT_COUNT
            assert int(row["unit_count"]) == MANUAL_CURRICULUM_UNIT_COUNT
            assert settings.reference_medium_timesteps == REFERENCE_MEDIUM_TIMESTEPS
            assert reference_settings.total_timesteps == settings.reference_medium_timesteps
            assert settings.stage_budget_multiplier == 1.0
            assert settings.stage_total_timesteps == REFERENCE_MEDIUM_TIMESTEPS
            assert settings.manual_stage_count == MANUAL_CURRICULUM_UNIT_COUNT
            assert settings.manual_total_budget_timesteps == MANUAL_TOTAL_BUDGET_TIMESTEPS
            assert [stage.total_timesteps for stage in settings.stages] == [REFERENCE_MEDIUM_TIMESTEPS] * MANUAL_CURRICULUM_UNIT_COUNT
            assert row["expected_run_name"] == f"curriculum_manual_{settings.curriculum_name.removeprefix('curriculum_manual_')}_seed{settings.seed}"
        elif row["kind"] == "llm_curriculum":
            settings = llm_training.load_llm_curriculum_settings(config_path)
            llm_training.validate_llm_curriculum(settings)
            reference_settings = ppo_tracking.load_ppo_tracking_settings(settings.reference_medium_config_path)
            assert settings.max_stages == LLM_CURRICULUM_UNIT_COUNT
            assert int(row["unit_count"]) == LLM_CURRICULUM_UNIT_COUNT
            assert settings.reference_medium_timesteps == REFERENCE_MEDIUM_TIMESTEPS
            assert reference_settings.total_timesteps == settings.reference_medium_timesteps
            assert settings.llm_stage_budget.enabled is True
            assert settings.llm_stage_budget.profiles == LLM_BUDGET_PROFILE_TIMESTEPS
            assert settings.llm_stage_budget.total_budget_cap_timesteps is not None
            assert settings.llm_stage_budget.total_budget_cap_timesteps >= 10 * REFERENCE_MEDIUM_TIMESTEPS
            assert settings.llm_stage_budget.total_budget_cap_timesteps <= 12 * REFERENCE_MEDIUM_TIMESTEPS
            assert settings.bootstrap_stage is not None
            assert settings.bootstrap_stage.requested_stage_budget_profile == "bootstrap"
            assert row["expected_run_name"] == f"curriculum_llm_{settings.curriculum_name.removeprefix('curriculum_llm_')}_seed{settings.seed}"
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


def test_task_distribution_training_configs_use_medium_distribution_only() -> None:
    """Verify scheduled task-distribution training configs point at the medium config."""
    for row in _assignment_rows():
        if row["kind"] != "direct_ppo":
            continue
        settings = ppo_tracking.load_ppo_tracking_settings(row["config_path"])
        if "taskdist" in row["experiment_id"]:
            assert settings.task_distribution_config_path == Path("configs/tasks/task_distribution_tracking_medium.yaml")
        else:
            assert settings.task_distribution_config_path is None


def test_task_distribution_configs_and_families_validate() -> None:
    """Verify task-distribution configs load and every supported family generates a valid task."""
    for config_path in (
        "configs/tasks/task_distribution_hover_small.yaml",
        "configs/tasks/task_distribution_line_small.yaml",
        "configs/tasks/task_distribution_tracking_small.yaml",
        "configs/tasks/task_distribution_tracking_medium.yaml",
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


def test_evaluation_suites_validate_and_include_supported_broad_tasks() -> None:
    """Verify fixed deterministic evaluation suites load through the suite parser."""
    variation = evaluation_suites.load_evaluation_suite("configs/evaluation/evaluation_task_suite_variation.yaml")
    broad = evaluation_suites.load_evaluation_suite("configs/evaluation/evaluation_task_suite_broad.yaml")

    assert variation.task_names[:6] == ["hover_center", "hover_low", "hover_high", "hover_offset_x", "hover_offset_y", "hover_offset_xy"]
    for task_name in ("polyline_l_shape", "square_slow", "circle_slow", "ellipse_slow", "figure_eight_slow"):
        assert task_name in broad.task_names


def test_runner_and_helper_scripts_have_valid_bash_syntax() -> None:
    """Verify final lane and helper scripts pass Bash syntax checks."""
    scripts = [
        "scripts/run_lane_1.sh",
        "scripts/run_lane_2.sh",
        "scripts/run_lane_3.sh",
        "scripts/run_lane_4.sh",
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


def test_runner_preserves_wandb_and_evaluation_phase_order() -> None:
    """Verify runners do not force W&B off and still run standard, variation, broad, render phases."""
    runner = Path("scripts/experiment_runner_common.sh").read_text(encoding="utf-8")

    assert "WANDB_MODE_OVERRIDE=disabled" not in runner
    assert '--wandb-mode "$WANDB_MODE_OVERRIDE"' in runner
    assert "experiments_cli_evaluate_policy" in runner
    assert "experiments_cli_evaluate_curriculum" in runner
    assert "--model-scope final-stage" in runner
    assert "evaluate_variation_suite.sh" in runner
    assert "render_run_gifs.sh" in runner
    assert runner.index(".eval.log") < runner.index(".variation_eval.log") < runner.index(".broad_eval.log") < runner.index(".render.log")


def test_lane_assignment_uses_updated_curriculum_unit_counts() -> None:
    """Verify documented lane assignment uses 5-stage manual and 10-stage LLM units."""
    rows = _assignment_rows()

    assert {row["unit_count"] for row in rows if row["kind"] == "manual_curriculum"} == {"5"}
    assert {row["unit_count"] for row in rows if row["kind"] == "llm_curriculum"} == {"10"}
    assert {row["unit_count"] for row in rows if row["kind"] == "direct_ppo"} == {"1"}
