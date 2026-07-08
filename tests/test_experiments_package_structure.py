"""Tests for clean experiments package structure."""

# ruff: noqa: S101

from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

import pytest

from src import experiments

CANONICAL_MODULE_SYMBOLS = [
    ("src.experiments.training.experiments_training_ppo_config", "PPOConfig"),
    ("src.experiments.training.experiments_training_ppo_tracking", "PPOTrackingSmokeSettings"),
    ("src.experiments.training.experiments_training_smoke", "TrainingSmokeSettings"),
    ("src.experiments.evaluation.experiments_evaluation_policy", "PolicyEvaluationSpec"),
    ("src.experiments.curriculum.experiments_curriculum_validation", "summarize_config_tasks"),
    ("src.experiments.curriculum.experiments_curriculum_training", "ManualCurriculumSettings"),
    ("src.experiments.curriculum.experiments_curriculum_llm_training", "LLMCurriculumSettings"),
    ("src.experiments.curriculum.experiments_curriculum_evaluation", "CurriculumEvaluationResult"),
    ("src.experiments.rendering.experiments_rendering_policy", "PolicyRenderSettings"),
    ("src.experiments.rendering.experiments_rendering_scenario", "ScenarioRenderSettings"),
    ("src.experiments.rendering.experiments_rendering_smoke", "RenderSmokeSettings"),
]

REMOVED_ROOT_MODULES = [
    f"src.experiments.{module_name}"
    for module_name in (
        "cli_train_tracking",
        "cli_train_curriculum",
        "cli_train_llm_curriculum",
        "cli_evaluate_curriculum",
        "cli_render_policy",
        "cli_render_scenario",
        "cli_render_smoke",
        "cli_training_smoke",
        "cli_mvp",
        "experiments_" + "ppo_config",
        "experiments_" + "ppo_tracking",
        "experiments_" + "training_smoke",
        "experiments_" + "policy_evaluation",
        "experiments_" + "curriculum",
        "experiments_" + "curriculum_training",
        "experiments_" + "curriculum_llm_training",
        "experiments_" + "curriculum_evaluation",
        "experiments_" + "policy_render",
        "experiments_" + "scenario_render",
        "experiments_" + "render_smoke",
    )
]

CANONICAL_CLI_MODULE_PATHS = [
    "src.experiments.cli.experiments_cli_train_tracking",
    "src.experiments.cli.experiments_cli_train_curriculum",
    "src.experiments.cli.experiments_cli_train_llm_curriculum",
    "src.experiments.cli.experiments_cli_evaluate_curriculum",
    "src.experiments.cli.experiments_cli_evaluate_policy",
    "src.experiments.cli.experiments_cli_render_policy",
    "src.experiments.cli.experiments_cli_render_scenario",
    "src.experiments.cli.experiments_cli_render_smoke",
    "src.experiments.cli.experiments_cli_training_smoke",
    "src.experiments.cli.experiments_cli_mvp",
]

THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "TORCH_NUM_THREADS",
)


def test_experiments_package_exposes_only_static_subpackages() -> None:
    """Verify the root package does not expose old compatibility aliases."""
    assert experiments.__all__ == ["cli", "config", "curriculum", "evaluation", "rendering", "training"]
    assert "__getattr__" not in vars(experiments)
    assert not hasattr(experiments, "ppo_tracking")
    assert not hasattr(experiments, "policy_render")
    assert not hasattr(experiments, "curriculum_training")


def test_docker_job_usage_points_at_canonical_train_tracking_cli() -> None:
    """Verify the Docker job helper no longer advertises the removed root CLI path."""
    script = Path("scripts/docker_job.sh").read_text(encoding="utf-8")

    assert "src/experiments/cli/experiments_cli_train_tracking.py" in script
    assert "src/experiments/cli_train_tracking.py" not in script


def test_docker_paths_set_overridable_cpu_thread_defaults() -> None:
    """Verify Docker entry paths set safe CPU thread defaults without requiring Docker."""
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")
    dev_script = Path("scripts/docker_dev.sh").read_text(encoding="utf-8")
    run_script = Path("scripts/_docker_run.sh").read_text(encoding="utf-8")

    assert '"${THREAD_ENV_ARGS[@]}"' in dev_script
    assert '"${THREAD_ENV_ARGS[@]}"' in run_script

    for env_var in THREAD_ENV_VARS:
        assert f"ENV {env_var}=1" in dockerfile
        assert f'"{env_var}=${{{env_var}:-1}}"' in dev_script
        assert f'"{env_var}=${{{env_var}:-1}}"' in run_script


@pytest.mark.parametrize(("module_path", "symbol"), CANONICAL_MODULE_SYMBOLS)
def test_canonical_subpackage_modules_import(module_path: str, symbol: str) -> None:
    """Verify canonical implementation module paths are importable."""
    module = importlib.import_module(module_path)

    assert getattr(module, symbol) is not None


@pytest.mark.parametrize("module_path", REMOVED_ROOT_MODULES)
def test_old_root_module_paths_are_removed(module_path: str) -> None:
    """Verify old root-level wrappers are not import-compatible."""
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_path)


@pytest.mark.parametrize("module_path", CANONICAL_CLI_MODULE_PATHS)
def test_canonical_cli_module_help_paths_work(module_path: str) -> None:
    """Verify canonical CLI module paths expose argparse help without running workflows."""
    completed = subprocess.run(  # noqa: S603
        [sys.executable, "-m", module_path, "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "usage:" in completed.stdout
