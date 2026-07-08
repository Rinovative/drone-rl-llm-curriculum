"""Tests for clean experiments package structure."""

# ruff: noqa: S101

from __future__ import annotations

import importlib
import subprocess
import sys

import pytest

from src import experiments

CANONICAL_MODULE_SYMBOLS = [
    ("src.experiments.training.experiments_training_ppo_config", "PPOConfig"),
    ("src.experiments.training.experiments_training_ppo_tracking", "PPOTrackingSmokeSettings"),
    ("src.experiments.training.experiments_training_smoke", "TrainingSmokeSettings"),
    ("src.experiments.evaluation.experiments_evaluation_policy", "PolicyEvaluationSpec"),
    ("src.experiments.curriculum.experiments_curriculum_validation", "summarize_config_tasks"),
    ("src.experiments.curriculum.experiments_curriculum_training", "ManualCurriculumSettings"),
    ("src.experiments.curriculum.experiments_curriculum_evaluation", "CurriculumEvaluationResult"),
    ("src.experiments.rendering.experiments_rendering_policy", "PolicyRenderSettings"),
    ("src.experiments.rendering.experiments_rendering_scenario", "ScenarioRenderSettings"),
    ("src.experiments.rendering.experiments_rendering_smoke", "RenderSmokeSettings"),
]

REMOVED_ROOT_MODULES = [
    "src.experiments.cli_train_tracking",
    "src.experiments.cli_train_curriculum",
    "src.experiments.cli_evaluate_curriculum",
    "src.experiments.cli_render_policy",
    "src.experiments.cli_render_scenario",
    "src.experiments.cli_render_smoke",
    "src.experiments.cli_training_smoke",
    "src.experiments.cli_mvp",
    "src.experiments.experiments_ppo_config",
    "src.experiments.experiments_ppo_tracking",
    "src.experiments.experiments_training_smoke",
    "src.experiments.experiments_policy_evaluation",
    "src.experiments.experiments_curriculum",
    "src.experiments.experiments_curriculum_training",
    "src.experiments.experiments_curriculum_evaluation",
    "src.experiments.experiments_policy_render",
    "src.experiments.experiments_scenario_render",
    "src.experiments.experiments_render_smoke",
]

CANONICAL_CLI_MODULE_PATHS = [
    "src.experiments.cli.experiments_cli_train_tracking",
    "src.experiments.cli.experiments_cli_train_curriculum",
    "src.experiments.cli.experiments_cli_evaluate_curriculum",
    "src.experiments.cli.experiments_cli_render_policy",
    "src.experiments.cli.experiments_cli_render_scenario",
    "src.experiments.cli.experiments_cli_render_smoke",
    "src.experiments.cli.experiments_cli_training_smoke",
    "src.experiments.cli.experiments_cli_mvp",
]


def test_experiments_package_exposes_only_static_subpackages() -> None:
    """Verify the root package does not expose old compatibility aliases."""
    assert experiments.__all__ == ["cli", "config", "curriculum", "evaluation", "rendering", "training"]
    assert "__getattr__" not in vars(experiments)
    assert not hasattr(experiments, "ppo_tracking")
    assert not hasattr(experiments, "policy_render")
    assert not hasattr(experiments, "curriculum_training")


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
