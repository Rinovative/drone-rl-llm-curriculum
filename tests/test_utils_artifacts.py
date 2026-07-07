"""Tests for category-scoped artifact path helpers."""

# ruff: noqa: S101

from __future__ import annotations

from pathlib import Path

import pytest

from src import utils


def test_storage_root_defaults_to_storage(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify storage root defaults to the repository storage path."""
    monkeypatch.delenv("STORAGE_ROOT", raising=False)

    assert utils.artifacts.get_storage_root() == Path("storage").resolve(strict=False)


def test_storage_root_respects_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify STORAGE_ROOT controls all category-scoped artifact helpers."""
    storage_root = tmp_path / "external_storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))

    assert utils.artifacts.get_training_run_dir("ppo_tracking_smoke") == storage_root / "training_runs" / "ppo_tracking_smoke"
    assert utils.artifacts.get_evaluation_run_dir("eval_example_on_hover") == storage_root / "evaluation_runs" / "eval_example_on_hover"
    assert utils.artifacts.get_training_metrics_dir("ppo_tracking_smoke") == storage_root / "training_runs" / "ppo_tracking_smoke" / "metrics"
    assert (
        utils.artifacts.get_evaluation_renders_dir("eval_example_on_hover") == storage_root / "evaluation_runs" / "eval_example_on_hover" / "renders"
    )
    assert utils.artifacts.get_evaluation_traces_dir("eval_example_on_hover") == storage_root / "evaluation_runs" / "eval_example_on_hover" / "traces"
    assert utils.artifacts.get_training_models_dir("ppo_tracking_smoke") == storage_root / "training_runs" / "ppo_tracking_smoke" / "models"
    assert utils.artifacts.get_evaluation_plots_dir("eval_example_on_hover") == storage_root / "evaluation_runs" / "eval_example_on_hover" / "plots"
    assert utils.artifacts.get_training_logs_dir("ppo_tracking_smoke") == storage_root / "training_runs" / "ppo_tracking_smoke" / "logs"
    assert utils.artifacts.get_training_diagnostics_dir("ppo_tracking_smoke") == storage_root / "training_runs" / "ppo_tracking_smoke" / "diagnostics"
    assert utils.artifacts.get_training_wandb_dir("ppo_tracking_smoke") == storage_root / "training_runs" / "ppo_tracking_smoke" / "wandb"


def test_ensure_category_run_dirs_create_new_layouts(tmp_path: Path) -> None:
    """Verify new training, evaluation, and comparison report trees are created explicitly."""
    training_paths = utils.artifacts.ensure_training_run_dirs("example_training_run", storage_root=tmp_path)
    evaluation_paths = utils.artifacts.ensure_evaluation_run_dirs("eval_example_on_hover", storage_root=tmp_path)
    comparison_paths = utils.artifacts.ensure_comparison_report_dirs("comparison_smoke", storage_root=tmp_path)

    assert training_paths["run"] == tmp_path / "training_runs" / "example_training_run"
    assert evaluation_paths["run"] == tmp_path / "evaluation_runs" / "eval_example_on_hover"
    assert comparison_paths["run"] == tmp_path / "comparison_reports" / "comparison_smoke"
    for name in ("models", "metrics", "manifests", "logs", "wandb", "diagnostics"):
        assert training_paths[name].is_dir()
    for name in ("renders", "traces", "plots", "manifests"):
        assert evaluation_paths[name].is_dir()
    for name in ("metrics", "plots", "manifests"):
        assert comparison_paths[name].is_dir()


def test_run_name_rejects_absolute_or_traversal_paths(tmp_path: Path) -> None:
    """Verify run names cannot escape the category-specific layout."""
    with pytest.raises(ValueError, match="run_name"):
        utils.artifacts.get_training_run_dir("../escape", storage_root=tmp_path)
    with pytest.raises(ValueError, match="run_name"):
        utils.artifacts.get_evaluation_run_dir(str(tmp_path / "absolute"), storage_root=tmp_path)
