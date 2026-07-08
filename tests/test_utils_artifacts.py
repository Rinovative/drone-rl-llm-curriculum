"""Tests for canonical storage/runs artifact path helpers."""

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
    """Verify STORAGE_ROOT controls the canonical storage/runs helpers."""
    storage_root = tmp_path / "external_storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))

    assert utils.artifacts.get_run_dir("direct_ppo_line_seed0") == storage_root / "runs" / "direct_ppo_line_seed0"
    assert utils.artifacts.get_run_manifest_path("direct_ppo_line_seed0") == storage_root / "runs" / "direct_ppo_line_seed0" / "run_manifest.json"
    assert utils.artifacts.get_run_config_dir("direct_ppo_line_seed0") == storage_root / "runs" / "direct_ppo_line_seed0" / "config"
    assert (
        utils.artifacts.get_run_config_evaluation_suites_dir("direct_ppo_line_seed0")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "config" / "evaluation_suites"
    )
    assert (
        utils.artifacts.get_run_training_config_snapshot_path("direct_ppo_line_seed0")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "config" / "training_config.yaml"
    )
    assert (
        utils.artifacts.get_run_task_config_snapshot_path("direct_ppo_line_seed0")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "config" / "task_config.yaml"
    )
    assert (
        utils.artifacts.get_run_curriculum_config_snapshot_path("direct_ppo_line_seed0")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "config" / "curriculum_config.yaml"
    )
    assert utils.artifacts.get_run_training_dir("direct_ppo_line_seed0") == storage_root / "runs" / "direct_ppo_line_seed0" / "training"
    assert (
        utils.artifacts.get_run_training_manifest_path("direct_ppo_line_seed0")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "training" / "manifest.json"
    )
    assert (
        utils.artifacts.get_run_training_models_dir("direct_ppo_line_seed0")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "training" / "models"
    )
    assert (
        utils.artifacts.get_run_training_metrics_dir("direct_ppo_line_seed0")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "training" / "metrics"
    )
    assert (
        utils.artifacts.get_run_training_diagnostics_dir("direct_ppo_line_seed0")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "training" / "diagnostics"
    )
    assert utils.artifacts.get_run_training_logs_dir("direct_ppo_line_seed0") == storage_root / "runs" / "direct_ppo_line_seed0" / "training" / "logs"
    assert (
        utils.artifacts.get_run_training_wandb_dir("direct_ppo_line_seed0") == storage_root / "runs" / "direct_ppo_line_seed0" / "training" / "wandb"
    )
    assert utils.artifacts.get_run_llm_logs_dir("direct_ppo_line_seed0") == storage_root / "runs" / "direct_ppo_line_seed0" / "llm_logs"
    assert (
        utils.artifacts.get_run_llm_proposals_path("direct_ppo_line_seed0")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "llm_logs" / "proposals.jsonl"
    )
    assert (
        utils.artifacts.get_run_evaluation_dir("direct_ppo_line_seed0", "line_basic")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "evaluations" / "line_basic"
    )
    assert (
        utils.artifacts.get_run_evaluation_metrics_dir("direct_ppo_line_seed0", "line_basic")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "evaluations" / "line_basic" / "metrics"
    )
    assert (
        utils.artifacts.get_run_evaluation_index_path("direct_ppo_line_seed0")
        == storage_root / "runs" / "direct_ppo_line_seed0" / "evaluation_index.json"
    )


def test_relative_path_helpers_are_portable(tmp_path: Path) -> None:
    """Verify manifest-relative paths avoid absolute storage paths when possible."""
    run_root = tmp_path / "runs" / "direct_ppo_line_seed0"
    metrics_path = run_root / "training" / "metrics" / "metrics.json"
    outside_path = tmp_path / "external" / "artifact.json"

    assert utils.artifacts.path_relative_to(metrics_path, run_root) == "training/metrics/metrics.json"
    assert utils.artifacts.path_relative_to_run(metrics_path, "direct_ppo_line_seed0", storage_root=tmp_path) == "training/metrics/metrics.json"
    assert utils.artifacts.path_relative_to("training/models/model.zip", run_root) == "training/models/model.zip"
    assert utils.artifacts.path_relative_to(None, run_root) is None
    assert utils.artifacts.path_relative_to(outside_path, run_root) == outside_path.as_posix()
    assert utils.artifacts.storage_root_from_run_dir(run_root) == tmp_path


def test_run_helpers_sanitize_single_path_segments(tmp_path: Path) -> None:
    """Verify run and evaluation labels are sanitized without allowing nested paths."""
    assert utils.artifacts.get_run_dir(" Direct PPO: Seed 0! ", storage_root=tmp_path) == tmp_path / "runs" / "Direct_PPO_Seed_0"
    assert (
        utils.artifacts.get_run_evaluation_dir("direct_ppo", " Line basic eval! ", storage_root=tmp_path)
        == tmp_path / "runs" / "direct_ppo" / "evaluations" / "Line_basic_eval"
    )


def test_ensure_run_dirs_create_canonical_layouts(tmp_path: Path) -> None:
    """Verify direct-run training and evaluation trees are created explicitly."""
    training_paths = utils.artifacts.ensure_run_training_dirs("direct_ppo_line_seed0", storage_root=tmp_path)
    evaluation_paths = utils.artifacts.ensure_run_evaluation_dirs("direct_ppo_line_seed0", "line_basic", storage_root=tmp_path)

    assert training_paths["run"] == tmp_path / "runs" / "direct_ppo_line_seed0"
    assert training_paths["config"] == tmp_path / "runs" / "direct_ppo_line_seed0" / "config"
    assert training_paths["evaluation_suites"] == tmp_path / "runs" / "direct_ppo_line_seed0" / "config" / "evaluation_suites"
    assert training_paths["training"] == tmp_path / "runs" / "direct_ppo_line_seed0" / "training"
    for name in ("models", "metrics", "diagnostics", "logs", "wandb"):
        assert training_paths[name].is_dir()
    assert utils.artifacts.get_run_manifest_path("direct_ppo_line_seed0", storage_root=tmp_path).parent.is_dir()
    assert utils.artifacts.get_run_training_manifest_path("direct_ppo_line_seed0", storage_root=tmp_path).parent.is_dir()
    assert utils.artifacts.ensure_run_llm_logs_dir("direct_ppo_line_seed0", storage_root=tmp_path).is_dir()

    assert evaluation_paths["evaluation"] == tmp_path / "runs" / "direct_ppo_line_seed0" / "evaluations" / "line_basic"
    for name in ("diagnostics", "traces", "plots", "renders", "metrics", "manifests"):
        assert evaluation_paths[name].is_dir()


def test_ensure_curriculum_stage_dirs_create_canonical_layouts(tmp_path: Path) -> None:
    """Verify curriculum stages use the storage/runs stage contract."""
    training_paths = utils.artifacts.ensure_curriculum_stage_training_dirs(
        "manual_line_seed0",
        stage_index=1,
        stage_name="short slow line",
        storage_root=tmp_path,
    )
    evaluation_paths = utils.artifacts.ensure_curriculum_stage_evaluation_dirs(
        "manual_line_seed0",
        stage_index=1,
        stage_name="short slow line",
        evaluation_name="final benchmark",
        storage_root=tmp_path,
    )

    stage_dir = tmp_path / "runs" / "manual_line_seed0" / "stages" / "stage01_short_slow_line"
    assert training_paths["stage"] == stage_dir
    assert training_paths["training"] == stage_dir / "training"
    assert (
        utils.artifacts.get_curriculum_stage_training_manifest_path("manual_line_seed0", 1, "short slow line", storage_root=tmp_path)
        == stage_dir / "training" / "manifest.json"
    )
    for name in ("models", "metrics", "diagnostics", "logs", "wandb"):
        assert training_paths[name].is_dir()
    assert evaluation_paths["evaluation"] == stage_dir / "evaluations" / "final_benchmark"
    for name in ("diagnostics", "traces", "plots", "renders", "metrics", "manifests"):
        assert evaluation_paths[name].is_dir()


def test_names_reject_absolute_nested_or_traversal_paths(tmp_path: Path) -> None:
    """Verify canonical helper identifiers stay inside one safe path segment."""
    with pytest.raises(ValueError, match="run_name"):
        utils.artifacts.get_run_dir("../escape", storage_root=tmp_path)
    with pytest.raises(ValueError, match="run_name"):
        utils.artifacts.get_run_dir("nested/run", storage_root=tmp_path)
    with pytest.raises(ValueError, match="evaluation_name"):
        utils.artifacts.get_run_evaluation_dir("direct_ppo", "../escape", storage_root=tmp_path)
    with pytest.raises(ValueError, match="stage_index"):
        utils.artifacts.get_curriculum_stage_dir("manual_line", 0, "line", storage_root=tmp_path)
