"""Tests for run-scoped artifact path helpers."""

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
    """Verify STORAGE_ROOT controls all run-scoped artifact helpers."""
    storage_root = tmp_path / "external_storage"
    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))

    assert utils.artifacts.get_run_dir("ppo_tracking_smoke") == storage_root / "runs" / "ppo_tracking_smoke"
    assert utils.artifacts.get_metrics_dir("ppo_tracking_smoke") == storage_root / "runs" / "ppo_tracking_smoke" / "metrics"
    assert utils.artifacts.get_manifests_dir("ppo_tracking_smoke") == storage_root / "runs" / "ppo_tracking_smoke" / "manifests"
    assert utils.artifacts.get_renders_dir("ppo_tracking_smoke") == storage_root / "runs" / "ppo_tracking_smoke" / "renders"
    assert utils.artifacts.get_models_dir("ppo_tracking_smoke") == storage_root / "runs" / "ppo_tracking_smoke" / "models"
    assert utils.artifacts.get_plots_dir("ppo_tracking_smoke") == storage_root / "runs" / "ppo_tracking_smoke" / "plots"
    assert utils.artifacts.get_config_dir("ppo_tracking_smoke") == storage_root / "runs" / "ppo_tracking_smoke" / "config"
    assert utils.artifacts.get_logs_dir("ppo_tracking_smoke") == storage_root / "runs" / "ppo_tracking_smoke" / "logs"
    assert utils.artifacts.get_llm_logs_dir("ppo_tracking_smoke") == storage_root / "runs" / "ppo_tracking_smoke" / "llm_logs"
    assert utils.artifacts.get_wandb_dir("ppo_tracking_smoke") == storage_root / "runs" / "ppo_tracking_smoke" / "wandb"


def test_ensure_run_dirs_creates_standard_layout(tmp_path: Path) -> None:
    """Verify run directory creation is explicit and complete."""
    paths = utils.artifacts.ensure_run_dirs("render_smoke", storage_root=tmp_path)

    assert paths["run"] == tmp_path / "runs" / "render_smoke"
    for name in ("config", "metrics", "manifests", "plots", "renders", "models", "logs", "llm_logs", "wandb"):
        assert paths[name].is_dir()


def test_run_name_rejects_absolute_or_traversal_paths(tmp_path: Path) -> None:
    """Verify run names cannot escape the storage/runs layout."""
    with pytest.raises(ValueError, match="run_name"):
        utils.artifacts.get_run_dir("../escape", storage_root=tmp_path)
    with pytest.raises(ValueError, match="run_name"):
        utils.artifacts.get_run_dir(str(tmp_path / "absolute"), storage_root=tmp_path)
