"""Tests for central project and storage path helpers."""

# ruff: noqa: S101

from __future__ import annotations

from typing import TYPE_CHECKING

from src import utils

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

EXPECTED_STORAGE_SUBDIRECTORIES = {
    "results",
    "models",
    "videos",
    "gifs",
    "llm_logs",
    "wandb",
    "datasets",
    "tmp",
}


def test_project_root_uses_non_empty_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify PROJECT_ROOT takes precedence when it is non-empty."""
    project_root = tmp_path / "configured-project"

    monkeypatch.setenv("PROJECT_ROOT", str(project_root))

    assert utils.paths.get_project_root() == project_root.resolve(strict=False)


def test_empty_project_root_environment_variable_is_ignored(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify empty PROJECT_ROOT values fall back to the module location."""
    project_root = tmp_path / "repo"
    module_file = project_root / "src" / "utils" / "utils_paths.py"

    monkeypatch.setenv("PROJECT_ROOT", "  ")
    monkeypatch.setattr(utils.paths, "__file__", str(module_file))

    assert utils.paths.get_project_root() == project_root.resolve(strict=False)


def test_storage_root_uses_non_empty_environment_variable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify STORAGE_ROOT takes precedence for all storage path helpers."""
    storage_root = tmp_path / "configured-storage"

    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))

    assert utils.paths.get_storage_root() == storage_root.resolve(strict=False)
    assert utils.paths.get_results_root() == (storage_root / "results").resolve(strict=False)
    assert utils.paths.get_models_root() == (storage_root / "models").resolve(strict=False)
    assert utils.paths.get_videos_root() == (storage_root / "videos").resolve(strict=False)
    assert utils.paths.get_gifs_root() == (storage_root / "gifs").resolve(strict=False)
    assert utils.paths.get_llm_logs_root() == (storage_root / "llm_logs").resolve(strict=False)
    assert utils.paths.get_wandb_root() == (storage_root / "wandb").resolve(strict=False)
    assert utils.paths.get_datasets_root() == (storage_root / "datasets").resolve(strict=False)
    assert utils.paths.get_tmp_root() == (storage_root / "tmp").resolve(strict=False)


def test_default_storage_root_uses_project_sibling_when_project_storage_is_absent(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the default storage root is beside the project when no project storage path exists."""
    project_root = tmp_path / "repo"
    module_file = project_root / "src" / "utils" / "utils_paths.py"

    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.delenv("STORAGE_ROOT", raising=False)
    monkeypatch.setattr(utils.paths, "__file__", str(module_file))

    assert utils.paths.get_storage_root() == (tmp_path / "storage").resolve(strict=False)
    assert not (project_root / "storage").exists()
    assert not (tmp_path / "storage").exists()


def test_default_storage_root_prefers_existing_project_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify an existing project storage path is preferred over sibling storage."""
    project_root = tmp_path / "repo"
    project_storage = project_root / "storage"
    module_file = project_root / "src" / "utils" / "utils_paths.py"

    project_storage.mkdir(parents=True)
    monkeypatch.delenv("PROJECT_ROOT", raising=False)
    monkeypatch.delenv("STORAGE_ROOT", raising=False)
    monkeypatch.setattr(utils.paths, "__file__", str(module_file))

    assert utils.paths.get_storage_root() == project_storage.resolve(strict=False)


def test_getters_do_not_create_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify path getters are side-effect free."""
    storage_root = tmp_path / "storage"

    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))

    _ = utils.paths.get_storage_root()
    _ = utils.paths.get_results_root()
    _ = utils.paths.get_models_root()
    _ = utils.paths.get_videos_root()
    _ = utils.paths.get_gifs_root()
    _ = utils.paths.get_llm_logs_root()
    _ = utils.paths.get_wandb_root()
    _ = utils.paths.get_datasets_root()
    _ = utils.paths.get_tmp_root()

    assert not storage_root.exists()


def test_ensure_storage_layout_creates_expected_directories(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify storage layout creation is isolated to the configured storage root."""
    storage_root = tmp_path / "storage"

    monkeypatch.setenv("STORAGE_ROOT", str(storage_root))

    utils.paths.ensure_storage_layout()

    assert storage_root.is_dir()
    assert {path.name for path in storage_root.iterdir()} == EXPECTED_STORAGE_SUBDIRECTORIES
    for subdirectory in EXPECTED_STORAGE_SUBDIRECTORIES:
        assert (storage_root / subdirectory).is_dir()
