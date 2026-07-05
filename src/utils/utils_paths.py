"""
===============================================================================
utils_paths.py
===============================================================================
Resolve project and external storage paths for experiment artifacts.

Responsibilities:
  - Resolve the project root from environment configuration or repository layout
  - Resolve the external storage root without creating directories
  - Create the approved storage directory layout on request

Design principles:
  - Keep path resolution deterministic and side-effect free
  - Re-read environment variables on every call so tests and runtime overrides work
  - Centralize storage subdirectory names to avoid hardcoded paths elsewhere

Boundaries:
  - Training, logging, and serialization logic belong in their own utility modules
  - Generated artifacts must not be created outside the storage layout
===============================================================================

"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT_ENV = "PROJECT_ROOT"
STORAGE_ROOT_ENV = "STORAGE_ROOT"

STORAGE_SUBDIRECTORIES: tuple[str, ...] = (
    "results",
    "models",
    "videos",
    "gifs",
    "llm_logs",
    "wandb",
    "datasets",
    "tmp",
)


def get_project_root() -> Path:
    """Get the project root from ``PROJECT_ROOT`` or the repository layout."""
    env_root = _get_env_path(PROJECT_ROOT_ENV)
    if env_root is not None:
        return env_root

    return Path(__file__).expanduser().resolve(strict=False).parents[2]


def get_storage_root() -> Path:
    """Get the external storage root without creating directories."""
    env_root = _get_env_path(STORAGE_ROOT_ENV)
    if env_root is not None:
        return env_root

    project_root = get_project_root()
    project_storage = project_root / "storage"
    if project_storage.is_symlink() or project_storage.exists():
        return project_storage.expanduser().resolve(strict=False)

    return (project_root.parent / "storage").expanduser().resolve(strict=False)


def get_results_root() -> Path:
    """Get the storage directory for experiment result artifacts."""
    return _get_storage_subdirectory("results")


def get_models_root() -> Path:
    """Get the storage directory for trained model artifacts."""
    return _get_storage_subdirectory("models")


def get_videos_root() -> Path:
    """Get the storage directory for rendered video artifacts."""
    return _get_storage_subdirectory("videos")


def get_gifs_root() -> Path:
    """Get the storage directory for rendered GIF artifacts."""
    return _get_storage_subdirectory("gifs")


def get_llm_logs_root() -> Path:
    """Get the storage directory for LLM proposal and validation logs."""
    return _get_storage_subdirectory("llm_logs")


def get_wandb_root() -> Path:
    """Get the storage directory for Weights & Biases run data."""
    return _get_storage_subdirectory("wandb")


def get_datasets_root() -> Path:
    """Get the storage directory for generated or downloaded datasets."""
    return _get_storage_subdirectory("datasets")


def get_tmp_root() -> Path:
    """Get the storage directory for temporary project artifacts."""
    return _get_storage_subdirectory("tmp")


def ensure_storage_layout() -> None:
    """Create the storage root and approved storage subdirectories."""
    get_storage_root().mkdir(parents=True, exist_ok=True)
    for subdirectory in STORAGE_SUBDIRECTORIES:
        _get_storage_subdirectory(subdirectory).mkdir(parents=True, exist_ok=True)


def _get_env_path(name: str) -> Path | None:
    """Resolve a non-empty path environment variable if one is configured."""
    value = os.environ.get(name)
    if value is None or not value.strip():
        return None

    return Path(value).expanduser().resolve(strict=False)


def _get_storage_subdirectory(name: str) -> Path:
    """Resolve a named storage subdirectory without creating it."""
    return (get_storage_root() / name).expanduser().resolve(strict=False)


__all__ = [
    "ensure_storage_layout",
    "get_datasets_root",
    "get_gifs_root",
    "get_llm_logs_root",
    "get_models_root",
    "get_project_root",
    "get_results_root",
    "get_storage_root",
    "get_tmp_root",
    "get_videos_root",
    "get_wandb_root",
]
