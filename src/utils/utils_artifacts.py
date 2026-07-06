"""
===============================================================================
utils_artifacts.py
===============================================================================
Resolve run-scoped artifact directories under the project storage root.

Responsibilities:
  - Centralize the storage/runs/<run_name>/ artifact layout
  - Respect STORAGE_ROOT without creating directories at import time
  - Create run directory trees only when explicitly requested

Design principles:
  - Keep path helpers small, deterministic, and side-effect free by default
  - Use pathlib paths so CLIs, tests, Docker and HPC jobs share one contract

Boundaries:
  - Experiment modules decide when artifacts are written
  - This module does not delete or migrate legacy storage folders
===============================================================================

"""

from __future__ import annotations

import os
from pathlib import Path

RUNS_DIRNAME = "runs"
CONFIG_DIRNAME = "config"
METRICS_DIRNAME = "metrics"
MANIFESTS_DIRNAME = "manifests"
PLOTS_DIRNAME = "plots"
RENDERS_DIRNAME = "renders"
MODELS_DIRNAME = "models"
LOGS_DIRNAME = "logs"
LLM_LOGS_DIRNAME = "llm_logs"
WANDB_DIRNAME = "wandb"

_RUN_SUBDIRS = (
    CONFIG_DIRNAME,
    METRICS_DIRNAME,
    MANIFESTS_DIRNAME,
    PLOTS_DIRNAME,
    RENDERS_DIRNAME,
    MODELS_DIRNAME,
    LOGS_DIRNAME,
    LLM_LOGS_DIRNAME,
    WANDB_DIRNAME,
)


def get_storage_root() -> Path:
    """Get the storage root directory from STORAGE_ROOT or the local storage symlink."""
    return Path(os.environ.get("STORAGE_ROOT", "storage")).expanduser().resolve(strict=False)


def get_run_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the run root directory for a named experiment run."""
    return _storage_root(storage_root) / RUNS_DIRNAME / _validate_run_name(run_name)


def get_metrics_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the metrics artifact directory for a named run."""
    return get_run_dir(run_name, storage_root) / METRICS_DIRNAME


def get_manifests_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the manifest artifact directory for a named run."""
    return get_run_dir(run_name, storage_root) / MANIFESTS_DIRNAME


def get_renders_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the render artifact directory for a named run."""
    return get_run_dir(run_name, storage_root) / RENDERS_DIRNAME


def get_models_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the model artifact directory for a named run."""
    return get_run_dir(run_name, storage_root) / MODELS_DIRNAME


def get_plots_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the plot artifact directory for a named run."""
    return get_run_dir(run_name, storage_root) / PLOTS_DIRNAME


def get_config_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the copied/resolved config artifact directory for a named run."""
    return get_run_dir(run_name, storage_root) / CONFIG_DIRNAME


def get_logs_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the run-specific log artifact directory for a named run."""
    return get_run_dir(run_name, storage_root) / LOGS_DIRNAME


def get_llm_logs_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the run-specific LLM log artifact directory for a named run."""
    return get_run_dir(run_name, storage_root) / LLM_LOGS_DIRNAME


def get_wandb_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the run-specific W&B artifact directory for a named run."""
    return get_run_dir(run_name, storage_root) / WANDB_DIRNAME


def ensure_run_dirs(run_name: str, storage_root: str | Path | None = None) -> dict[str, Path]:
    """
    Create and return the standard artifact directories for a named run.

    Parameters
    ----------
    run_name
        Stable run identifier used under ``storage/runs``.
    storage_root
        Optional storage root override. Defaults to ``STORAGE_ROOT`` or ``storage``.

    Returns
    -------
    dict[str, Path]
        Mapping from subdirectory name to created path, including ``run`` for the run root.

    """
    run_dir = get_run_dir(run_name, storage_root)
    paths = {"run": run_dir, **{subdir: run_dir / subdir for subdir in _RUN_SUBDIRS}}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _storage_root(storage_root: str | Path | None) -> Path:
    """Return an explicit storage root or the configured default."""
    if storage_root is None:
        return get_storage_root()
    return Path(storage_root).expanduser().resolve(strict=False)


def _validate_run_name(run_name: str) -> str:
    """Validate and return a simple relative run name."""
    normalized = run_name.strip()
    if not normalized:
        message = "run_name must be non-empty"
        raise ValueError(message)
    if Path(normalized).is_absolute() or any(part in {"", ".", ".."} for part in Path(normalized).parts):
        message = "run_name must be a simple relative path without traversal"
        raise ValueError(message)
    return normalized


__all__ = [
    "CONFIG_DIRNAME",
    "LLM_LOGS_DIRNAME",
    "LOGS_DIRNAME",
    "MANIFESTS_DIRNAME",
    "METRICS_DIRNAME",
    "MODELS_DIRNAME",
    "PLOTS_DIRNAME",
    "RENDERS_DIRNAME",
    "RUNS_DIRNAME",
    "WANDB_DIRNAME",
    "ensure_run_dirs",
    "get_config_dir",
    "get_llm_logs_dir",
    "get_logs_dir",
    "get_manifests_dir",
    "get_metrics_dir",
    "get_models_dir",
    "get_plots_dir",
    "get_renders_dir",
    "get_run_dir",
    "get_storage_root",
    "get_wandb_dir",
]
