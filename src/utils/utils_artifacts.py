"""
===============================================================================
utils_artifacts.py
===============================================================================
Resolve category-scoped artifact directories under the project storage root.

Responsibilities:
    - Centralize training, evaluation, and comparison-report storage layouts
  - Respect STORAGE_ROOT without creating directories at import time
    - Create category directory trees only when explicitly requested

Design principles:
  - Keep path helpers small, deterministic, and side-effect free by default
  - Use pathlib paths so CLIs, tests, Docker and HPC jobs share one contract

Boundaries:
    - Experiment modules decide when artifacts are written
    - This module only creates category-specific storage trees
===============================================================================

"""

from __future__ import annotations

import os
from pathlib import Path

TRAINING_RUNS_DIRNAME = "training_runs"
EVALUATION_RUNS_DIRNAME = "evaluation_runs"
COMPARISON_REPORTS_DIRNAME = "comparison_reports"
CONFIG_DIRNAME = "config"
METRICS_DIRNAME = "metrics"
MANIFESTS_DIRNAME = "manifests"
PLOTS_DIRNAME = "plots"
RENDERS_DIRNAME = "renders"
TRACES_DIRNAME = "traces"
MODELS_DIRNAME = "models"
LOGS_DIRNAME = "logs"
LLM_LOGS_DIRNAME = "llm_logs"
WANDB_DIRNAME = "wandb"

_TRAINING_RUN_SUBDIRS = (
    MODELS_DIRNAME,
    METRICS_DIRNAME,
    MANIFESTS_DIRNAME,
    LOGS_DIRNAME,
    WANDB_DIRNAME,
)

_EVALUATION_RUN_SUBDIRS = (
    RENDERS_DIRNAME,
    TRACES_DIRNAME,
    PLOTS_DIRNAME,
    MANIFESTS_DIRNAME,
)

_COMPARISON_REPORT_SUBDIRS = (
    METRICS_DIRNAME,
    PLOTS_DIRNAME,
    MANIFESTS_DIRNAME,
)


def get_storage_root() -> Path:
    """Get the storage root directory from STORAGE_ROOT or the local storage symlink."""
    return Path(os.environ.get("STORAGE_ROOT", "storage")).expanduser().resolve(strict=False)


def get_training_run_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the training run root directory for a named experiment run."""
    return _storage_root(storage_root) / TRAINING_RUNS_DIRNAME / _validate_run_name(run_name)


def get_evaluation_run_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the evaluation run root directory for a named experiment run."""
    return _storage_root(storage_root) / EVALUATION_RUNS_DIRNAME / _validate_run_name(run_name)


def get_training_metrics_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the metrics artifact directory for a training run."""
    return get_training_run_dir(run_name, storage_root) / METRICS_DIRNAME


def get_evaluation_metrics_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the metrics artifact directory for an evaluation run."""
    return get_evaluation_run_dir(run_name, storage_root) / METRICS_DIRNAME


def get_training_manifests_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the manifest artifact directory for a training run."""
    return get_training_run_dir(run_name, storage_root) / MANIFESTS_DIRNAME


def get_evaluation_manifests_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the manifest artifact directory for an evaluation run."""
    return get_evaluation_run_dir(run_name, storage_root) / MANIFESTS_DIRNAME


def get_evaluation_renders_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the render artifact directory for an evaluation run."""
    return get_evaluation_run_dir(run_name, storage_root) / RENDERS_DIRNAME


def get_evaluation_traces_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the trace artifact directory for an evaluation run."""
    return get_evaluation_run_dir(run_name, storage_root) / TRACES_DIRNAME


def get_training_models_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the model artifact directory for a training run."""
    return get_training_run_dir(run_name, storage_root) / MODELS_DIRNAME


def get_evaluation_plots_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the plot artifact directory for an evaluation run."""
    return get_evaluation_run_dir(run_name, storage_root) / PLOTS_DIRNAME


def get_training_logs_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the log artifact directory for a training run."""
    return get_training_run_dir(run_name, storage_root) / LOGS_DIRNAME


def get_training_wandb_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the W&B artifact directory for a training run."""
    return get_training_run_dir(run_name, storage_root) / WANDB_DIRNAME


def get_evaluation_wandb_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the W&B artifact directory for an evaluation run."""
    return get_evaluation_run_dir(run_name, storage_root) / WANDB_DIRNAME


def ensure_training_run_dirs(run_name: str, storage_root: str | Path | None = None) -> dict[str, Path]:
    """
    Create and return the standard artifact directories for a training run.

    Parameters
    ----------
    run_name
        Stable run identifier used under ``storage/training_runs``.
    storage_root
        Optional storage root override. Defaults to ``STORAGE_ROOT`` or ``storage``.

    Returns
    -------
    dict[str, Path]
        Mapping from subdirectory name to created path, including ``run`` for the run root.

    """
    run_dir = get_training_run_dir(run_name, storage_root)
    paths = {"run": run_dir, **{subdir: run_dir / subdir for subdir in _TRAINING_RUN_SUBDIRS}}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_evaluation_run_dirs(run_name: str, storage_root: str | Path | None = None) -> dict[str, Path]:
    """
    Create and return the standard artifact directories for an evaluation run.

    Parameters
    ----------
    run_name
        Stable run identifier used under ``storage/evaluation_runs``.
    storage_root
        Optional storage root override. Defaults to ``STORAGE_ROOT`` or ``storage``.

    Returns
    -------
    dict[str, Path]
        Mapping from subdirectory name to created path, including ``run`` for the run root.

    """
    run_dir = get_evaluation_run_dir(run_name, storage_root)
    paths = {"run": run_dir, **{subdir: run_dir / subdir for subdir in _EVALUATION_RUN_SUBDIRS}}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_comparison_report_dirs(run_name: str, storage_root: str | Path | None = None) -> dict[str, Path]:
    """
    Create and return the standard artifact directories for a comparison report.

    Parameters
    ----------
    run_name
        Stable report identifier used under ``storage/comparison_reports``.
    storage_root
        Optional storage root override. Defaults to ``STORAGE_ROOT`` or ``storage``.

    Returns
    -------
    dict[str, Path]
        Mapping from subdirectory name to created path, including ``run`` for the run root.

    """
    run_dir = get_comparison_report_dir(run_name, storage_root)
    paths = {"run": run_dir, **{subdir: run_dir / subdir for subdir in _COMPARISON_REPORT_SUBDIRS}}
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def get_comparison_report_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the comparison report root directory for a named report."""
    return _storage_root(storage_root) / COMPARISON_REPORTS_DIRNAME / _validate_run_name(run_name)


def get_comparison_report_metrics_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the metrics directory for a comparison report."""
    return get_comparison_report_dir(run_name, storage_root) / METRICS_DIRNAME


def get_comparison_report_plots_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the plot directory for a comparison report."""
    return get_comparison_report_dir(run_name, storage_root) / PLOTS_DIRNAME


def get_comparison_report_manifests_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the manifest directory for a comparison report."""
    return get_comparison_report_dir(run_name, storage_root) / MANIFESTS_DIRNAME


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
    "COMPARISON_REPORTS_DIRNAME",
    "CONFIG_DIRNAME",
    "EVALUATION_RUNS_DIRNAME",
    "MANIFESTS_DIRNAME",
    "METRICS_DIRNAME",
    "MODELS_DIRNAME",
    "PLOTS_DIRNAME",
    "RENDERS_DIRNAME",
    "TRACES_DIRNAME",
    "TRAINING_RUNS_DIRNAME",
    "WANDB_DIRNAME",
    "ensure_evaluation_run_dirs",
    "ensure_training_run_dirs",
    "get_comparison_report_dir",
    "get_comparison_report_manifests_dir",
    "get_comparison_report_metrics_dir",
    "get_comparison_report_plots_dir",
    "get_evaluation_manifests_dir",
    "get_evaluation_metrics_dir",
    "get_evaluation_plots_dir",
    "get_evaluation_renders_dir",
    "get_evaluation_run_dir",
    "get_evaluation_traces_dir",
    "get_evaluation_wandb_dir",
    "get_storage_root",
    "get_training_logs_dir",
    "get_training_manifests_dir",
    "get_training_metrics_dir",
    "get_training_models_dir",
    "get_training_run_dir",
    "get_training_wandb_dir",
]
