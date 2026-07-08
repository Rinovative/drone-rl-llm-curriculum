"""
===============================================================================
utils_artifacts.py
===============================================================================
Resolve canonical run artifact directories under the project storage root.

Responsibilities:
  - Centralize the storage/runs run, training, evaluation, and curriculum-stage layout
  - Validate artifact path identifiers before they are joined into storage paths
  - Respect STORAGE_ROOT without creating directories at import time
  - Create canonical directory trees only when explicitly requested

Design principles:
  - Keep path helpers small, deterministic, and side-effect free by default
  - Use pathlib paths so CLIs, tests, Docker and HPC jobs share one contract

Boundaries:
  - Experiment modules decide when artifacts are written
  - This module only creates canonical run-scoped storage trees
===============================================================================

"""

from __future__ import annotations

import os
import re
from pathlib import Path

RUNS_DIRNAME = "runs"
CONFIG_DIRNAME = "config"
EVALUATION_SUITES_DIRNAME = "evaluation_suites"
TRAINING_CONFIG_SNAPSHOT_FILENAME = "training_config.yaml"
TASK_CONFIG_SNAPSHOT_FILENAME = "task_config.yaml"
CURRICULUM_CONFIG_SNAPSHOT_FILENAME = "curriculum_config.yaml"
TRAINING_DIRNAME = "training"
EVALUATIONS_DIRNAME = "evaluations"
EVALUATION_INDEX_FILENAME = "evaluation_index.json"
EVALUATION_SUMMARY_FILENAME = "evaluation_summary.json"
STAGES_DIRNAME = "stages"
METRICS_DIRNAME = "metrics"
MANIFESTS_DIRNAME = "manifests"
PLOTS_DIRNAME = "plots"
RENDERS_DIRNAME = "renders"
TRACES_DIRNAME = "traces"
MODELS_DIRNAME = "models"
LOGS_DIRNAME = "logs"
LLM_LOGS_DIRNAME = "llm_logs"
WANDB_DIRNAME = "wandb"
DIAGNOSTICS_DIRNAME = "diagnostics"
RUN_MANIFEST_FILENAME = "run_manifest.json"
MANIFEST_FILENAME = "manifest.json"
_ARTIFACT_NAME_INVALID_CHARS = re.compile(r"[^A-Za-z0-9_.-]+")

_RUN_TRAINING_SUBDIRS = (
    MODELS_DIRNAME,
    METRICS_DIRNAME,
    DIAGNOSTICS_DIRNAME,
    LOGS_DIRNAME,
    WANDB_DIRNAME,
)

_RUN_EVALUATION_SUBDIRS = (
    DIAGNOSTICS_DIRNAME,
    TRACES_DIRNAME,
    PLOTS_DIRNAME,
    RENDERS_DIRNAME,
    METRICS_DIRNAME,
    MANIFESTS_DIRNAME,
)


def get_storage_root() -> Path:
    """Get the storage root directory from STORAGE_ROOT or the local storage symlink."""
    return Path(os.environ.get("STORAGE_ROOT", "storage")).expanduser().resolve(strict=False)


def get_run_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the canonical run root directory for a named experiment run."""
    return _storage_root(storage_root) / RUNS_DIRNAME / _validate_artifact_name(run_name, "run_name")


def get_run_manifest_path(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the root manifest path for a canonical run."""
    return get_run_dir(run_name, storage_root) / RUN_MANIFEST_FILENAME


def get_run_config_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the copied-configuration directory for a canonical run."""
    return get_run_dir(run_name, storage_root) / CONFIG_DIRNAME


def get_run_config_evaluation_suites_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the evaluation-suite configuration directory for a canonical run."""
    return get_run_config_dir(run_name, storage_root) / EVALUATION_SUITES_DIRNAME


def get_run_training_config_snapshot_path(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the canonical copied training-config snapshot path for a run."""
    return get_run_config_dir(run_name, storage_root) / TRAINING_CONFIG_SNAPSHOT_FILENAME


def get_run_task_config_snapshot_path(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the canonical copied task-config snapshot path for a run."""
    return get_run_config_dir(run_name, storage_root) / TASK_CONFIG_SNAPSHOT_FILENAME


def get_run_curriculum_config_snapshot_path(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the canonical copied curriculum-config snapshot path for a run."""
    return get_run_config_dir(run_name, storage_root) / CURRICULUM_CONFIG_SNAPSHOT_FILENAME


def get_run_training_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the training artifact directory for a canonical run."""
    return get_run_dir(run_name, storage_root) / TRAINING_DIRNAME


def get_run_training_manifest_path(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the training manifest path for a canonical run."""
    return get_run_training_dir(run_name, storage_root) / MANIFEST_FILENAME


def get_run_training_models_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the model artifact directory for canonical-run training."""
    return get_run_training_dir(run_name, storage_root) / MODELS_DIRNAME


def get_run_training_metrics_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the metrics artifact directory for canonical-run training."""
    return get_run_training_dir(run_name, storage_root) / METRICS_DIRNAME


def get_run_training_diagnostics_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the diagnostics artifact directory for canonical-run training."""
    return get_run_training_dir(run_name, storage_root) / DIAGNOSTICS_DIRNAME


def get_run_training_logs_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the log artifact directory for canonical-run training."""
    return get_run_training_dir(run_name, storage_root) / LOGS_DIRNAME


def get_run_training_wandb_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the W&B artifact directory for canonical-run training."""
    return get_run_training_dir(run_name, storage_root) / WANDB_DIRNAME


def get_run_evaluations_dir(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the evaluations container directory for a canonical run."""
    return get_run_dir(run_name, storage_root) / EVALUATIONS_DIRNAME


def get_run_evaluation_index_path(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the deterministic evaluation-index JSON path for a canonical run."""
    return get_run_dir(run_name, storage_root) / EVALUATION_INDEX_FILENAME


def get_run_evaluation_summary_path(run_name: str, storage_root: str | Path | None = None) -> Path:
    """Return the optional root evaluation-summary JSON path for a canonical run."""
    return get_run_dir(run_name, storage_root) / EVALUATION_SUMMARY_FILENAME


def get_run_evaluation_dir(
    run_name: str,
    evaluation_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return one named evaluation directory for a canonical run."""
    return get_run_evaluations_dir(run_name, storage_root) / _validate_artifact_name(evaluation_name, "evaluation_name")


def get_run_evaluation_diagnostics_dir(
    run_name: str,
    evaluation_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return the diagnostics directory for a canonical-run evaluation."""
    return get_run_evaluation_dir(run_name, evaluation_name, storage_root) / DIAGNOSTICS_DIRNAME


def get_run_evaluation_traces_dir(
    run_name: str,
    evaluation_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return the trace directory for a canonical-run evaluation."""
    return get_run_evaluation_dir(run_name, evaluation_name, storage_root) / TRACES_DIRNAME


def get_run_evaluation_plots_dir(
    run_name: str,
    evaluation_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return the plot directory for a canonical-run evaluation."""
    return get_run_evaluation_dir(run_name, evaluation_name, storage_root) / PLOTS_DIRNAME


def get_run_evaluation_renders_dir(
    run_name: str,
    evaluation_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return the render directory for a canonical-run evaluation."""
    return get_run_evaluation_dir(run_name, evaluation_name, storage_root) / RENDERS_DIRNAME


def get_run_evaluation_metrics_dir(
    run_name: str,
    evaluation_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return the metrics directory for a canonical-run evaluation."""
    return get_run_evaluation_dir(run_name, evaluation_name, storage_root) / METRICS_DIRNAME


def get_run_evaluation_manifests_dir(
    run_name: str,
    evaluation_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return the manifests directory for a canonical-run evaluation."""
    return get_run_evaluation_dir(run_name, evaluation_name, storage_root) / MANIFESTS_DIRNAME


def get_curriculum_stage_dir(
    run_name: str,
    stage_index: int,
    stage_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return one curriculum stage directory inside a canonical run."""
    return get_run_dir(run_name, storage_root) / STAGES_DIRNAME / _stage_dirname(stage_index, stage_name)


def get_curriculum_stage_training_dir(
    run_name: str,
    stage_index: int,
    stage_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return the training directory for one curriculum stage."""
    return get_curriculum_stage_dir(run_name, stage_index, stage_name, storage_root) / TRAINING_DIRNAME


def get_curriculum_stage_training_manifest_path(
    run_name: str,
    stage_index: int,
    stage_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return the training manifest path for one curriculum stage."""
    return get_curriculum_stage_training_dir(run_name, stage_index, stage_name, storage_root) / MANIFEST_FILENAME


def get_curriculum_stage_evaluations_dir(
    run_name: str,
    stage_index: int,
    stage_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return the evaluations container for one curriculum stage."""
    return get_curriculum_stage_dir(run_name, stage_index, stage_name, storage_root) / EVALUATIONS_DIRNAME


def get_curriculum_stage_evaluation_dir(
    run_name: str,
    stage_index: int,
    stage_name: str,
    evaluation_name: str,
    storage_root: str | Path | None = None,
) -> Path:
    """Return one named evaluation directory for a curriculum stage."""
    return get_curriculum_stage_evaluations_dir(run_name, stage_index, stage_name, storage_root) / _validate_artifact_name(
        evaluation_name,
        "evaluation_name",
    )


def ensure_run_training_dirs(run_name: str, storage_root: str | Path | None = None) -> dict[str, Path]:
    """
    Create and return the standard training directories for a canonical run.

    Parameters
    ----------
    run_name
        Stable run identifier used under ``storage/runs``.
    storage_root
        Optional storage root override. Defaults to ``STORAGE_ROOT`` or ``storage``.

    Returns
    -------
    dict[str, Path]
        Mapping from directory labels to created paths for the run config and training tree.

    """
    run_dir = get_run_dir(run_name, storage_root)
    config_dir = get_run_config_dir(run_name, storage_root)
    training_dir = get_run_training_dir(run_name, storage_root)
    paths = {
        "run": run_dir,
        CONFIG_DIRNAME: config_dir,
        EVALUATION_SUITES_DIRNAME: get_run_config_evaluation_suites_dir(run_name, storage_root),
        TRAINING_DIRNAME: training_dir,
        **{subdir: training_dir / subdir for subdir in _RUN_TRAINING_SUBDIRS},
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_run_evaluation_dirs(
    run_name: str,
    evaluation_name: str,
    storage_root: str | Path | None = None,
) -> dict[str, Path]:
    """
    Create and return the standard directories for a canonical-run evaluation.

    Parameters
    ----------
    run_name
        Stable run identifier used under ``storage/runs``.
    evaluation_name
        Stable evaluation identifier used under ``evaluations``.
    storage_root
        Optional storage root override. Defaults to ``STORAGE_ROOT`` or ``storage``.

    Returns
    -------
    dict[str, Path]
        Mapping from directory labels to created paths for the evaluation tree.

    """
    run_dir = get_run_dir(run_name, storage_root)
    evaluations_dir = get_run_evaluations_dir(run_name, storage_root)
    evaluation_dir = get_run_evaluation_dir(run_name, evaluation_name, storage_root)
    paths = {
        "run": run_dir,
        EVALUATIONS_DIRNAME: evaluations_dir,
        "evaluation": evaluation_dir,
        **{subdir: evaluation_dir / subdir for subdir in _RUN_EVALUATION_SUBDIRS},
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def path_relative_to(path: str | Path | None, base_dir: str | Path) -> str | None:
    """Return a stable POSIX path relative to a base directory when possible."""
    if path is None:
        return None
    raw_path = Path(path).expanduser()
    if not raw_path.is_absolute():
        return raw_path.as_posix()
    resolved_path = raw_path.resolve(strict=False)
    resolved_base = Path(base_dir).expanduser().resolve(strict=False)
    try:
        return resolved_path.relative_to(resolved_base).as_posix()
    except ValueError:
        return raw_path.as_posix()


def path_relative_to_run(
    path: str | Path | None,
    run_name: str,
    storage_root: str | Path | None = None,
) -> str | None:
    """Return a stable POSIX path relative to a canonical run root."""
    return path_relative_to(path, get_run_dir(run_name, storage_root))


def storage_root_from_run_dir(run_dir: str | Path) -> Path | None:
    """Infer the storage root from a ``storage/runs/<run_name>`` directory."""
    resolved = Path(run_dir).expanduser().resolve(strict=False)
    if resolved.parent.name != RUNS_DIRNAME:
        return None
    return resolved.parent.parent


def ensure_curriculum_stage_training_dirs(
    run_name: str,
    stage_index: int,
    stage_name: str,
    storage_root: str | Path | None = None,
) -> dict[str, Path]:
    """
    Create and return the standard training directories for one curriculum stage.

    Parameters
    ----------
    run_name
        Stable curriculum run identifier used under ``storage/runs``.
    stage_index
        One-based stage index used in the stage directory prefix.
    stage_name
        Human-readable stage name sanitized into a stable path segment.
    storage_root
        Optional storage root override. Defaults to ``STORAGE_ROOT`` or ``storage``.

    Returns
    -------
    dict[str, Path]
        Mapping from directory labels to created paths for the stage training tree.

    """
    run_dir = get_run_dir(run_name, storage_root)
    stages_dir = run_dir / STAGES_DIRNAME
    stage_dir = get_curriculum_stage_dir(run_name, stage_index, stage_name, storage_root)
    training_dir = get_curriculum_stage_training_dir(run_name, stage_index, stage_name, storage_root)
    paths = {
        "run": run_dir,
        STAGES_DIRNAME: stages_dir,
        "stage": stage_dir,
        TRAINING_DIRNAME: training_dir,
        **{subdir: training_dir / subdir for subdir in _RUN_TRAINING_SUBDIRS},
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def ensure_curriculum_stage_evaluation_dirs(
    run_name: str,
    stage_index: int,
    stage_name: str,
    evaluation_name: str,
    storage_root: str | Path | None = None,
) -> dict[str, Path]:
    """
    Create and return the standard evaluation directories for one curriculum stage.

    Parameters
    ----------
    run_name
        Stable curriculum run identifier used under ``storage/runs``.
    stage_index
        One-based stage index used in the stage directory prefix.
    stage_name
        Human-readable stage name sanitized into a stable path segment.
    evaluation_name
        Stable evaluation identifier used under the stage ``evaluations`` directory.
    storage_root
        Optional storage root override. Defaults to ``STORAGE_ROOT`` or ``storage``.

    Returns
    -------
    dict[str, Path]
        Mapping from directory labels to created paths for the stage evaluation tree.

    """
    run_dir = get_run_dir(run_name, storage_root)
    stages_dir = run_dir / STAGES_DIRNAME
    stage_dir = get_curriculum_stage_dir(run_name, stage_index, stage_name, storage_root)
    evaluations_dir = get_curriculum_stage_evaluations_dir(run_name, stage_index, stage_name, storage_root)
    evaluation_dir = get_curriculum_stage_evaluation_dir(run_name, stage_index, stage_name, evaluation_name, storage_root)
    paths = {
        "run": run_dir,
        STAGES_DIRNAME: stages_dir,
        "stage": stage_dir,
        EVALUATIONS_DIRNAME: evaluations_dir,
        "evaluation": evaluation_dir,
        **{subdir: evaluation_dir / subdir for subdir in _RUN_EVALUATION_SUBDIRS},
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def _storage_root(storage_root: str | Path | None) -> Path:
    """Return an explicit storage root or the configured default."""
    if storage_root is None:
        return get_storage_root()
    return Path(storage_root).expanduser().resolve(strict=False)


def _validate_artifact_name(value: str, label: str) -> str:
    """Validate and sanitize one storage artifact path segment."""
    normalized = value.strip()
    if not normalized:
        message = f"{label} must be non-empty"
        raise ValueError(message)
    candidate_path = Path(normalized)
    if candidate_path.is_absolute() or len(candidate_path.parts) != 1 or any(part in {"", ".", ".."} for part in candidate_path.parts):
        message = f"{label} must be a single relative path segment without traversal"
        raise ValueError(message)
    sanitized = _ARTIFACT_NAME_INVALID_CHARS.sub("_", normalized).strip("._-")
    if not sanitized:
        message = f"{label} must contain at least one safe path character"
        raise ValueError(message)
    if sanitized in {".", ".."}:
        message = f"{label} must not resolve to traversal"
        raise ValueError(message)
    return sanitized


def _stage_dirname(stage_index: int, stage_name: str) -> str:
    """Return the canonical stage directory name for a one-based stage index."""
    if stage_index < 1:
        message = "stage_index must be positive"
        raise ValueError(message)
    return f"stage{stage_index:02d}_{_validate_artifact_name(stage_name, 'stage_name')}"


__all__ = [
    "CONFIG_DIRNAME",
    "CURRICULUM_CONFIG_SNAPSHOT_FILENAME",
    "DIAGNOSTICS_DIRNAME",
    "EVALUATIONS_DIRNAME",
    "EVALUATION_INDEX_FILENAME",
    "EVALUATION_SUITES_DIRNAME",
    "EVALUATION_SUMMARY_FILENAME",
    "LLM_LOGS_DIRNAME",
    "LOGS_DIRNAME",
    "MANIFESTS_DIRNAME",
    "MANIFEST_FILENAME",
    "METRICS_DIRNAME",
    "MODELS_DIRNAME",
    "PLOTS_DIRNAME",
    "RENDERS_DIRNAME",
    "RUNS_DIRNAME",
    "RUN_MANIFEST_FILENAME",
    "STAGES_DIRNAME",
    "TASK_CONFIG_SNAPSHOT_FILENAME",
    "TRACES_DIRNAME",
    "TRAINING_CONFIG_SNAPSHOT_FILENAME",
    "TRAINING_DIRNAME",
    "WANDB_DIRNAME",
    "ensure_curriculum_stage_evaluation_dirs",
    "ensure_curriculum_stage_training_dirs",
    "ensure_run_evaluation_dirs",
    "ensure_run_training_dirs",
    "get_curriculum_stage_dir",
    "get_curriculum_stage_evaluation_dir",
    "get_curriculum_stage_evaluations_dir",
    "get_curriculum_stage_training_dir",
    "get_curriculum_stage_training_manifest_path",
    "get_run_config_dir",
    "get_run_config_evaluation_suites_dir",
    "get_run_curriculum_config_snapshot_path",
    "get_run_dir",
    "get_run_evaluation_diagnostics_dir",
    "get_run_evaluation_dir",
    "get_run_evaluation_index_path",
    "get_run_evaluation_manifests_dir",
    "get_run_evaluation_metrics_dir",
    "get_run_evaluation_plots_dir",
    "get_run_evaluation_renders_dir",
    "get_run_evaluation_summary_path",
    "get_run_evaluation_traces_dir",
    "get_run_evaluations_dir",
    "get_run_manifest_path",
    "get_run_task_config_snapshot_path",
    "get_run_training_config_snapshot_path",
    "get_run_training_diagnostics_dir",
    "get_run_training_dir",
    "get_run_training_logs_dir",
    "get_run_training_manifest_path",
    "get_run_training_metrics_dir",
    "get_run_training_models_dir",
    "get_run_training_wandb_dir",
    "get_storage_root",
    "path_relative_to",
    "path_relative_to_run",
    "storage_root_from_run_dir",
]
