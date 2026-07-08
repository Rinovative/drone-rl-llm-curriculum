"""
===============================================================================
experiments_curriculum_training.py
===============================================================================
Train PPO trajectory tracking through a fixed manual curriculum.

Responsibilities:
  - Load and validate manual curriculum training configurations
  - Materialize explicit per-stage task configs for existing PPO training helpers
  - Run PPO stages sequentially with optional model transfer
  - Write compact curriculum summaries and manifests without duplicating traces

Design principles:
  - Reuse PPO training, diagnostics, W&B, and artifact helpers
  - Keep manual curriculum orchestration separate from LLM curriculum logic
  - Fail before training when any configured stage task is invalid

Boundaries:
  - Reward logic, action semantics, environment physics, and rendering stay elsewhere
  - LLM task proposal and repair are not part of this module
===============================================================================

"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src import utils, validation
from src.experiments import experiments_config as config_loader
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking

DEFAULT_CURRICULUM_CONFIG_PATH = Path("configs/curricula/curriculum_manual_line_smoke.yaml")
MANUAL_CURRICULUM_KIND = "manual"
SUMMARY_METRIC_KEYS = (
    "start_hold_enabled",
    "start_hold_sec",
    "exclude_start_hold_from_tracking_metrics",
    "tracking_phase_start_step",
    "tracking_phase_start_time_sec",
    "mean_position_error_m",
    "mean_position_error_tracking_m",
    "final_position_error_m",
    "max_position_error_m",
    "actual_xy_span_m",
    "reference_xy_span_m",
    "xy_tracking_ratio",
    "action_saturation_fraction",
    "real_action_saturation_fraction",
    "failure_overall_status",
    "failure_primary_mode",
    "failure_modes",
    "curriculum_readiness_level",
    "curriculum_recommended_next_tasks",
    "curriculum_avoid_next_tasks",
)


@dataclass(frozen=True)
class ManualCurriculumStage:
    """
    One stage in a manual PPO tracking curriculum.

    Parameters
    ----------
    stage_name
        Stable human-readable stage identifier used in run names and W&B tags.
    task_shape
        Expected trajectory task shape for this stage.
    task
        Explicit task mapping passed through deterministic validation and PPO training.
    total_timesteps
        PPO timestep budget for this stage.
    eval_steps
        Deterministic evaluation steps after this stage trains.
    notes
        Optional rationale or operator notes copied into summaries.

    """

    stage_name: str
    task_shape: str
    task: dict[str, Any]
    total_timesteps: int
    eval_steps: int
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate stage metadata that does not require trajectory sampling."""
        if not self.stage_name.strip():
            message = "stage_name must be non-empty"
            raise ValueError(message)
        if not self.task_shape.strip():
            message = "task_shape must be non-empty"
            raise ValueError(message)
        if self.total_timesteps <= 0:
            message = "stage total_timesteps must be positive"
            raise ValueError(message)
        if self.eval_steps <= 0:
            message = "stage eval_steps must be positive"
            raise ValueError(message)
        if self.task.get(validation.contracts.FIELD_SHAPE) != self.task_shape:
            message = f"stage {self.stage_name!r} task shape must match task_shape {self.task_shape!r}"
            raise ValueError(message)


@dataclass(frozen=True)
class ManualCurriculumSettings:
    """
    Settings for sequential manual PPO curriculum training.

    Parameters
    ----------
    curriculum_name
        Stable curriculum identifier used in stage run names and summary artifacts.
    base_training_config
        Existing PPO tracking config used as defaults for stage training.
    seed
        Default deterministic seed for all stages.
    wandb_mode
        W&B mode override for every stage.
    normalize_actions
        Whether every stage should use the normalized PPO action interface.
    stages
        Ordered manual curriculum stages.
    config_path
        Optional source path copied into summary metadata.

    """

    curriculum_name: str
    base_training_config: Path
    seed: int
    wandb_mode: str
    normalize_actions: bool
    stages: tuple[ManualCurriculumStage, ...]
    config_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate curriculum-level metadata."""
        if not self.curriculum_name.strip():
            message = "curriculum_name must be non-empty"
            raise ValueError(message)
        utils.artifacts.get_run_dir(_curriculum_artifact_run_name(self.curriculum_name, self.seed))
        if self.wandb_mode not in utils.wandb.WANDB_MODES:
            message = f"wandb_mode must be one of: {', '.join(utils.wandb.WANDB_MODES)}"
            raise ValueError(message)
        if not self.stages:
            message = "manual curriculum requires at least one stage"
            raise ValueError(message)


@dataclass(frozen=True)
class ManualCurriculumResult:
    """
    Result returned after a manual curriculum training run.

    Parameters
    ----------
    summary_path
        Path to the written curriculum summary JSON.
    manifest_path
        Path to the written curriculum manifest JSON.
    summary
        JSON-serializable curriculum summary payload.

    """

    summary_path: str
    manifest_path: str
    summary: dict[str, Any]


def load_manual_curriculum_settings(path: str | Path) -> ManualCurriculumSettings:
    """
    Load manual curriculum training settings from YAML.

    Parameters
    ----------
    path
        Curriculum YAML path.

    Returns
    -------
    ManualCurriculumSettings
        Validated curriculum settings.

    """
    config_path = Path(path)
    config = config_loader.load_experiment_config(config_path)
    return manual_curriculum_settings_from_mapping(config, config_path=config_path)


def manual_curriculum_settings_from_mapping(
    config: Mapping[str, Any],
    config_path: Path | None = None,
) -> ManualCurriculumSettings:
    """
    Build manual curriculum settings from a loaded mapping.

    Parameters
    ----------
    config
        Loaded curriculum configuration mapping.
    config_path
        Optional source path copied into metadata.

    Returns
    -------
    ManualCurriculumSettings
        Validated manual curriculum settings.

    Raises
    ------
    ValueError
        If required curriculum or stage fields are missing or malformed.

    """
    stages_raw = config.get("stages")
    if not isinstance(stages_raw, list):
        message = "curriculum config must contain a top-level 'stages' list"
        raise TypeError(message)
    stages = tuple(_stage_from_mapping(index, stage) for index, stage in enumerate(stages_raw, start=1))
    return ManualCurriculumSettings(
        curriculum_name=str(config.get("curriculum_name") or ""),
        base_training_config=Path(str(config.get("base_training_config") or ppo_tracking.DEFAULT_PPO_TRACKING_CONFIG_PATH)),
        seed=int(config.get("seed", ppo_tracking.DEFAULT_SEED)),
        wandb_mode=str(config.get("wandb_mode") or utils.wandb.WANDB_MODE_AUTO),
        normalize_actions=bool(config.get("normalize_actions", ppo_tracking.DEFAULT_NORMALIZE_ACTIONS)),
        stages=stages,
        config_path=config_path,
    )


def validate_manual_curriculum(settings: ManualCurriculumSettings) -> None:
    """
    Validate every stage task before training starts.

    Parameters
    ----------
    settings
        Loaded manual curriculum settings.

    Raises
    ------
    ValueError
        If any stage task is rejected by deterministic validation.

    """
    for stage in settings.stages:
        result = validation.tasks.validate_task(stage.task)
        if not result.is_valid:
            details = "; ".join(result.messages)
            message = f"invalid curriculum stage {stage.stage_name!r}: {details}"
            raise ValueError(message)


def derive_stage_run_name(curriculum_name: str, stage_index: int, stage_name: str, seed: int) -> str:
    """
    Derive the stable run name for one manual curriculum stage.

    Parameters
    ----------
    curriculum_name
        Curriculum identifier.
    stage_index
        One-based stage index.
    stage_name
        Stage identifier.
    seed
        Deterministic seed.

    Returns
    -------
    str
        Stage run name used in metadata and W&B tracking.

    """
    return f"{curriculum_name}_stage{stage_index:02d}_{stage_name}_seed{seed}"


def run_manual_curriculum_training(settings: ManualCurriculumSettings) -> ManualCurriculumResult:
    """
    Train PPO tracking sequentially across all manual curriculum stages.

    Parameters
    ----------
    settings
        Loaded and validated manual curriculum settings.

    Returns
    -------
    ManualCurriculumResult
        Summary and manifest paths plus the summary payload.

    """
    validate_manual_curriculum(settings)
    stage_entries: list[dict[str, Any]] = []
    previous_model_path: str | None = None
    transfer_used = False

    for stage_index, stage in enumerate(settings.stages, start=1):
        run_name = derive_stage_run_name(settings.curriculum_name, stage_index, stage.stage_name, settings.seed)
        curriculum_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
        stage_dirs = utils.artifacts.ensure_curriculum_stage_training_dirs(curriculum_run_name, stage_index, stage.stage_name)
        stage_training_dir = stage_dirs[utils.artifacts.TRAINING_DIRNAME]
        task_config_path = _write_stage_task_config(
            settings=settings,
            stage=stage,
            stage_index=stage_index,
            stage_training_dir=stage_training_dir,
        )
        initial_model_path = previous_model_path
        result = ppo_tracking.run_ppo_tracking_smoke_from_config(
            config_path=settings.base_training_config,
            task_config_path=task_config_path,
            task_index=0,
            task_shape=stage.task_shape,
            run_name=run_name,
            total_timesteps=stage.total_timesteps,
            artifact_root=stage_training_dir,
            eval_steps=stage.eval_steps,
            seed=settings.seed,
            wandb_mode=settings.wandb_mode,
            normalize_actions=settings.normalize_actions,
            wandb_group=_curriculum_wandb_group(settings.curriculum_name),
            wandb_tags=("curriculum", "manual", f"stage:{stage.stage_name}", f"task:{stage.task_shape}"),
            initial_model_path=initial_model_path,
        )
        transfer_used = transfer_used or initial_model_path is not None
        entry = _stage_summary_entry(
            stage_index=stage_index,
            stage=stage,
            run_name=run_name,
            result=result,
            training_dir=stage_training_dir,
            previous_model_path=previous_model_path,
            initial_model_path=initial_model_path,
            normalize_actions=settings.normalize_actions,
        )
        stage_entries.append(entry)
        previous_model_path = result.model_path

    summary = _build_curriculum_summary(settings=settings, stage_entries=stage_entries, model_transfer_enabled=transfer_used)
    summary_path, manifest_path = _write_curriculum_artifacts(settings=settings, summary=summary)
    return ManualCurriculumResult(summary_path=str(summary_path), manifest_path=str(manifest_path), summary=summary)


def run_manual_curriculum_training_from_config(
    config_path: str | Path = DEFAULT_CURRICULUM_CONFIG_PATH,
    seed: int | None = None,
    wandb_mode: str | None = None,
) -> ManualCurriculumResult:
    """
    Load curriculum settings with CLI-style overrides and run training.

    Parameters
    ----------
    config_path
        Curriculum YAML path.
    seed
        Optional deterministic seed override.
    wandb_mode
        Optional W&B mode override.

    Returns
    -------
    ManualCurriculumResult
        Summary and manifest metadata for the curriculum run.

    """
    settings = load_manual_curriculum_settings(config_path)
    overridden = ManualCurriculumSettings(
        curriculum_name=settings.curriculum_name,
        base_training_config=settings.base_training_config,
        seed=settings.seed if seed is None else seed,
        wandb_mode=settings.wandb_mode if wandb_mode is None else wandb_mode,
        normalize_actions=settings.normalize_actions,
        stages=settings.stages,
        config_path=settings.config_path,
    )
    return run_manual_curriculum_training(overridden)


def _stage_from_mapping(index: int, raw_stage: Any) -> ManualCurriculumStage:
    """Return a validated stage from one raw YAML mapping."""
    if not isinstance(raw_stage, Mapping):
        message = f"stage {index} must be a mapping"
        raise TypeError(message)
    task = raw_stage.get("task")
    if not isinstance(task, Mapping):
        message = f"stage {index} must contain an explicit task mapping"
        raise TypeError(message)
    return ManualCurriculumStage(
        stage_name=str(raw_stage.get("stage_name") or ""),
        task_shape=str(raw_stage.get("task_shape") or ""),
        task=dict(task),
        total_timesteps=int(raw_stage.get("total_timesteps", 0)),
        eval_steps=int(raw_stage.get("eval_steps", 0)),
        notes=str(raw_stage["notes"]) if raw_stage.get("notes") is not None else None,
    )


def _write_stage_task_config(
    settings: ManualCurriculumSettings,
    stage: ManualCurriculumStage,
    stage_index: int,
    stage_training_dir: Path,
) -> Path:
    """Write the one-task config consumed by the existing PPO helper."""
    config_dir = _curriculum_config_dir(settings)
    config_dir.mkdir(parents=True, exist_ok=True)
    task_config_path = config_dir / f"stage{stage_index:02d}_{stage.stage_name}_task.yaml"
    stage_training_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "name": f"{settings.curriculum_name}_stage{stage_index:02d}",
        "seed": settings.seed,
        "tasks": [stage.task],
    }
    task_config_path.write_text(_to_yaml(payload), encoding="utf-8")
    return task_config_path


def _stage_summary_entry(
    stage_index: int,
    stage: ManualCurriculumStage,
    run_name: str,
    result: ppo_tracking.PPOTrackingSmokeResult,
    training_dir: Path,
    previous_model_path: str | None,
    initial_model_path: str | None,
    normalize_actions: bool,
) -> dict[str, Any]:
    """Build one compact stage summary entry from PPO metrics."""
    metrics = result.metrics
    run_root = training_dir.parent.parent.parent
    entry: dict[str, Any] = {
        "stage_index": stage_index,
        "stage_name": stage.stage_name,
        "task_shape": stage.task_shape,
        "run_name": run_name,
        "stage_dir": str(training_dir.parent),
        "stage_dir_relative": utils.artifacts.path_relative_to(training_dir.parent, run_root),
        "training_dir": str(training_dir),
        "training_dir_relative": utils.artifacts.path_relative_to(training_dir, run_root),
        "model_path": result.model_path,
        "model_path_relative": utils.artifacts.path_relative_to(result.model_path, run_root),
        "metrics_path": result.metrics_path,
        "metrics_path_relative": utils.artifacts.path_relative_to(result.metrics_path, run_root),
        "manifest_path": result.manifest_path,
        "manifest_path_relative": utils.artifacts.path_relative_to(result.manifest_path, run_root),
        "diagnostics_dir": metrics.get("diagnostics_dir"),
        "diagnostics_dir_relative": utils.artifacts.path_relative_to(metrics.get("diagnostics_dir"), run_root),
        "total_timesteps": stage.total_timesteps,
        "eval_steps": stage.eval_steps,
        "seed": metrics.get("seed"),
        "normalize_actions": normalize_actions,
        "initial_model_path": initial_model_path,
        "initial_model_path_relative": utils.artifacts.path_relative_to(initial_model_path, run_root),
        "previous_model_path": previous_model_path,
        "previous_model_path_relative": utils.artifacts.path_relative_to(previous_model_path, run_root),
        "model_transfer_enabled": initial_model_path is not None,
        "model_transfer_source": initial_model_path,
        "model_transfer_source_relative": utils.artifacts.path_relative_to(initial_model_path, run_root),
    }
    for key in SUMMARY_METRIC_KEYS:
        entry[key] = metrics.get(key)
    return entry


def _build_curriculum_summary(
    settings: ManualCurriculumSettings,
    stage_entries: Sequence[dict[str, Any]],
    model_transfer_enabled: bool,
) -> dict[str, Any]:
    """Build the curriculum-level JSON summary payload."""
    final_stage = stage_entries[-1]
    run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    run_manifest_path = utils.artifacts.get_run_manifest_path(run_name)
    return {
        "run_type": "training",
        "run_kind": "curriculum",
        "curriculum_kind": MANUAL_CURRICULUM_KIND,
        "mode": "manual_curriculum",
        "curriculum_name": settings.curriculum_name,
        "run_name": run_name,
        "run_manifest_path": str(run_manifest_path),
        "config_path": str(settings.config_path) if settings.config_path is not None else None,
        "base_training_config": str(settings.base_training_config),
        "seed": settings.seed,
        "stage_count": len(stage_entries),
        "model_transfer_enabled": model_transfer_enabled,
        "final_stage_run_name": final_stage["run_name"],
        "final_model_path": final_stage["model_path"],
        "final_stage": {
            "stage_index": final_stage["stage_index"],
            "stage_name": final_stage["stage_name"],
            "run_name": final_stage["run_name"],
            "model_path": final_stage["model_path"],
            "model_path_relative": final_stage.get("model_path_relative"),
            "manifest_path": final_stage["manifest_path"],
            "manifest_path_relative": final_stage.get("manifest_path_relative"),
        },
        "stages": list(stage_entries),
    }


def _write_curriculum_artifacts(settings: ManualCurriculumSettings, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write the canonical curriculum run manifest JSON."""
    artifact_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    curriculum_root = _curriculum_artifact_root(settings)
    manifest_path = utils.artifacts.get_run_manifest_path(artifact_run_name)
    curriculum_root.mkdir(parents=True, exist_ok=True)
    _curriculum_config_dir(settings).mkdir(parents=True, exist_ok=True)
    config_snapshot_path = _write_curriculum_config_snapshot(settings)
    manifest = {
        **summary,
        "artifact_root": str(curriculum_root),
        "artifact_root_relative": ".",
        "summary_path": str(manifest_path),
        "summary_path_relative": utils.artifacts.path_relative_to(manifest_path, curriculum_root),
        "manifest_path": str(manifest_path),
        "manifest_path_relative": utils.artifacts.path_relative_to(manifest_path, curriculum_root),
        "curriculum_config_snapshot_path": str(config_snapshot_path),
        "curriculum_config_snapshot_path_relative": utils.artifacts.path_relative_to(config_snapshot_path, curriculum_root),
        "config": {
            "curriculum_config_path": str(settings.config_path) if settings.config_path is not None else None,
            "curriculum_config_snapshot_path": str(config_snapshot_path),
            "curriculum_config_snapshot_path_relative": utils.artifacts.path_relative_to(config_snapshot_path, curriculum_root),
            "base_training_config": str(settings.base_training_config),
        },
        "evaluation_index": _evaluation_index_manifest(artifact_run_name),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, manifest_path


def _write_curriculum_config_snapshot(settings: ManualCurriculumSettings) -> Path:
    """Copy or materialize the curriculum config snapshot for a run."""
    artifact_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    snapshot_path = utils.artifacts.get_run_curriculum_config_snapshot_path(artifact_run_name)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    if settings.config_path is not None and settings.config_path.is_file():
        snapshot_path.write_text(settings.config_path.read_text(encoding="utf-8"), encoding="utf-8")
        return snapshot_path
    payload = {
        "curriculum_name": settings.curriculum_name,
        "base_training_config": str(settings.base_training_config),
        "seed": settings.seed,
        "wandb_mode": settings.wandb_mode,
        "normalize_actions": settings.normalize_actions,
        "stages": [
            {
                "stage_name": stage.stage_name,
                "task_shape": stage.task_shape,
                "total_timesteps": stage.total_timesteps,
                "eval_steps": stage.eval_steps,
                "notes": stage.notes,
                "task": stage.task,
            }
            for stage in settings.stages
        ],
    }
    snapshot_path.write_text(_to_yaml(payload), encoding="utf-8")
    return snapshot_path


def _evaluation_index_manifest(run_name: str) -> dict[str, Any]:
    """Return the run-manifest link to the deterministic evaluation index."""
    index_path = utils.artifacts.get_run_evaluation_index_path(run_name)
    entries: list[Any] = []
    if index_path.exists():
        try:
            payload = json.loads(index_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        raw_entries = payload.get("evaluations") if isinstance(payload, dict) else None
        entries = raw_entries if isinstance(raw_entries, list) else []
    return {
        "path": str(index_path),
        "path_relative": utils.artifacts.path_relative_to_run(index_path, run_name),
        "entry_count": len(entries),
        "evaluations": entries,
    }


def _curriculum_artifact_run_name(curriculum_name: str, seed: int, curriculum_kind: str = MANUAL_CURRICULUM_KIND) -> str:
    """Return the self-describing storage run name used for curriculum-level artifacts."""
    topic = _curriculum_run_topic(curriculum_name, curriculum_kind)
    return f"curriculum_{curriculum_kind}_{topic}_seed{seed}"


def _curriculum_run_topic(curriculum_name: str, curriculum_kind: str) -> str:
    """Return the curriculum topic without a duplicated curriculum/kind prefix."""
    canonical_prefix = f"curriculum_{curriculum_kind}_"
    if curriculum_name.startswith(canonical_prefix):
        return curriculum_name[len(canonical_prefix) :]
    kind_prefix = f"{curriculum_kind}_"
    if curriculum_name.startswith(kind_prefix):
        return curriculum_name[len(kind_prefix) :]
    return curriculum_name


def _curriculum_artifact_root(settings: ManualCurriculumSettings) -> Path:
    """Return the canonical curriculum run root."""
    return utils.artifacts.get_run_dir(_curriculum_artifact_run_name(settings.curriculum_name, settings.seed))


def _curriculum_stage_artifact_root(settings: ManualCurriculumSettings, stage_index: int, stage_name: str) -> Path:
    """Return the training artifact root for one curriculum stage."""
    return utils.artifacts.get_curriculum_stage_training_dir(
        _curriculum_artifact_run_name(settings.curriculum_name, settings.seed),
        stage_index,
        stage_name,
    )


def _curriculum_config_dir(settings: ManualCurriculumSettings) -> Path:
    """Return the curriculum-level generated config directory."""
    return utils.artifacts.get_run_config_dir(_curriculum_artifact_run_name(settings.curriculum_name, settings.seed))


def _curriculum_wandb_group(curriculum_name: str) -> str:
    """Return the W&B group used for all stages in one curriculum."""
    return f"curriculum/{curriculum_name}"


def _to_yaml(payload: Mapping[str, Any]) -> str:
    """Serialize a small task config to YAML with the project dependency."""
    return yaml.safe_dump(dict(payload), sort_keys=False)


__all__ = [
    "DEFAULT_CURRICULUM_CONFIG_PATH",
    "ManualCurriculumResult",
    "ManualCurriculumSettings",
    "ManualCurriculumStage",
    "derive_stage_run_name",
    "load_manual_curriculum_settings",
    "manual_curriculum_settings_from_mapping",
    "run_manual_curriculum_training",
    "run_manual_curriculum_training_from_config",
    "validate_manual_curriculum",
]
