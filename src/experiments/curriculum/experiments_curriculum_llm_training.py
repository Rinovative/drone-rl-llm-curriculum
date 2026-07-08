"""
===============================================================================
experiments_curriculum_llm_training.py
===============================================================================
Train PPO trajectory tracking through a local-LLM-guided curriculum.

Responsibilities:
  - Load and validate LLM curriculum training configurations
  - Use strict JSON LLM proposals with deterministic validation and repair
  - Materialize per-stage task configs for the existing PPO tracking helper
  - Write run-scoped manifests, stage summaries, and proposal JSONL logs

Design principles:
  - Reuse PPO training, W&B, diagnostics, and canonical artifact helpers
  - Keep the LLM as a task proposer, never a runtime controller
  - Make dry-run proposal checks deterministic and training-free

Boundaries:
  - Reward logic, action semantics, environment physics, and evaluation metrics stay elsewhere
  - External local-LLM infrastructure and server startup are not managed here
===============================================================================

"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from src import llm, utils, validation
from src.experiments import experiments_config as config_loader
from src.experiments.curriculum import experiments_curriculum_training as manual_curriculum
from src.experiments.training import experiments_training_ppo_tracking as ppo_tracking

DEFAULT_LLM_CURRICULUM_CONFIG_PATH = Path("configs/curricula/curriculum_llm_smoke.yaml")
LLM_CURRICULUM_KIND = "llm"
LLM_CURRICULUM_MODE = "llm_curriculum"
DEFAULT_RECENT_CONTEXT_LIMIT = 3
DEFAULT_STAGE_TOTAL_TIMESTEPS = ppo_tracking.DEFAULT_TOTAL_TIMESTEPS
DEFAULT_STAGE_EVAL_STEPS = ppo_tracking.DEFAULT_EVAL_STEPS


@dataclass(frozen=True)
class LLMCurriculumStage:
    """
    One stage in an LLM-guided PPO tracking curriculum.

    Parameters
    ----------
    stage_name
        Stable human-readable stage identifier used in run names and artifact paths.
    task_shape
        Expected trajectory task shape for this stage.
    task
        Validated task mapping passed to PPO training.
    total_timesteps
        PPO timestep budget for this stage.
    eval_steps
        Deterministic evaluation steps after this stage trains.
    task_reason
        Optional LLM rationale metadata retained in summaries only.
    notes
        Optional operator notes copied into summaries.

    """

    stage_name: str
    task_shape: str
    task: dict[str, Any]
    total_timesteps: int
    eval_steps: int
    task_reason: str | None = None
    notes: str | None = None

    def __post_init__(self) -> None:
        """Validate stage metadata that does not require PPO training."""
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
class LLMCurriculumSettings:
    """
    Settings for sequential LLM-guided PPO curriculum training.

    Parameters
    ----------
    curriculum_name
        Stable curriculum identifier used in run names and summary artifacts.
    base_training_config
        Existing PPO tracking config used as defaults for stage training.
    seed
        Default deterministic seed for all stages.
    wandb_mode
        W&B mode override for every stage.
    normalize_actions
        Whether every stage should use the normalized PPO action interface.
    max_stages
        Maximum number of curriculum stages including an enabled bootstrap stage.
    stage_total_timesteps
        Default timestep budget for LLM-proposed stages.
    stage_eval_steps
        Default evaluation budget for LLM-proposed stages.
    bootstrap_stage
        Optional first stage supplied by configuration before LLM proposals begin.
    llm_config
        Provider configuration used to construct the LLM client.
    proposal_settings
        Strict JSON repair and prompt-context settings.
    config_path
        Optional source config path included in summary metadata.

    """

    curriculum_name: str
    base_training_config: Path
    seed: int
    wandb_mode: str
    normalize_actions: bool
    max_stages: int
    stage_total_timesteps: int
    stage_eval_steps: int
    bootstrap_stage: LLMCurriculumStage | None
    llm_config: dict[str, Any]
    proposal_settings: llm.curriculum.ProposalSettings
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
        if self.max_stages <= 0:
            message = "max_stages must be positive"
            raise ValueError(message)
        if self.stage_total_timesteps <= 0:
            message = "stage_total_timesteps must be positive"
            raise ValueError(message)
        if self.stage_eval_steps <= 0:
            message = "stage_eval_steps must be positive"
            raise ValueError(message)
        if not isinstance(self.llm_config, dict):
            message = "llm_config must be a dictionary"
            raise TypeError(message)

    @property
    def llm_provider(self) -> str:
        """Return the configured LLM provider name."""
        return str(self.llm_config.get("provider") or llm.client.PROVIDER_MOCK)

    @property
    def llm_model(self) -> str | None:
        """Return the configured LLM model name when available."""
        model = self.llm_config.get("model")
        return str(model) if model is not None else None


@dataclass(frozen=True)
class LLMCurriculumResult:
    """
    Result returned after an LLM curriculum training or dry-run proposal check.

    Parameters
    ----------
    summary_path
        Path to the written curriculum summary JSON.
    manifest_path
        Path to the written curriculum manifest JSON.
    proposal_log_path
        Path to the run-scoped proposal JSONL log.
    summary
        JSON-serializable curriculum summary payload.

    """

    summary_path: str
    manifest_path: str
    proposal_log_path: str
    summary: dict[str, Any]


def load_llm_curriculum_settings(path: str | Path) -> LLMCurriculumSettings:
    """
    Load LLM curriculum training settings from YAML.

    Parameters
    ----------
    path
        Curriculum YAML path.

    Returns
    -------
    LLMCurriculumSettings
        Validated curriculum settings.

    """
    config_path = Path(path)
    config = config_loader.load_experiment_config(config_path)
    return llm_curriculum_settings_from_mapping(config, config_path=config_path)


def llm_curriculum_settings_from_mapping(
    config: Mapping[str, Any],
    config_path: Path | None = None,
) -> LLMCurriculumSettings:
    """
    Build LLM curriculum settings from a loaded mapping.

    Parameters
    ----------
    config
        Loaded curriculum configuration mapping.
    config_path
        Optional source path copied into metadata.

    Returns
    -------
    LLMCurriculumSettings
        Validated LLM curriculum settings.

    Raises
    ------
    TypeError
        If nested config sections have invalid types.
    ValueError
        If required curriculum fields are missing or malformed.

    """
    stage_defaults = _mapping_or_empty(config.get("stage_defaults"), "stage_defaults")
    stage_total_timesteps = int(stage_defaults.get("total_timesteps", DEFAULT_STAGE_TOTAL_TIMESTEPS))
    stage_eval_steps = int(stage_defaults.get("eval_steps", DEFAULT_STAGE_EVAL_STEPS))
    llm_config = _llm_config_from_mapping(config)
    proposal_settings = llm.curriculum.ProposalSettings(
        max_repair_attempts=int(llm_config.get("max_repair_attempts", 1)),
        skip_invalid_proposals=bool(llm_config.get("skip_invalid_proposals", False)),
        recent_context_limit=int(llm_config.get("recent_context_limit", DEFAULT_RECENT_CONTEXT_LIMIT)),
    )
    return LLMCurriculumSettings(
        curriculum_name=str(config.get("curriculum_name") or ""),
        base_training_config=Path(str(config.get("base_training_config") or ppo_tracking.DEFAULT_PPO_TRACKING_CONFIG_PATH)),
        seed=int(config.get("seed", ppo_tracking.DEFAULT_SEED)),
        wandb_mode=str(config.get("wandb_mode") or utils.wandb.WANDB_MODE_AUTO),
        normalize_actions=bool(config.get("normalize_actions", ppo_tracking.DEFAULT_NORMALIZE_ACTIONS)),
        max_stages=int(config.get("max_stages", 1)),
        stage_total_timesteps=stage_total_timesteps,
        stage_eval_steps=stage_eval_steps,
        bootstrap_stage=_bootstrap_stage_from_config(config.get("bootstrap"), stage_total_timesteps, stage_eval_steps),
        llm_config=llm_config,
        proposal_settings=proposal_settings,
        config_path=config_path,
    )


def validate_llm_curriculum(settings: LLMCurriculumSettings) -> None:
    """
    Validate bootstrap task and provider configuration before training starts.

    Parameters
    ----------
    settings
        Loaded LLM curriculum settings.

    Raises
    ------
    ValueError
        If configured tasks or provider settings are invalid.

    """
    if settings.bootstrap_stage is not None:
        _validate_stage_task(settings.bootstrap_stage)
    llm.client.client_from_config(settings.llm_config)


def derive_stage_run_name(curriculum_name: str, stage_index: int, stage_name: str, seed: int) -> str:
    """
    Derive the stable run name for one LLM curriculum stage.

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


def run_llm_curriculum_training(settings: LLMCurriculumSettings, dry_run_proposals: bool = False) -> LLMCurriculumResult:
    """
    Train PPO tracking sequentially through an LLM-guided curriculum.

    Parameters
    ----------
    settings
        Loaded and validated LLM curriculum settings.
    dry_run_proposals
        If true, exercise proposal parsing, repair, validation, and logging
        without launching PPO training.

    Returns
    -------
    LLMCurriculumResult
        Summary, manifest, proposal-log paths, and summary payload.

    """
    validate_llm_curriculum(settings)
    curriculum_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    utils.artifacts.get_run_dir(curriculum_run_name).mkdir(parents=True, exist_ok=True)
    utils.artifacts.get_run_config_dir(curriculum_run_name).mkdir(parents=True, exist_ok=True)
    llm_logs_dir = utils.artifacts.ensure_run_llm_logs_dir(curriculum_run_name)
    proposal_log_path = llm_logs_dir / utils.artifacts.LLM_PROPOSALS_FILENAME
    proposal_log_path.write_text("", encoding="utf-8")
    proposal_logger = llm.logging.ProposalEventLogger(proposal_log_path)
    client = llm.client.client_from_config(settings.llm_config)

    stage_entries: list[dict[str, Any]] = []
    recent_accepted_tasks: list[dict[str, Any]] = []
    recent_rejected_tasks: list[dict[str, Any]] = []
    proposal_stats = llm.curriculum.empty_proposal_stats()
    previous_model_path: str | None = None
    latest_metrics_summary: dict[str, Any] = {"status": "not_started", "dry_run_proposals": dry_run_proposals}

    if settings.bootstrap_stage is not None and len(stage_entries) < settings.max_stages:
        entry, previous_model_path, latest_metrics_summary = _run_or_dry_stage(
            settings=settings,
            stage=settings.bootstrap_stage,
            stage_index=1,
            previous_model_path=previous_model_path,
            dry_run_proposals=dry_run_proposals,
        )
        stage_entries.append(entry)
        recent_accepted_tasks.append(_accepted_task_context(entry))

    while len(stage_entries) < settings.max_stages:
        next_stage_index = len(stage_entries) + 1
        context = llm.curriculum.ProposalContext(
            curriculum_name=settings.curriculum_name,
            stage_index=next_stage_index,
            recent_accepted_tasks=tuple(recent_accepted_tasks),
            recent_rejected_tasks=tuple(recent_rejected_tasks),
            metrics_summary=latest_metrics_summary,
        )
        proposal = llm.curriculum.propose_next_task(client=client, context=context, settings=settings.proposal_settings, logger=proposal_logger)
        llm.curriculum.merge_proposal_stats(proposal_stats, proposal.stats)
        recent_rejected_tasks.extend(proposal.rejected_proposals)
        if proposal.task is None:
            break
        stage = _stage_from_proposal(settings=settings, task=proposal.task, task_reason=proposal.task_reason)
        _validate_stage_task(stage)
        entry, previous_model_path, latest_metrics_summary = _run_or_dry_stage(
            settings=settings,
            stage=stage,
            stage_index=next_stage_index,
            previous_model_path=previous_model_path,
            dry_run_proposals=dry_run_proposals,
        )
        stage_entries.append(entry)
        recent_accepted_tasks.append(_accepted_task_context(entry))

    summary = _build_curriculum_summary(
        settings=settings,
        stage_entries=stage_entries,
        proposal_log_path=proposal_log_path,
        proposal_stats=proposal_stats,
        dry_run_proposals=dry_run_proposals,
    )
    summary_path, manifest_path = _write_curriculum_artifacts(settings=settings, summary=summary)
    return LLMCurriculumResult(
        summary_path=str(summary_path),
        manifest_path=str(manifest_path),
        proposal_log_path=str(proposal_log_path),
        summary=summary,
    )


def run_llm_curriculum_training_from_config(
    config_path: str | Path = DEFAULT_LLM_CURRICULUM_CONFIG_PATH,
    seed: int | None = None,
    wandb_mode: str | None = None,
    provider: str | None = None,
    api_base: str | None = None,
    model: str | None = None,
    max_stages: int | None = None,
    max_repair_attempts: int | None = None,
    dry_run_proposals: bool = False,
) -> LLMCurriculumResult:
    """
    Load LLM curriculum settings with CLI-style overrides and run the workflow.

    Parameters
    ----------
    config_path
        Curriculum YAML path.
    seed
        Optional deterministic seed override.
    wandb_mode
        Optional W&B mode override.
    provider
        Optional LLM provider override.
    api_base
        Optional OpenAI-compatible API base override.
    model
        Optional LLM model override.
    max_stages
        Optional maximum stage-count override.
    max_repair_attempts
        Optional repair-attempt override.
    dry_run_proposals
        If true, do not launch PPO training.

    Returns
    -------
    LLMCurriculumResult
        Summary, manifest, and proposal-log metadata for the run.

    """
    settings = load_llm_curriculum_settings(config_path)
    overridden = _settings_with_overrides(
        settings=settings,
        seed=seed,
        wandb_mode=wandb_mode,
        provider=provider,
        api_base=api_base,
        model=model,
        max_stages=max_stages,
        max_repair_attempts=max_repair_attempts,
    )
    return run_llm_curriculum_training(overridden, dry_run_proposals=dry_run_proposals)


def _run_or_dry_stage(
    *,
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
    previous_model_path: str | None,
    dry_run_proposals: bool,
) -> tuple[dict[str, Any], str | None, dict[str, Any]]:
    """Run a PPO stage or produce a dry-run stage summary."""
    curriculum_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    stage_dirs = utils.artifacts.ensure_curriculum_stage_training_dirs(curriculum_run_name, stage_index, stage.stage_name)
    stage_training_dir = stage_dirs[utils.artifacts.TRAINING_DIRNAME]
    task_config_path = _write_stage_task_config(
        settings=settings,
        stage=stage,
        stage_index=stage_index,
        stage_training_dir=stage_training_dir,
    )
    run_name = derive_stage_run_name(settings.curriculum_name, stage_index, stage.stage_name, settings.seed)
    if dry_run_proposals:
        entry = _dry_stage_summary_entry(
            settings=settings,
            stage=stage,
            stage_index=stage_index,
            run_name=run_name,
            training_dir=stage_training_dir,
            previous_model_path=previous_model_path,
            task_config_path=task_config_path,
        )
        return entry, previous_model_path, _metrics_summary_from_entry(entry)

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
        wandb_tags=("curriculum", LLM_CURRICULUM_KIND, f"stage:{stage.stage_name}", f"task:{stage.task_shape}"),
        initial_model_path=previous_model_path,
    )
    entry = _stage_summary_entry(
        settings=settings,
        stage=stage,
        stage_index=stage_index,
        run_name=run_name,
        result=result,
        training_dir=stage_training_dir,
        previous_model_path=previous_model_path,
        initial_model_path=previous_model_path,
        task_config_path=task_config_path,
    )
    return entry, result.model_path, _metrics_summary_from_entry(entry)


def _stage_summary_entry(
    *,
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
    run_name: str,
    result: ppo_tracking.PPOTrackingSmokeResult,
    training_dir: Path,
    previous_model_path: str | None,
    initial_model_path: str | None,
    task_config_path: Path,
) -> dict[str, Any]:
    """Build one compact LLM curriculum stage summary from PPO metrics."""
    metrics = result.metrics
    run_root = training_dir.parent.parent.parent
    entry: dict[str, Any] = {
        "stage_index": stage_index,
        "stage_name": stage.stage_name,
        "task_shape": stage.task_shape,
        "task": stage.task,
        "task_reason": stage.task_reason,
        "notes": stage.notes,
        "run_name": run_name,
        "stage_dir": str(training_dir.parent),
        "stage_dir_relative": utils.artifacts.path_relative_to(training_dir.parent, run_root),
        "training_dir": str(training_dir),
        "training_dir_relative": utils.artifacts.path_relative_to(training_dir, run_root),
        "task_config_path": str(task_config_path),
        "task_config_path_relative": utils.artifacts.path_relative_to(task_config_path, run_root),
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
        "normalize_actions": settings.normalize_actions,
        "initial_model_path": initial_model_path,
        "initial_model_path_relative": utils.artifacts.path_relative_to(initial_model_path, run_root),
        "previous_model_path": previous_model_path,
        "previous_model_path_relative": utils.artifacts.path_relative_to(previous_model_path, run_root),
        "model_transfer_enabled": initial_model_path is not None,
        "model_transfer_source": initial_model_path,
        "model_transfer_source_relative": utils.artifacts.path_relative_to(initial_model_path, run_root),
        "dry_run_proposals": False,
    }
    for key in manual_curriculum.SUMMARY_METRIC_KEYS:
        entry[key] = metrics.get(key)
    return entry


def _dry_stage_summary_entry(
    *,
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
    run_name: str,
    training_dir: Path,
    previous_model_path: str | None,
    task_config_path: Path,
) -> dict[str, Any]:
    """Build one stage summary without launching PPO training."""
    run_root = training_dir.parent.parent.parent
    validation_result = validation.tasks.validate_task(stage.task)
    entry: dict[str, Any] = {
        "stage_index": stage_index,
        "stage_name": stage.stage_name,
        "task_shape": stage.task_shape,
        "task": stage.task,
        "task_reason": stage.task_reason,
        "notes": stage.notes,
        "run_name": run_name,
        "stage_dir": str(training_dir.parent),
        "stage_dir_relative": utils.artifacts.path_relative_to(training_dir.parent, run_root),
        "training_dir": str(training_dir),
        "training_dir_relative": utils.artifacts.path_relative_to(training_dir, run_root),
        "task_config_path": str(task_config_path),
        "task_config_path_relative": utils.artifacts.path_relative_to(task_config_path, run_root),
        "model_path": None,
        "model_path_relative": None,
        "metrics_path": None,
        "metrics_path_relative": None,
        "manifest_path": None,
        "manifest_path_relative": None,
        "diagnostics_dir": None,
        "diagnostics_dir_relative": None,
        "total_timesteps": stage.total_timesteps,
        "eval_steps": stage.eval_steps,
        "seed": settings.seed,
        "normalize_actions": settings.normalize_actions,
        "initial_model_path": previous_model_path,
        "initial_model_path_relative": utils.artifacts.path_relative_to(previous_model_path, run_root),
        "previous_model_path": previous_model_path,
        "previous_model_path_relative": utils.artifacts.path_relative_to(previous_model_path, run_root),
        "model_transfer_enabled": False,
        "model_transfer_source": None,
        "model_transfer_source_relative": None,
        "dry_run_proposals": True,
        "validation_status": "valid" if validation_result.is_valid else "invalid",
        "validation_messages": list(validation_result.messages),
    }
    for key in manual_curriculum.SUMMARY_METRIC_KEYS:
        entry[key] = None
    return entry


def _build_curriculum_summary(
    *,
    settings: LLMCurriculumSettings,
    stage_entries: Sequence[dict[str, Any]],
    proposal_log_path: Path,
    proposal_stats: Mapping[str, Any],
    dry_run_proposals: bool,
) -> dict[str, Any]:
    """Build the curriculum-level JSON summary payload."""
    run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    run_manifest_path = utils.artifacts.get_run_manifest_path(run_name)
    final_stage = stage_entries[-1] if stage_entries else None
    return {
        "run_type": "training",
        "run_kind": "curriculum",
        "curriculum_kind": LLM_CURRICULUM_KIND,
        "mode": LLM_CURRICULUM_MODE,
        "curriculum_name": settings.curriculum_name,
        "run_name": run_name,
        "run_manifest_path": str(run_manifest_path),
        "config_path": str(settings.config_path) if settings.config_path is not None else None,
        "base_training_config": str(settings.base_training_config),
        "seed": settings.seed,
        "stage_count": len(stage_entries),
        "max_stages": settings.max_stages,
        "model_transfer_enabled": any(bool(stage.get("model_transfer_enabled")) for stage in stage_entries),
        "action_interface": final_stage.get("action_interface") if final_stage is not None else None,
        "ppo_action_dim": final_stage.get("ppo_action_dim") if final_stage is not None else None,
        "real_action_type": final_stage.get("real_action_type") if final_stage is not None else None,
        "include_dynamics_observation": final_stage.get("include_dynamics_observation") if final_stage is not None else None,
        "include_previous_action": final_stage.get("include_previous_action") if final_stage is not None else None,
        "observation_dim": final_stage.get("observation_dim") if final_stage is not None else None,
        "observation_components": final_stage.get("observation_components") if final_stage is not None else None,
        "policy_kwargs": final_stage.get("policy_kwargs") if final_stage is not None else None,
        "final_stage_run_name": final_stage.get("run_name") if final_stage is not None else None,
        "final_model_path": final_stage.get("model_path") if final_stage is not None else None,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "proposal_log_path": str(proposal_log_path),
        "proposal_log_path_relative": utils.artifacts.path_relative_to_run(proposal_log_path, run_name),
        "proposal_stats": dict(proposal_stats),
        "dry_run_proposals": dry_run_proposals,
        "final_stage": _final_stage_summary(final_stage),
        "stages": list(stage_entries),
    }


def _write_curriculum_artifacts(settings: LLMCurriculumSettings, summary: dict[str, Any]) -> tuple[Path, Path]:
    """Write the canonical LLM curriculum run manifest JSON."""
    artifact_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    curriculum_root = utils.artifacts.get_run_dir(artifact_run_name)
    manifest_path = utils.artifacts.get_run_manifest_path(artifact_run_name)
    curriculum_root.mkdir(parents=True, exist_ok=True)
    utils.artifacts.get_run_config_dir(artifact_run_name).mkdir(parents=True, exist_ok=True)
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
            "llm_provider": settings.llm_provider,
            "llm_model": settings.llm_model,
        },
        "evaluation_index": _evaluation_index_manifest(artifact_run_name),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path, manifest_path


def _write_curriculum_config_snapshot(settings: LLMCurriculumSettings) -> Path:
    """Materialize a sanitized curriculum config snapshot for a run."""
    artifact_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    snapshot_path = utils.artifacts.get_run_curriculum_config_snapshot_path(artifact_run_name)
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "curriculum_name": settings.curriculum_name,
        "base_training_config": str(settings.base_training_config),
        "seed": settings.seed,
        "wandb_mode": settings.wandb_mode,
        "normalize_actions": settings.normalize_actions,
        "max_stages": settings.max_stages,
        "stage_defaults": {
            "total_timesteps": settings.stage_total_timesteps,
            "eval_steps": settings.stage_eval_steps,
        },
        "bootstrap": _bootstrap_snapshot(settings.bootstrap_stage),
        "llm": _sanitized_llm_config(settings.llm_config),
    }
    snapshot_path.write_text(_to_yaml(payload), encoding="utf-8")
    return snapshot_path


def _settings_with_overrides(
    *,
    settings: LLMCurriculumSettings,
    seed: int | None,
    wandb_mode: str | None,
    provider: str | None,
    api_base: str | None,
    model: str | None,
    max_stages: int | None,
    max_repair_attempts: int | None,
) -> LLMCurriculumSettings:
    """Return settings with CLI-style overrides applied."""
    llm_config = dict(settings.llm_config)
    if provider is not None:
        llm_config["provider"] = provider
    if api_base is not None:
        llm_config["api_base"] = api_base
    if model is not None:
        llm_config["model"] = model
    proposal_settings = settings.proposal_settings
    if max_repair_attempts is not None:
        llm_config["max_repair_attempts"] = max_repair_attempts
        proposal_settings = replace(proposal_settings, max_repair_attempts=max_repair_attempts)
    return LLMCurriculumSettings(
        curriculum_name=settings.curriculum_name,
        base_training_config=settings.base_training_config,
        seed=settings.seed if seed is None else seed,
        wandb_mode=settings.wandb_mode if wandb_mode is None else wandb_mode,
        normalize_actions=settings.normalize_actions,
        max_stages=settings.max_stages if max_stages is None else max_stages,
        stage_total_timesteps=settings.stage_total_timesteps,
        stage_eval_steps=settings.stage_eval_steps,
        bootstrap_stage=settings.bootstrap_stage,
        llm_config=llm_config,
        proposal_settings=proposal_settings,
        config_path=settings.config_path,
    )


def _llm_config_from_mapping(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copied LLM provider configuration mapping."""
    raw_llm_config = config.get("llm")
    if not isinstance(raw_llm_config, Mapping):
        message = "LLM curriculum config must contain a top-level 'llm' mapping"
        raise TypeError(message)
    llm_config = dict(raw_llm_config)
    provider = str(llm_config.get("provider") or llm.client.PROVIDER_MOCK)
    if provider not in llm.client.SUPPORTED_PROVIDERS:
        message = f"llm.provider must be one of: {', '.join(llm.client.SUPPORTED_PROVIDERS)}"
        raise ValueError(message)
    llm_config["provider"] = provider
    return llm_config


def _bootstrap_stage_from_config(raw_bootstrap: Any, default_total_timesteps: int, default_eval_steps: int) -> LLMCurriculumStage | None:
    """Return the optional configured bootstrap stage."""
    if raw_bootstrap is None:
        message = "LLM curriculum config must include bootstrap settings or explicitly disable bootstrap"
        raise ValueError(message)
    if raw_bootstrap is False:
        return None
    if not isinstance(raw_bootstrap, Mapping):
        message = "bootstrap must be a mapping or false"
        raise TypeError(message)
    if not bool(raw_bootstrap.get("enabled", True)):
        return None
    return _stage_from_mapping(raw_bootstrap, default_total_timesteps=default_total_timesteps, default_eval_steps=default_eval_steps)


def _stage_from_mapping(raw_stage: Mapping[str, Any], default_total_timesteps: int, default_eval_steps: int) -> LLMCurriculumStage:
    """Return a validated stage from a raw YAML mapping."""
    task = raw_stage.get("task")
    if not isinstance(task, Mapping):
        message = "stage must contain an explicit task mapping"
        raise TypeError(message)
    task_without_metadata = llm.task_schema.task_without_metadata(dict(task))
    task_shape = str(raw_stage.get("task_shape") or task_without_metadata.get(validation.contracts.FIELD_SHAPE) or "")
    return LLMCurriculumStage(
        stage_name=str(raw_stage.get("stage_name") or task_shape),
        task_shape=task_shape,
        task=task_without_metadata,
        total_timesteps=int(raw_stage.get("total_timesteps", default_total_timesteps)),
        eval_steps=int(raw_stage.get("eval_steps", default_eval_steps)),
        task_reason=str(raw_stage[llm.task_schema.REASON_FIELD]) if raw_stage.get(llm.task_schema.REASON_FIELD) is not None else None,
        notes=str(raw_stage["notes"]) if raw_stage.get("notes") is not None else None,
    )


def _stage_from_proposal(settings: LLMCurriculumSettings, task: dict[str, Any], task_reason: str | None) -> LLMCurriculumStage:
    """Return a stage from one accepted LLM proposal task."""
    task_shape = str(task.get(validation.contracts.FIELD_SHAPE) or "")
    return LLMCurriculumStage(
        stage_name=task_shape,
        task_shape=task_shape,
        task=dict(task),
        total_timesteps=settings.stage_total_timesteps,
        eval_steps=settings.stage_eval_steps,
        task_reason=task_reason,
    )


def _validate_stage_task(stage: LLMCurriculumStage) -> None:
    """Validate a stage task before PPO training or dry-run acceptance."""
    result = validation.tasks.validate_task(stage.task)
    if not result.is_valid:
        details = "; ".join(result.messages)
        message = f"invalid LLM curriculum stage {stage.stage_name!r}: {details}"
        raise ValueError(message)


def _write_stage_task_config(
    settings: LLMCurriculumSettings,
    stage: LLMCurriculumStage,
    stage_index: int,
    stage_training_dir: Path,
) -> Path:
    """Write the one-task config consumed by the existing PPO helper."""
    curriculum_run_name = _curriculum_artifact_run_name(settings.curriculum_name, settings.seed)
    config_dir = utils.artifacts.get_run_config_dir(curriculum_run_name)
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


def _accepted_task_context(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact accepted-task context for future LLM prompts."""
    return {
        "stage_index": entry.get("stage_index"),
        "stage_name": entry.get("stage_name"),
        "task_shape": entry.get("task_shape"),
        "task": entry.get("task"),
        "task_reason": entry.get("task_reason"),
        "metrics": _metrics_summary_from_entry(entry),
    }


def _metrics_summary_from_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact metrics used in subsequent proposal prompts."""
    keys = (
        "stage_index",
        "stage_name",
        "task_shape",
        "dry_run_proposals",
        "validation_status",
        "mean_position_error_m",
        "mean_position_error_tracking_m",
        "final_position_error_m",
        "max_position_error_m",
        "xy_tracking_ratio",
        "failure_overall_status",
        "failure_primary_mode",
        "curriculum_readiness_level",
        "curriculum_recommended_next_tasks",
        "curriculum_avoid_next_tasks",
    )
    return {key: entry.get(key) for key in keys if key in entry}


def _final_stage_summary(final_stage: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the compact final-stage manifest entry."""
    if final_stage is None:
        return None
    return {
        "stage_index": final_stage.get("stage_index"),
        "stage_name": final_stage.get("stage_name"),
        "run_name": final_stage.get("run_name"),
        "model_path": final_stage.get("model_path"),
        "model_path_relative": final_stage.get("model_path_relative"),
        "manifest_path": final_stage.get("manifest_path"),
        "manifest_path_relative": final_stage.get("manifest_path_relative"),
    }


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


def _bootstrap_snapshot(stage: LLMCurriculumStage | None) -> dict[str, Any]:
    """Return sanitized bootstrap settings for config snapshots."""
    if stage is None:
        return {"enabled": False}
    return {
        "enabled": True,
        "stage_name": stage.stage_name,
        "task_shape": stage.task_shape,
        "total_timesteps": stage.total_timesteps,
        "eval_steps": stage.eval_steps,
        "reason": stage.task_reason,
        "notes": stage.notes,
        "task": stage.task,
    }


def _sanitized_llm_config(llm_config: Mapping[str, Any]) -> dict[str, Any]:
    """Return provider config safe for manifests and config snapshots."""
    sanitized = dict(llm_config)
    if sanitized.get("api_key"):
        sanitized["api_key"] = "[REDACTED]"
    return sanitized


def _mapping_or_empty(value: Any, label: str) -> Mapping[str, Any]:
    """Return a mapping config section or an empty mapping."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        message = f"{label} must be a mapping"
        raise TypeError(message)
    return value


def _curriculum_artifact_run_name(curriculum_name: str, seed: int, curriculum_kind: str = LLM_CURRICULUM_KIND) -> str:
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


def _curriculum_wandb_group(curriculum_name: str) -> str:
    """Return the W&B group used for all stages in one curriculum."""
    return f"curriculum/{curriculum_name}"


def _to_yaml(payload: Mapping[str, Any]) -> str:
    """Serialize a small config payload to YAML with the project dependency."""
    return yaml.safe_dump(dict(payload), sort_keys=False)


__all__ = [
    "DEFAULT_LLM_CURRICULUM_CONFIG_PATH",
    "LLMCurriculumResult",
    "LLMCurriculumSettings",
    "LLMCurriculumStage",
    "derive_stage_run_name",
    "llm_curriculum_settings_from_mapping",
    "load_llm_curriculum_settings",
    "run_llm_curriculum_training",
    "run_llm_curriculum_training_from_config",
    "validate_llm_curriculum",
]
