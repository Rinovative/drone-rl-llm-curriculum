"""
===============================================================================
experiments_ppo_tracking.py
===============================================================================
Run tiny Stable-Baselines3 PPO smoke training on TrajectoryTrackingEnv.

Responsibilities:
  - Load a configured validated trajectory task for PPO smoke training
  - Verify the Gymnasium wrapper with Stable-Baselines3 when available
  - Train, save, and evaluate a bounded deterministic PPO model headlessly
  - Write compact JSON metrics and action/liftoff diagnostics under storage

Design principles:
  - Keep defaults tiny, deterministic, bounded, and safe for Docker or HPC smoke runs
  - Import heavyweight RL dependencies lazily inside training functions
  - Keep generated models and metrics out of source-controlled paths

Boundaries:
  - Curriculum generation and LLM calls belong in llm modules
  - Long training runs, plotting, and trained-policy rendering belong elsewhere
===============================================================================

"""

from __future__ import annotations

import importlib.util
import json
import warnings as py_warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src import envs, experiments, utils

DEFAULT_PPO_TRACKING_CONFIG_PATH = Path("configs/smoke/ppo_hover_smoke.yaml")
DEFAULT_TASK_CONFIG_PATH = Path("configs/smoke/trajectory_validation.yaml")
DEFAULT_RUN_NAME = "ppo_hover_smoke"
DEFAULT_TASK_INDEX = 0
DEFAULT_TOTAL_TIMESTEPS = 4096
DEFAULT_EVAL_STEPS = 120
DEFAULT_SEED = 0
DEFAULT_MODEL_FILENAME = f"{DEFAULT_RUN_NAME}.zip"
DEFAULT_METRICS_FILENAME = f"{DEFAULT_RUN_NAME}_metrics.json"
DEFAULT_MANIFEST_FILENAME = f"{DEFAULT_RUN_NAME}_manifest.json"
_MIN_PPO_ROLLOUT_STEPS = 2
_MAX_PPO_ROLLOUT_STEPS = 64
DEFAULT_LIFTOFF_DIAGNOSTIC_STEPS = 120
_MOVEMENT_WARNING_SPAN_THRESHOLD_M = 0.05
_POSITION_BOUNDS_MAX_NDIM = 2


@dataclass(frozen=True)
class PPOTrackingSmokeSettings:
    """
    Settings for a tiny PPO trajectory-tracking smoke run.

    Parameters
    ----------
    task_config_path
        YAML config containing a top-level list of trajectory tasks.
    task_index
        Zero-based task index selected from the task config.
    task_shape
        Optional task-shape selector matched against the configured task list.
    run_name
        Optional explicit output directory for model, metrics, and W&B artifacts.
    total_timesteps
        Tiny upper-level PPO learning budget passed to Stable-Baselines3.
    eval_steps
        Number of deterministic evaluation steps to run after training.
    seed
        Deterministic seed used for environment resets and PPO initialization.
    output_dir
        Directory where the metrics JSON artifact is written.
    model_dir
        Directory where the trained PPO model zip is written.
    manifest_filename
        Manifest JSON filename within the training run manifests directory.
    model_filename
        Trained model filename within ``model_dir``.
    metrics_filename
        Metrics JSON filename within ``output_dir``.
    check_env
        Whether to run the Stable-Baselines3 environment checker before training.
    wandb_mode
        Optional W&B mode. Disabled by default for safe local and test execution.
    wandb_project
        W&B project name used when tracking is enabled.
    wandb_entity
        Optional W&B entity/team.
    wandb_group
        Optional W&B run group.
    wandb_name
        Optional W&B run name.
    wandb_tags
        Optional W&B run tags.
    wandb_dir
        Optional W&B output directory. Defaults to the run-specific wandb directory.

    """

    task_config_path: Path = DEFAULT_TASK_CONFIG_PATH
    task_index: int = DEFAULT_TASK_INDEX
    task_shape: str | None = None
    run_name: str | None = None
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    eval_steps: int = DEFAULT_EVAL_STEPS
    seed: int = DEFAULT_SEED
    output_dir: Path | None = None
    model_dir: Path | None = None
    manifest_filename: str = DEFAULT_MANIFEST_FILENAME
    model_filename: str = DEFAULT_MODEL_FILENAME
    metrics_filename: str = DEFAULT_METRICS_FILENAME
    check_env: bool = True
    wandb_mode: str = utils.wandb.WANDB_MODE_DISABLED
    wandb_project: str = utils.wandb.DEFAULT_WANDB_PROJECT
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate PPO smoke-run settings."""
        if self.task_index < 0:
            message = "task_index must be nonnegative"
            raise ValueError(message)
        if self.task_shape is not None and not self.task_shape.strip():
            message = "task_shape must be non-empty when provided"
            raise ValueError(message)
        if self.run_name is not None:
            utils.artifacts.get_training_run_dir(self.run_name)
        if self.total_timesteps <= 0:
            message = "total_timesteps must be positive"
            raise ValueError(message)
        if self.eval_steps <= 0:
            message = "eval_steps must be positive"
            raise ValueError(message)
        if not self.model_filename.endswith(".zip"):
            message = "model_filename must end with .zip"
            raise ValueError(message)
        if not self.metrics_filename.endswith(".json"):
            message = "metrics_filename must end with .json"
            raise ValueError(message)
        if self.wandb_mode not in utils.wandb.WANDB_MODES:
            message = f"wandb_mode must be one of: {', '.join(utils.wandb.WANDB_MODES)}"
            raise ValueError(message)
        if not self.wandb_project.strip():
            message = "wandb_project must be non-empty"
            raise ValueError(message)


@dataclass(frozen=True)
class PPOTrackingSmokeResult:
    """
    Summary returned by a PPO trajectory-tracking smoke run.

    Parameters
    ----------
    model_path
        Path to the saved Stable-Baselines3 PPO model zip.
    metrics_path
        Path to the written metrics JSON artifact.
    manifest_path
        Path to the written training manifest JSON artifact.
    metrics
        JSON-serializable metrics proving PPO trained and evaluated.
    warnings
        Nonfatal compatibility or checker warnings captured during the run.

    """

    model_path: str
    metrics_path: str
    manifest_path: str
    metrics: dict[str, Any]
    warnings: tuple[str, ...] = ()


def load_ppo_tracking_settings(path: str | Path) -> PPOTrackingSmokeSettings:
    """
    Load PPO trajectory-tracking smoke settings from YAML.

    Parameters
    ----------
    path
        Path to a YAML settings file.

    Returns
    -------
    PPOTrackingSmokeSettings
        Validated settings with paths expanded as ``Path`` objects.

    """
    config = experiments.config.load_experiment_config(path)
    return _settings_from_mapping(config)


def default_output_dir() -> Path:
    """Return the default PPO training run directory under the training layout."""
    return utils.artifacts.get_training_run_dir(DEFAULT_RUN_NAME)


def default_metrics_dir() -> Path:
    """Return the default PPO training metrics directory under the training layout."""
    return utils.artifacts.get_training_metrics_dir(DEFAULT_RUN_NAME)


def default_model_dir() -> Path:
    """Return the default PPO training model directory under the training layout."""
    return utils.artifacts.get_training_models_dir(DEFAULT_RUN_NAME)


def default_manifests_dir() -> Path:
    """Return the default PPO training manifest directory under the training layout."""
    return utils.artifacts.get_training_manifests_dir(DEFAULT_RUN_NAME)


def detect_ppo_tracking_dependencies() -> dict[str, bool]:
    """Return availability flags for PPO trajectory-tracking runtime dependencies."""
    return {
        "stable_baselines3": importlib.util.find_spec("stable_baselines3") is not None,
        "gymnasium": importlib.util.find_spec("gymnasium") is not None,
        "gym_pybullet_drones": importlib.util.find_spec("gym_pybullet_drones") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }


def detect_ppo_runtime_info() -> dict[str, Any]:
    """Return torch/CUDA runtime information without requiring GPU availability."""
    try:
        import torch  # noqa: PLC0415
    except ImportError as exc:
        return {
            "torch_available": False,
            "torch_cuda_available": False,
            "torch_cuda_device_count": 0,
            "torch_cuda_device_name": "",
            "torch_import_error": str(exc),
        }

    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count()) if cuda_available else 0
    device_name = str(torch.cuda.get_device_name(0)) if cuda_available and device_count > 0 else ""
    return {
        "torch_available": True,
        "torch_cuda_available": cuda_available,
        "torch_cuda_device_count": device_count,
        "torch_cuda_device_name": device_name,
    }


def describe_tracking_env_action_metadata(task: dict[str, Any]) -> dict[str, Any]:
    """
    Build TrajectoryTrackingEnv and return action-space metadata for diagnostics.

    Parameters
    ----------
    task
        Valid trajectory task mapping used to construct the tracking environment.

    Returns
    -------
    dict[str, Any]
        JSON-serializable action-space and upstream action-type metadata.

    """
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False)
    try:
        return _tracking_env_action_metadata(tracking_env)
    finally:
        tracking_env.close()


def run_ppo_tracking_smoke(settings: PPOTrackingSmokeSettings | None = None) -> PPOTrackingSmokeResult:
    """
    Train and evaluate a bounded Stable-Baselines3 PPO model on TrajectoryTrackingEnv.

    Parameters
    ----------
    settings
        Optional PPO smoke-run settings. Defaults are used when omitted.

    Returns
    -------
    PPOTrackingSmokeResult
        Saved model path, metrics path, metrics payload, and nonfatal warnings.

    Raises
    ------
    RuntimeError
        If required RL dependencies are unavailable or the environment checker fails.
    ValueError
        If the selected configured task is invalid.

    """
    active_settings = settings or PPOTrackingSmokeSettings()
    dependencies = detect_ppo_tracking_dependencies()
    runtime_info = detect_ppo_runtime_info()
    _require_training_dependencies(dependencies)
    task, task_source, selected_task_index, selection_warnings = _select_task(
        task_config_path=active_settings.task_config_path,
        default_task_index=active_settings.task_index,
        task_shape=active_settings.task_shape,
    )

    warnings = [*selection_warnings, *(_check_tracking_env(task) if active_settings.check_env else ())]
    diagnostic_steps = min(active_settings.eval_steps, DEFAULT_LIFTOFF_DIAGNOSTIC_STEPS)
    simple_liftoff_diagnostics = run_liftoff_diagnostics(
        task=task,
        max_steps=diagnostic_steps,
        seed=active_settings.seed,
    )
    model_path = _resolve_model_path(active_settings)
    metrics_path = _resolve_metrics_path(active_settings)
    manifest_path = _resolve_manifest_path(active_settings)
    wandb_settings = _wandb_settings(active_settings)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO  # noqa: PLC0415

    training_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False)
    try:
        action_metadata = _tracking_env_action_metadata(training_env)
        rollout_steps = _ppo_rollout_steps(active_settings.total_timesteps)
        model = PPO(
            "MlpPolicy",
            training_env,
            batch_size=rollout_steps,
            device="cpu",
            gamma=0.95,
            learning_rate=1.0e-3,
            n_epochs=4,
            n_steps=rollout_steps,
            seed=active_settings.seed,
            verbose=0,
        )
        wandb_run = utils.wandb.start_wandb_run(
            settings=wandb_settings,
            config=_wandb_config(active_settings, model_path, metrics_path, selected_task_index, task),
        )
        model.learn(total_timesteps=active_settings.total_timesteps, progress_bar=False)
        model.save(str(model_path))
        ppo_device = str(model.device)
        eval_metrics = _evaluate_model(model, training_env, active_settings)
    finally:
        training_env.close()

    trained_liftoff_diagnostics = run_liftoff_diagnostics(
        task=task,
        max_steps=diagnostic_steps,
        seed=active_settings.seed,
        model=model,
        include_simple_policies=False,
    )
    liftoff_diagnostics = {
        **simple_liftoff_diagnostics,
        **trained_liftoff_diagnostics,
    }
    warnings.extend(_movement_warnings(eval_metrics=eval_metrics, action_metadata=action_metadata))

    metrics: dict[str, Any] = {
        "run_type": "training",
        "mode": "ppo_smoke",
        "training_run_name": _run_name(active_settings),
        "run_name": _run_name(active_settings),
        "task_config_path": str(active_settings.task_config_path),
        "task_index": selected_task_index,
        "configured_task_index": active_settings.task_index,
        "task_source": task_source,
        "task_shape": str(task.get("shape", "unknown")),
        "training_task_shape": str(task.get("shape", "unknown")),
        "task_shape_requested": active_settings.task_shape,
        "total_timesteps": active_settings.total_timesteps,
        "eval_steps": active_settings.eval_steps,
        "seed": active_settings.seed,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "dependency_available": dependencies,
        "runtime": runtime_info,
        "ppo_device": ppo_device,
        "action_metadata": action_metadata,
        "liftoff_diagnostics": liftoff_diagnostics,
        "warnings": warnings,
        "trained": True,
        "env_checked": active_settings.check_env,
        "wandb": {
            "mode": active_settings.wandb_mode,
            "project": active_settings.wandb_project,
            "entity": active_settings.wandb_entity,
            "group": active_settings.wandb_group,
            "name": active_settings.wandb_name,
            "tags": list(active_settings.wandb_tags),
            "dir": str(wandb_settings.dir or utils.wandb.default_wandb_dir()),
            "enabled": active_settings.wandb_mode != utils.wandb.WANDB_MODE_DISABLED,
        },
        **eval_metrics,
    }
    utils.wandb.log_wandb_metrics(wandb_run, metrics)
    if wandb_run is not None:
        wandb_run.finish()
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = _build_manifest(active_settings, metrics, task_source=task_source, selected_task_index=selected_task_index, task=task)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return PPOTrackingSmokeResult(
        model_path=str(model_path),
        metrics_path=str(metrics_path),
        manifest_path=str(manifest_path),
        metrics=metrics,
        warnings=tuple(warnings),
    )


def run_ppo_tracking_smoke_from_config(
    config_path: str | Path = DEFAULT_PPO_TRACKING_CONFIG_PATH,
    task_index: int | None = None,
    task_shape: str | None = None,
    run_name: str | None = None,
    total_timesteps: int | None = None,
    eval_steps: int | None = None,
    output_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    seed: int | None = None,
    wandb_mode: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_name: str | None = None,
    wandb_tags: tuple[str, ...] | None = None,
    wandb_dir: str | Path | None = None,
) -> PPOTrackingSmokeResult:
    """
    Load settings, apply CLI-style overrides, and run PPO smoke training.

    Parameters
    ----------
    config_path
        YAML settings path.
    task_index
        Optional task-index override.
    task_shape
        Optional configured task-shape override.
    run_name
        Optional storage/training_runs/<run_name> root for generated artifacts.
    total_timesteps
        Optional PPO timestep-budget override.
    eval_steps
        Optional evaluation-step override.
    output_dir
        Optional metrics output directory override.
    model_dir
        Optional model output directory override.
    seed
        Optional deterministic seed override.
    wandb_mode
        Optional W&B mode override.
    wandb_project
        Optional W&B project override.
    wandb_entity
        Optional W&B entity override.
    wandb_group
        Optional W&B group override.
    wandb_name
        Optional W&B run-name override.
    wandb_tags
        Optional W&B tag override.
    wandb_dir
        Optional W&B directory override.

    Returns
    -------
    PPOTrackingSmokeResult
        Saved model path, metrics path, metrics payload, and nonfatal warnings.

    """
    settings = load_ppo_tracking_settings(config_path)
    overridden = PPOTrackingSmokeSettings(
        task_config_path=settings.task_config_path,
        task_index=settings.task_index if task_index is None else task_index,
        task_shape=settings.task_shape if task_shape is None else task_shape,
        run_name=settings.run_name if run_name is None else run_name,
        total_timesteps=settings.total_timesteps if total_timesteps is None else total_timesteps,
        eval_steps=settings.eval_steps if eval_steps is None else eval_steps,
        seed=settings.seed if seed is None else seed,
        output_dir=settings.output_dir if output_dir is None else Path(output_dir),
        model_dir=settings.model_dir if model_dir is None else Path(model_dir),
        manifest_filename=settings.manifest_filename,
        model_filename=settings.model_filename,
        metrics_filename=settings.metrics_filename,
        check_env=settings.check_env,
        wandb_mode=settings.wandb_mode if wandb_mode is None else wandb_mode,
        wandb_project=settings.wandb_project if wandb_project is None else wandb_project,
        wandb_entity=settings.wandb_entity if wandb_entity is None else wandb_entity,
        wandb_group=settings.wandb_group if wandb_group is None else wandb_group,
        wandb_name=settings.wandb_name if wandb_name is None else wandb_name,
        wandb_tags=settings.wandb_tags if wandb_tags is None else wandb_tags,
        wandb_dir=settings.wandb_dir if wandb_dir is None else Path(wandb_dir),
    )
    return run_ppo_tracking_smoke(overridden)


def _settings_from_mapping(config: dict[str, Any]) -> PPOTrackingSmokeSettings:
    """Build settings from a loaded YAML mapping."""
    output_dir_value = config.get("output_dir")
    model_dir_value = config.get("model_dir")
    wandb_dir_value = config.get("wandb_dir")
    settings_kwargs: dict[str, Any] = {
        "task_config_path": Path(config.get("task_config_path", DEFAULT_TASK_CONFIG_PATH)),
        "task_index": int(config.get("task_index", DEFAULT_TASK_INDEX)),
        "total_timesteps": int(config.get("total_timesteps", DEFAULT_TOTAL_TIMESTEPS)),
        "eval_steps": int(config.get("eval_steps", DEFAULT_EVAL_STEPS)),
        "seed": int(config.get("seed", DEFAULT_SEED)),
        "output_dir": Path(output_dir_value) if output_dir_value is not None else None,
        "model_dir": Path(model_dir_value) if model_dir_value is not None else None,
        "manifest_filename": str(config.get("manifest_filename", DEFAULT_MANIFEST_FILENAME)),
        "model_filename": str(config.get("model_filename", DEFAULT_MODEL_FILENAME)),
        "metrics_filename": str(config.get("metrics_filename", DEFAULT_METRICS_FILENAME)),
        "check_env": bool(config.get("check_env", True)),
        "wandb_mode": str(config.get("wandb_mode") or utils.wandb.WANDB_MODE_DISABLED),
        "wandb_project": str(config.get("wandb_project") or utils.wandb.DEFAULT_WANDB_PROJECT),
        "wandb_entity": config.get("wandb_entity") or None,
        "wandb_group": config.get("wandb_group") or None,
        "wandb_name": config.get("wandb_name") or None,
        "wandb_tags": utils.wandb.parse_wandb_tags(config.get("wandb_tags")),
    }
    if wandb_dir_value is not None:
        settings_kwargs["wandb_dir"] = Path(wandb_dir_value)
    return PPOTrackingSmokeSettings(**settings_kwargs)


def _load_task(task_config_path: Path, task_index: int) -> dict[str, Any]:
    """Load and return a copied task from a task config path."""
    task, _, _, _ = _select_task(task_config_path=task_config_path, default_task_index=task_index, task_shape=None)
    return task


def _select_task(
    task_config_path: Path,
    default_task_index: int,
    task_shape: str | None,
) -> tuple[dict[str, Any], str, int, tuple[str, ...]]:
    """Load one configured training task by index or shape."""
    config = experiments.config.load_experiment_config(task_config_path)
    tasks = config.get("tasks")
    if not isinstance(tasks, list):
        message = "task config must contain a top-level tasks list"
        raise ValueError(message)  # noqa: TRY004 - public config contract reports config errors as ValueError.
    if task_shape is None:
        if default_task_index < 0 or default_task_index >= len(tasks):
            message = "task_index is outside the configured task list"
            raise ValueError(message)
        task = tasks[default_task_index]
        if not isinstance(task, dict):
            message = "selected task must be a mapping"
            raise ValueError(message)
        return dict(task), "config", default_task_index, ()

    requested_shape = task_shape.strip().lower()
    if not requested_shape:
        message = "task_shape must be non-empty when provided"
        raise ValueError(message)

    for index, candidate in enumerate(tasks):
        if not isinstance(candidate, dict):
            continue
        if str(candidate.get("shape", "")).lower() == requested_shape:
            warning = "task_shape override selected a training task from config"
            return dict(candidate), "shape_override", index, (warning,)

    available_shapes = sorted({str(task.get("shape")) for task in tasks if isinstance(task, dict) and task.get("shape") is not None})
    message = f"task_shape '{task_shape}' not found in task config; available: {', '.join(available_shapes)}"
    raise ValueError(message)


def _run_name(settings: PPOTrackingSmokeSettings) -> str:
    """Return the configured training run name or the default smoke training run name."""
    return DEFAULT_RUN_NAME if settings.run_name is None else settings.run_name


def _resolve_model_path(settings: PPOTrackingSmokeSettings) -> Path:
    """Resolve the trained model output path."""
    default_dir = utils.artifacts.get_training_models_dir(_run_name(settings))
    filename = settings.model_filename if settings.model_filename != DEFAULT_MODEL_FILENAME else f"{_run_name(settings)}.zip"
    return _resolve_directory(settings.model_dir, default_dir) / filename


def _resolve_metrics_path(settings: PPOTrackingSmokeSettings) -> Path:
    """Resolve the metrics output path."""
    filename = settings.metrics_filename if settings.metrics_filename != DEFAULT_METRICS_FILENAME else f"{_run_name(settings)}_metrics.json"
    if settings.output_dir is None:
        return utils.artifacts.get_training_metrics_dir(_run_name(settings)) / filename
    output_dir = settings.output_dir.expanduser().resolve(strict=False)
    if "results" in output_dir.parts or output_dir.name == utils.artifacts.METRICS_DIRNAME:
        return output_dir / filename
    return output_dir / utils.artifacts.METRICS_DIRNAME / filename


def _resolve_manifest_path(settings: PPOTrackingSmokeSettings) -> Path:
    """Resolve the manifest output path."""
    filename = settings.manifest_filename if settings.manifest_filename != DEFAULT_MANIFEST_FILENAME else f"{_run_name(settings)}_manifest.json"
    if settings.output_dir is None:
        return utils.artifacts.get_training_manifests_dir(_run_name(settings)) / filename
    output_dir = settings.output_dir.expanduser().resolve(strict=False)
    if "results" in output_dir.parts or output_dir.name == utils.artifacts.MANIFESTS_DIRNAME:
        return output_dir / filename
    return output_dir / utils.artifacts.MANIFESTS_DIRNAME / filename


def _resolve_directory(path: Path | None, default: Path) -> Path:
    """Resolve a configured directory or its storage-backed default."""
    directory = default if path is None else path
    return directory.expanduser().resolve(strict=False)


def _wandb_settings(settings: PPOTrackingSmokeSettings) -> utils.wandb.WandbTrackingSettings:
    """Build W&B settings from PPO smoke settings."""
    return utils.wandb.WandbTrackingSettings(
        mode=settings.wandb_mode,
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        group=settings.wandb_group,
        name=settings.wandb_name,
        tags=settings.wandb_tags,
        dir=settings.wandb_dir or utils.artifacts.get_training_wandb_dir(_run_name(settings)),
    )


def _build_manifest(
    settings: PPOTrackingSmokeSettings,
    metrics: dict[str, Any],
    task_source: str,
    selected_task_index: int,
    task: dict[str, Any],
) -> dict[str, Any]:
    """Build a manifest payload for a PPO training run."""
    return {
        "run_type": "training",
        "mode": "ppo_smoke",
        "training_run_name": _run_name(settings),
        "run_name": _run_name(settings),
        "task_config_path": str(settings.task_config_path),
        "task_source": task_source,
        "task_index": selected_task_index,
        "task_shape": str(task.get("shape", "unknown")),
        "training_task_shape": str(task.get("shape", "unknown")),
        "task_shape_requested": settings.task_shape,
        "total_timesteps": settings.total_timesteps,
        "eval_steps": settings.eval_steps,
        "seed": settings.seed,
        "model_path": metrics["model_path"],
        "metrics_path": metrics["metrics_path"],
        "manifest_path": metrics["manifest_path"],
        "output_dir": str(settings.output_dir) if settings.output_dir is not None else str(default_output_dir()),
        "warnings": list(metrics.get("warnings", [])),
        "wandb": metrics.get("wandb", {}),
    }


def _wandb_config(
    settings: PPOTrackingSmokeSettings,
    model_path: Path,
    metrics_path: Path,
    selected_task_index: int,
    task: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact W&B config payload for PPO smoke training."""
    return {
        "run_name": _run_name(settings),
        "task_config_path": str(settings.task_config_path),
        "task_index": selected_task_index,
        "task_shape": str(task.get("shape", "unknown")),
        "task_shape_requested": settings.task_shape,
        "total_timesteps": settings.total_timesteps,
        "eval_steps": settings.eval_steps,
        "seed": settings.seed,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
    }


def _require_training_dependencies(dependencies: dict[str, bool]) -> None:
    """Raise if a dependency required for real PPO training is unavailable."""
    required = ("stable_baselines3", "gymnasium", "gym_pybullet_drones", "torch")
    missing = [name for name in required if not dependencies.get(name, False)]
    if missing:
        message = f"PPO smoke training requires missing dependencies: {', '.join(sorted(missing))}"
        raise RuntimeError(message)


def _check_tracking_env(task: dict[str, Any]) -> tuple[str, ...]:
    """Run Stable-Baselines3's environment checker and return captured warnings."""
    try:
        from stable_baselines3.common.env_checker import check_env  # noqa: PLC0415
    except ImportError as exc:
        return (f"stable_baselines3 env checker unavailable: {exc}",)

    checker_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False)
    try:
        with py_warnings.catch_warnings(record=True) as records:
            py_warnings.simplefilter("always")
            check_env(checker_env, warn=True)
        return tuple(str(record.message) for record in records)
    except Exception as exc:
        message = f"TrajectoryTrackingEnv failed Stable-Baselines3 check_env: {exc}"
        raise RuntimeError(message) from exc
    finally:
        checker_env.close()


def _ppo_rollout_steps(total_timesteps: int) -> int:
    """Choose a tiny PPO rollout length compatible with PPO batch sizing."""
    return max(_MIN_PPO_ROLLOUT_STEPS, min(_MAX_PPO_ROLLOUT_STEPS, total_timesteps))


def _evaluate_model(model: Any, tracking_env: Any, settings: PPOTrackingSmokeSettings) -> dict[str, Any]:
    """Evaluate a trained model for a deterministic rollout with movement bounds."""
    observation, _ = tracking_env.reset(seed=settings.seed)
    rewards: list[float] = []
    errors: list[float] = []
    positions: list[np.ndarray] = []
    reference_positions: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    terminated_count = 0
    truncated_count = 0
    reset_count = 0
    for _ in range(settings.eval_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = tracking_env.step(action)
        rewards.append(float(reward))
        errors.append(float(info["position_error_m"]))
        positions.append(np.asarray(info["current_position"], dtype=float))
        reference_positions.append(np.asarray(info["reference_position"], dtype=float))
        actions.append(np.asarray(action, dtype=float))
        if terminated:
            terminated_count += 1
        if truncated:
            truncated_count += 1
        if terminated or truncated:
            reset_count += 1
            observation, _ = tracking_env.reset(seed=settings.seed + reset_count)

    position_bounds = _position_bounds(positions)
    reference_position_bounds = _position_bounds(reference_positions)
    action_bounds = _position_bounds(actions)
    return {
        "actual_eval_steps": len(rewards),
        "eval_resets": reset_count,
        "eval_terminated_count": terminated_count,
        "eval_truncated_count": truncated_count,
        "mean_eval_reward": float(np.mean(rewards)),
        "final_eval_reward": float(rewards[-1]),
        "mean_position_error_m": float(np.mean(errors)),
        "final_position_error_m": float(errors[-1]),
        "position_bounds": position_bounds,
        "reference_position_bounds": reference_position_bounds,
        "action_bounds": action_bounds,
        "actual_z_span_m": _axis_span(position_bounds, axis=2),
        "actual_xy_span_m": float(np.linalg.norm([_axis_span(position_bounds, axis=0), _axis_span(position_bounds, axis=1)])),
        "reference_z_span_m": _axis_span(reference_position_bounds, axis=2),
        "reference_xy_span_m": float(np.linalg.norm([_axis_span(reference_position_bounds, axis=0), _axis_span(reference_position_bounds, axis=1)])),
    }


def run_liftoff_diagnostics(
    task: dict[str, Any],
    max_steps: int = DEFAULT_LIFTOFF_DIAGNOSTIC_STEPS,
    seed: int = DEFAULT_SEED,
    model: Any | None = None,
    include_simple_policies: bool = True,
) -> dict[str, Any]:
    """
    Run short headless rollouts that reveal whether valid actions can lift the drone.

    Parameters
    ----------
    task
        Valid trajectory task mapping used to construct TrajectoryTrackingEnv.
    max_steps
        Maximum steps for each diagnostic rollout.
    seed
        Deterministic reset and action-space seed.
    model
        Optional trained PPO-like model exposing ``predict`` for a policy rollout.
    include_simple_policies
        Whether to include zero, sampled, middle, and high action probes.

    Returns
    -------
    dict[str, Any]
        JSON-serializable rollout summaries keyed by policy name.

    """
    diagnostic_task, reference_samples, preparation_warnings = _task_with_minimum_reference_samples(task, required_steps=max_steps)
    diagnostics: dict[str, Any] = {
        "task_preparation": {
            "reference_samples": reference_samples,
            "warnings": list(preparation_warnings),
        }
    }
    if include_simple_policies:
        diagnostics["zero_action"] = _run_liftoff_rollout(diagnostic_task, "zero_action", _zero_action, max_steps=max_steps, seed=seed)
        diagnostics["sampled_action"] = _run_liftoff_rollout(diagnostic_task, "sampled_action", _sampled_action, max_steps=max_steps, seed=seed)
        diagnostics["middle_action"] = _run_liftoff_rollout(diagnostic_task, "middle_action", _middle_action, max_steps=max_steps, seed=seed)
        diagnostics["high_action"] = _run_liftoff_rollout(diagnostic_task, "high_action", _high_action, max_steps=max_steps, seed=seed)
    if model is not None:
        diagnostics["trained_policy"] = _run_liftoff_rollout(
            diagnostic_task,
            "trained_policy",
            lambda _env, observation, _step: model.predict(observation, deterministic=True)[0],
            max_steps=max_steps,
            seed=seed,
        )
    return diagnostics


def _task_with_minimum_reference_samples(task: dict[str, Any], required_steps: int) -> tuple[dict[str, Any], int, tuple[str, ...]]:
    """Return a copied task with enough reference samples for diagnostics."""
    reference = envs.task_adapter.make_task_reference(task)
    reference_samples = int(reference.positions.shape[0])
    required_samples = required_steps + 1
    if reference_samples >= required_samples:
        return dict(task), reference_samples, ()

    duration_value = task.get("duration_sec")
    sample_rate_value = task.get("sample_rate_hz")
    if duration_value is None or sample_rate_value is None:
        warning = "diagnostic task has too few reference samples and cannot be extended because duration_sec/sample_rate_hz are missing"
        return dict(task), reference_samples, (warning,)

    duration_sec = float(duration_value)
    sample_rate_hz = float(sample_rate_value)
    if duration_sec <= 0.0 or sample_rate_hz <= 0.0:
        warning = "diagnostic task has non-positive duration_sec or sample_rate_hz and cannot be safely extended"
        return dict(task), reference_samples, (warning,)

    required_sample_rate_hz = int(np.ceil(required_samples / duration_sec))
    extended_task = dict(task)
    extended_task["sample_rate_hz"] = float(required_sample_rate_hz)
    extended_reference = envs.task_adapter.make_task_reference(extended_task)
    extended_samples = int(extended_reference.positions.shape[0])
    warning = f"extended diagnostic task sample_rate_hz from {sample_rate_hz} to {required_sample_rate_hz} for {required_steps} steps"
    return extended_task, extended_samples, (warning,)


def _run_liftoff_rollout(
    task: dict[str, Any],
    name: str,
    action_factory: Any,
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Run one diagnostic rollout and return bounds and termination metadata."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False, max_steps=max_steps)
    try:
        seed_action_space = getattr(tracking_env.action_space, "seed", None)
        if callable(seed_action_space):
            seed_action_space(seed)
        observation, _ = tracking_env.reset(seed=seed)
        positions: list[np.ndarray] = []
        references: list[np.ndarray] = []
        rewards: list[float] = []
        errors: list[float] = []
        final_action: np.ndarray | None = None
        final_info: dict[str, Any] = {}
        terminated = False
        truncated = False
        for step_index in range(max_steps):
            action = action_factory(tracking_env, observation, step_index)
            final_action = np.asarray(action, dtype=float)
            observation, reward, terminated, truncated, info = tracking_env.step(action)
            final_info = dict(info)
            positions.append(np.asarray(info["current_position"], dtype=float))
            references.append(np.asarray(info["reference_position"], dtype=float))
            rewards.append(float(reward))
            errors.append(float(info["position_error_m"]))
            if terminated or truncated:
                break

        position_bounds = _position_bounds(positions)
        reference_position_bounds = _position_bounds(references)
        steps = len(positions)
        fallback_reason = _rollout_termination_reason(
            terminated=terminated,
            truncated=truncated,
            actual_steps=steps,
            requested_max_steps=max_steps,
            reference_sample_count=int(tracking_env.reference.positions.shape[0]),
        )
        return {
            "name": name,
            "steps": steps,
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "base_terminated": bool(final_info.get("base_terminated", False)),
            "base_truncated": bool(final_info.get("base_truncated", False)),
            "termination_reason": str(final_info.get("termination_reason", fallback_reason)),
            "base_info_keys": list(final_info.get("base_info_keys", [])),
            "base_reason_fields": dict(final_info.get("base_reason_fields", {})),
            "base_truncation_causes": list(final_info.get("base_truncation_causes", [])),
            "z_min": _axis_min(position_bounds, axis=2),
            "z_max": _axis_max(position_bounds, axis=2),
            "x_min": _axis_min(position_bounds, axis=0),
            "x_max": _axis_max(position_bounds, axis=0),
            "y_min": _axis_min(position_bounds, axis=1),
            "y_max": _axis_max(position_bounds, axis=1),
            "position_bounds": position_bounds,
            "reference_position_bounds": reference_position_bounds,
            "final_position": _array_to_jsonable(final_info.get("current_position", [])),
            "final_reference_position": _array_to_jsonable(final_info.get("reference_position", [])),
            "final_attitude_rpy": _array_to_jsonable(final_info.get("roll_pitch_yaw", [])),
            "final_velocity": _array_to_jsonable(final_info.get("velocity", [])),
            "final_angular_velocity": _array_to_jsonable(final_info.get("angular_velocity", [])),
            "final_action": _array_to_jsonable(final_action if final_action is not None else []),
            "final_last_action": _array_to_jsonable(final_info.get("last_action", [])),
            "mean_reward": float(np.mean(rewards)) if rewards else 0.0,
            "mean_position_error_m": float(np.mean(errors)) if errors else 0.0,
            "base_action_shape": list(final_info.get("base_action_shape", [])),
            "base_action_dtype": str(final_info.get("base_action_dtype", "")),
        }
    finally:
        tracking_env.close()


def _zero_action(tracking_env: Any, _observation: np.ndarray, _step_index: int) -> np.ndarray:
    """Return the zero action for a tracking environment."""
    return np.zeros(tracking_env.action_space.shape, dtype=tracking_env.action_space.dtype)


def _sampled_action(tracking_env: Any, _observation: np.ndarray, _step_index: int) -> np.ndarray:
    """Return one deterministic action-space sample after the rollout seed is applied."""
    return tracking_env.action_space.sample()


def _middle_action(tracking_env: Any, _observation: np.ndarray, _step_index: int) -> np.ndarray:
    """Return the midpoint action for finite bounded Box spaces."""
    low = np.asarray(tracking_env.action_space.low, dtype=float)
    high = np.asarray(tracking_env.action_space.high, dtype=float)
    return ((low + high) / 2.0).astype(tracking_env.action_space.dtype)


def _high_action(tracking_env: Any, _observation: np.ndarray, _step_index: int) -> np.ndarray:
    """Return the high action for finite bounded Box spaces."""
    return np.asarray(tracking_env.action_space.high, dtype=tracking_env.action_space.dtype)


def _tracking_env_action_metadata(tracking_env: Any) -> dict[str, Any]:
    """Return JSON-serializable action-space metadata for a tracking environment."""
    action_space = tracking_env.action_space
    sample = action_space.sample()
    action_type = getattr(getattr(tracking_env, "base_env", None), "ACT_TYPE", None)
    action_type_value = _enum_value(action_type)
    return {
        "action_space": str(action_space),
        "action_space_shape": _shape_list(getattr(action_space, "shape", ())),
        "action_space_dtype": str(getattr(action_space, "dtype", "")),
        "action_space_low": _array_to_jsonable(getattr(action_space, "low", [])),
        "action_space_high": _array_to_jsonable(getattr(action_space, "high", [])),
        "sampled_action_shape": _shape_list(np.asarray(sample).shape),
        "sampled_action_dtype": str(np.asarray(sample).dtype),
        "sampled_action": _array_to_jsonable(sample),
        "base_action_type": action_type_value,
        "base_action_semantics": _action_semantics(action_type_value),
    }


def _movement_warnings(eval_metrics: dict[str, Any], action_metadata: dict[str, Any]) -> list[str]:
    """Return nonfatal warnings when metrics reveal an unsuitable movement setup."""
    warnings: list[str] = []
    if (
        action_metadata.get("base_action_type") == "one_d_rpm"
        and float(eval_metrics.get("reference_xy_span_m", 0.0)) > _MOVEMENT_WARNING_SPAN_THRESHOLD_M
    ):
        warnings.append("ONE_D_RPM exposes collective thrust only; horizontal reference motion cannot be tracked with this action interface")
    if (
        float(eval_metrics.get("reference_z_span_m", 0.0)) > _MOVEMENT_WARNING_SPAN_THRESHOLD_M
        and float(eval_metrics.get("actual_z_span_m", 0.0)) < _MOVEMENT_WARNING_SPAN_THRESHOLD_M
    ):
        warnings.append("trained policy produced little vertical movement during evaluation")
    return warnings


def _action_semantics(action_type_value: str) -> str:
    """Describe the upstream HoverAviary action semantics relevant to smoke training."""
    if action_type_value == "one_d_rpm":
        return (
            "one collective normalized thrust command shaped (num_drones, 1); "
            "expanded to equal motor RPMs, so it can change altitude but cannot command x/y motion"
        )
    if action_type_value == "pid":
        return (
            "three-dimensional normalized target-position command shaped (num_drones, 3); "
            "upstream PID converts targets into motor RPMs for x/y/z movement"
        )
    if action_type_value == "one_d_pid":
        return (
            "one-dimensional vertical target-position increment shaped (num_drones, 1); "
            "upstream PID converts it into motor RPMs for altitude movement"
        )
    if action_type_value == "vel":
        return "four-dimensional velocity command shaped (num_drones, 4); first three values define direction and the fourth scales target speed"
    return "upstream gym-pybullet-drones action semantics"


def _enum_value(value: Any) -> str:
    """Return an enum value or string representation for JSON metrics."""
    enum_value = getattr(value, "value", None)
    if enum_value is not None:
        return str(enum_value)
    return str(value)


def _array_to_jsonable(value: Any) -> list[Any]:
    """Convert an array-like value to nested JSON-compatible lists."""
    return np.asarray(value).tolist()


def _shape_list(shape: Any) -> list[int]:
    """Convert a shape-like object to a list of ints."""
    return [int(dimension) for dimension in tuple(shape)]


def _position_bounds(positions: list[np.ndarray]) -> dict[str, list[float]]:
    """Return min/max bounds for a list of position-like arrays."""
    if not positions:
        return {"min": [], "max": []}
    array = np.asarray(positions, dtype=float)
    if array.ndim > _POSITION_BOUNDS_MAX_NDIM:
        array = array.reshape(array.shape[0], -1)
    return {
        "min": [float(value) for value in np.min(array, axis=0)],
        "max": [float(value) for value in np.max(array, axis=0)],
    }


def _axis_min(bounds: dict[str, list[float]], axis: int) -> float:
    """Return one min-bound axis value or 0.0 when unavailable."""
    values = bounds.get("min", [])
    if len(values) <= axis:
        return 0.0
    return float(values[axis])


def _axis_max(bounds: dict[str, list[float]], axis: int) -> float:
    """Return one max-bound axis value or 0.0 when unavailable."""
    values = bounds.get("max", [])
    if len(values) <= axis:
        return 0.0
    return float(values[axis])


def _axis_span(bounds: dict[str, list[float]], axis: int) -> float:
    """Return the span of one bounded axis."""
    return float(_axis_max(bounds, axis=axis) - _axis_min(bounds, axis=axis))


def _rollout_termination_reason(
    terminated: bool,
    truncated: bool,
    actual_steps: int,
    requested_max_steps: int,
    reference_sample_count: int,
) -> str:
    """Explain why a diagnostic rollout ended."""
    if truncated:
        return "truncated"
    if terminated and actual_steps >= reference_sample_count:
        return "terminated_reference_complete"
    if terminated:
        return "terminated"
    if actual_steps >= requested_max_steps:
        return "requested_max_steps_exhausted_without_terminal"
    return "rollout_loop_ended_early"


__all__ = [
    "DEFAULT_LIFTOFF_DIAGNOSTIC_STEPS",
    "DEFAULT_PPO_TRACKING_CONFIG_PATH",
    "DEFAULT_RUN_NAME",
    "DEFAULT_TASK_CONFIG_PATH",
    "PPOTrackingSmokeResult",
    "PPOTrackingSmokeSettings",
    "default_metrics_dir",
    "default_model_dir",
    "default_output_dir",
    "describe_tracking_env_action_metadata",
    "detect_ppo_runtime_info",
    "detect_ppo_tracking_dependencies",
    "load_ppo_tracking_settings",
    "run_liftoff_diagnostics",
    "run_ppo_tracking_smoke",
    "run_ppo_tracking_smoke_from_config",
]
