"""
===============================================================================
experiments_training_ppo_tracking.py
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

import copy
import importlib.util
import json
import sys
import warnings as py_warnings
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import numpy as np

from src import envs, evaluation, utils, validation
from src.experiments import experiments_config as config_loader

from . import experiments_training_ppo_config as ppo_config

DEFAULT_PPO_TRACKING_CONFIG_PATH = Path("configs/training/ppo_tracking_smoke.yaml")
DEFAULT_TASK_CONFIG_PATH = Path("configs/training/ppo_tracking_tasks.yaml")
DEFAULT_TASK_INDEX = 0
DEFAULT_TOTAL_TIMESTEPS = 4096
DEFAULT_NUM_ENVS = 1
DEFAULT_EVAL_STEPS = 120
DEFAULT_SEED = 0
DEFAULT_LIFTOFF_DIAGNOSTIC_STEPS = 120
DEFAULT_NORMALIZE_ACTIONS = True
DEFAULT_ACTION_INTERFACE = envs.actions.DEFAULT_ACTION_INTERFACE.value
DEFAULT_INCLUDE_DYNAMICS_OBSERVATION = envs.actions.DEFAULT_INCLUDE_DYNAMICS_OBSERVATION
DEFAULT_INCLUDE_PREVIOUS_ACTION = envs.actions.DEFAULT_INCLUDE_PREVIOUS_ACTION
_MOVEMENT_WARNING_SPAN_THRESHOLD_M = 0.05
_POSITION_BOUNDS_MAX_NDIM = 2
_XY_TRACKING_RATIO_MIN_REFERENCE_SPAN_M = 1.0e-9
_ACTION_SATURATION_TOLERANCE = 1.0e-6
_TIMESTEPS_PER_THOUSAND_LABEL = 1_000
_TIMESTEPS_PER_MILLION_LABEL = 1_000_000
_MIN_TIMESTEPS_FOR_COMPACT_THOUSAND_LABEL = 10_000
_ENT005_COEFFICIENT = 0.005
_LOW_LR_LEARNING_RATE = 0.0001
VEC_MONITOR_ENABLED = True


@dataclass(frozen=True)
class PPOTrackingSmokeSettings:
    """
    Settings for a tiny PPO trajectory-tracking smoke run.

    Parameters
    ----------
    training_config_path
        Optional YAML training settings path used for reproducibility metadata.
    task_config_path
        YAML config containing a top-level list of trajectory tasks.
    task_index
        Zero-based task index selected from the task config.
    task_shape
        Optional task-shape selector matched against the configured task list.
    task_distribution_config_path
        Optional YAML config describing a fixed or randomized task distribution.
    task_distribution_settings
        Optional preloaded task-distribution settings used by curriculum callers.
    run_name
        Optional canonical storage/runs run name for model, metrics, and W&B artifacts.
    total_timesteps
        Tiny upper-level PPO learning budget passed to Stable-Baselines3.
    num_envs
        Number of parallel training environments used for PPO rollout collection.
    ppo_config
        Resolved PPO hyperparameters passed to Stable-Baselines3.
    eval_steps
        Number of deterministic evaluation steps to run after training.
    seed
        Deterministic seed used for environment resets and PPO initialization.
    output_dir
        Directory where the metrics JSON artifact is written.
    artifact_root
        Optional canonical training artifact root override used for curriculum stage training.
    model_dir
        Directory where the trained PPO model zip is written.
    manifest_filename
        Optional explicit manifest JSON filename within the canonical training directory.
    model_filename
        Optional explicit trained model filename within ``model_dir``.
    metrics_filename
        Optional explicit metrics JSON filename within ``output_dir``.
    check_env
        Whether to run the Stable-Baselines3 environment checker before training.
    normalize_actions
        Whether PID-position PPO should see a symmetric normalized action space mapped to real PID bounds.
    action_interface
        Explicit action interface, either ``pid_position`` or ``direct_rpm``.
    rpm_delta_scale
        Fractional RPM delta around hover used by ``direct_rpm``.
    include_dynamics_observation
        Whether tracking observations append velocity, attitude, and angular velocity.
    include_previous_action
        Whether tracking observations append the previous PPO-facing action.
    termination_limits
        Hard safety limits used by PPO training environments.
    diagnostic_limits
        Strict diagnostic thresholds reported during training and evaluation rollouts.
    wandb_mode
        Optional W&B mode. Auto mode uses online credentials when available and offline otherwise.
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
    initial_model_path
        Optional Stable-Baselines3 PPO model zip used to initialize this run.
    run_metadata
        Optional caller-owned metadata copied into metrics, manifests, and W&B config.

    """

    training_config_path: Path | None = None
    task_config_path: Path = DEFAULT_TASK_CONFIG_PATH
    task_index: int = DEFAULT_TASK_INDEX
    task_shape: str | None = None
    task_distribution_config_path: Path | None = None
    task_distribution_settings: envs.task_distribution.TaskDistributionSettings | None = None
    run_name: str | None = None
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    num_envs: int = DEFAULT_NUM_ENVS
    ppo_config: ppo_config.PPOConfig = field(default_factory=ppo_config.PPOConfig)
    eval_steps: int = DEFAULT_EVAL_STEPS
    seed: int = DEFAULT_SEED
    output_dir: Path | None = None
    artifact_root: Path | None = None
    model_dir: Path | None = None
    manifest_filename: str | None = None
    model_filename: str | None = None
    metrics_filename: str | None = None
    check_env: bool = True
    normalize_actions: bool = DEFAULT_NORMALIZE_ACTIONS
    action_interface: str = DEFAULT_ACTION_INTERFACE
    rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE
    include_dynamics_observation: bool = DEFAULT_INCLUDE_DYNAMICS_OBSERVATION
    include_previous_action: bool = DEFAULT_INCLUDE_PREVIOUS_ACTION
    termination_limits: envs.termination.TerminationLimitConfig | Mapping[str, Any] | str | None = None
    diagnostic_limits: envs.termination.DiagnosticLimitConfig | Mapping[str, Any] | str | None = None
    wandb_mode: str = utils.wandb.WANDB_MODE_AUTO
    wandb_project: str = utils.wandb.DEFAULT_WANDB_PROJECT
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_name: str | None = None
    wandb_tags: tuple[str, ...] = ()
    wandb_dir: Path | None = None
    initial_model_path: Path | None = None
    run_metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate PPO smoke-run settings."""
        if self.task_index < 0:
            message = "task_index must be nonnegative"
            raise ValueError(message)
        if self.task_shape is not None and not self.task_shape.strip():
            message = "task_shape must be non-empty when provided"
            raise ValueError(message)
        if self.task_distribution_settings is not None and not isinstance(
            self.task_distribution_settings,
            envs.task_distribution.TaskDistributionSettings,
        ):
            message = "task_distribution_settings must be a TaskDistributionSettings instance"
            raise TypeError(message)
        if not isinstance(self.run_metadata, dict):
            message = "run_metadata must be a dictionary"
            raise TypeError(message)
        if self.run_name is not None:
            utils.artifacts.get_run_dir(self.run_name)
        if self.total_timesteps <= 0:
            message = "total_timesteps must be positive"
            raise ValueError(message)
        object.__setattr__(self, "num_envs", _positive_int_setting(self.num_envs, "num_envs"))
        if not isinstance(self.ppo_config, ppo_config.PPOConfig):
            message = "ppo_config must be a PPOConfig"
            raise TypeError(message)
        self.ppo_config.validate_total_timesteps(self.total_timesteps)
        self.ppo_config.validate_rollout_consistency(self.num_envs)
        if self.eval_steps <= 0:
            message = "eval_steps must be positive"
            raise ValueError(message)
        if self.model_filename is not None and not self.model_filename.endswith(".zip"):
            message = "model_filename must end with .zip"
            raise ValueError(message)
        if self.metrics_filename is not None and not self.metrics_filename.endswith(".json"):
            message = "metrics_filename must end with .json"
            raise ValueError(message)
        if self.manifest_filename is not None and not self.manifest_filename.endswith(".json"):
            message = "manifest_filename must end with .json"
            raise ValueError(message)
        if not isinstance(self.normalize_actions, bool):
            message = "normalize_actions must be a boolean"
            raise TypeError(message)
        action_config = envs.actions.ActionInterfaceConfig(
            action_interface=self.action_interface,
            rpm_delta_scale=self.rpm_delta_scale,
            include_dynamics_observation=self.include_dynamics_observation,
            include_previous_action=self.include_previous_action,
        )
        object.__setattr__(self, "action_interface", action_config.parsed_action_interface.value)
        object.__setattr__(self, "rpm_delta_scale", action_config.rpm_delta_scale)
        object.__setattr__(self, "include_dynamics_observation", action_config.include_dynamics_observation)
        object.__setattr__(self, "include_previous_action", action_config.include_previous_action)
        object.__setattr__(
            self,
            "termination_limits",
            envs.termination.parse_termination_limits(self.termination_limits, action_config.parsed_action_interface),
        )
        object.__setattr__(self, "diagnostic_limits", envs.termination.parse_diagnostic_limits(self.diagnostic_limits))
        if action_config.parsed_action_interface == envs.actions.ActionInterface.DIRECT_RPM and not self.normalize_actions:
            message = "direct_rpm requires normalize_actions true because PPO actions are normalized motor commands"
            raise ValueError(message)
        if self.wandb_mode not in utils.wandb.WANDB_MODES:
            message = f"wandb_mode must be one of: {', '.join(utils.wandb.WANDB_MODES)}"
            raise ValueError(message)
        if not self.wandb_project.strip():
            message = "wandb_project must be non-empty"
            raise ValueError(message)
        if self.initial_model_path is not None and self.initial_model_path.suffix != ".zip":
            message = "initial_model_path must point to a .zip model"
            raise ValueError(message)


@dataclass(frozen=True)
class PPOTrackingSmokeResult:
    """
    Summary returned by a PPO trajectory-tracking smoke run.

    Parameters
    ----------
    model_path
        Backward-compatible selected model path. This is the last saved model when no best checkpoint was selected.
    metrics_path
        Path to the written metrics JSON artifact.
    manifest_path
        Path to the written training manifest JSON artifact.
    metrics
        JSON-serializable metrics proving PPO trained and evaluated.
    last_model_path
        Path to the last saved Stable-Baselines3 PPO model zip.
    best_model_path
        Path to a selected best model when a best-model selection mechanism produced one.
    best_model_metric
        Metric used for best-model selection, when available.
    best_model_step
        Training step associated with the best model, when available.
    best_model_source
        Human-readable source of best-model metadata.
    warnings
        Nonfatal compatibility or checker warnings captured during the run.

    """

    model_path: str
    metrics_path: str
    manifest_path: str
    metrics: dict[str, Any]
    last_model_path: str | None = None
    best_model_path: str | None = None
    best_model_metric: str | None = None
    best_model_step: int | None = None
    best_model_source: str | None = None
    warnings: tuple[str, ...] = ()


def _settings_termination_limits(settings: PPOTrackingSmokeSettings) -> envs.termination.TerminationLimitConfig:
    """Return the resolved termination limit config stored on validated settings."""
    return cast("envs.termination.TerminationLimitConfig", settings.termination_limits)


def _settings_diagnostic_limits(settings: PPOTrackingSmokeSettings) -> envs.termination.DiagnosticLimitConfig:
    """Return the resolved diagnostic limit config stored on validated settings."""
    return cast("envs.termination.DiagnosticLimitConfig", settings.diagnostic_limits)


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
    config_path = Path(path)
    config = config_loader.load_experiment_config(config_path)
    return _settings_from_mapping(config, training_config_path=config_path)


def default_output_dir() -> Path:
    """Return the derived default-config canonical run directory."""
    return utils.artifacts.get_run_dir(_default_training_run_name())


def default_metrics_dir() -> Path:
    """Return the derived default-config PPO training metrics directory."""
    return utils.artifacts.get_run_training_metrics_dir(_default_training_run_name())


def default_model_dir() -> Path:
    """Return the derived default-config PPO training model directory."""
    return utils.artifacts.get_run_training_models_dir(_default_training_run_name())


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


def describe_tracking_env_action_metadata(
    task: dict[str, Any],
    normalize_actions: bool = DEFAULT_NORMALIZE_ACTIONS,
    action_interface: str = DEFAULT_ACTION_INTERFACE,
    rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE,
    include_dynamics_observation: bool = DEFAULT_INCLUDE_DYNAMICS_OBSERVATION,
    include_previous_action: bool = DEFAULT_INCLUDE_PREVIOUS_ACTION,
    termination_limits: envs.termination.TerminationLimitConfig | Mapping[str, Any] | str | None = None,
    diagnostic_limits: envs.termination.DiagnosticLimitConfig | Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """
    Build TrajectoryTrackingEnv and return action-space metadata for diagnostics.

    Parameters
    ----------
    task
        Valid trajectory task mapping used to construct the tracking environment.
    normalize_actions
        Whether to describe the PPO-facing normalized wrapper instead of the real environment.
    action_interface
        Explicit action interface, either ``pid_position`` or ``direct_rpm``.
    rpm_delta_scale
        Fractional RPM delta around hover used by ``direct_rpm``.
    include_dynamics_observation
        Whether observations append velocity, attitude, and angular velocity.
    include_previous_action
        Whether observations append the previous PPO-facing action.
    termination_limits
        Optional hard episode-control safety limits used by the described environment.
    diagnostic_limits
        Optional strict diagnostic thresholds used by the described environment.

    Returns
    -------
    dict[str, Any]
        JSON-serializable action-space and upstream action-type metadata.

    """
    tracking_env = _make_seeded_ppo_tracking_env(
        task=task,
        normalize_actions=normalize_actions,
        seed=DEFAULT_SEED,
        action_interface=action_interface,
        rpm_delta_scale=rpm_delta_scale,
        include_dynamics_observation=include_dynamics_observation,
        include_previous_action=include_previous_action,
        termination_limits=termination_limits,
        diagnostic_limits=diagnostic_limits,
    )
    try:
        return _tracking_env_action_metadata(tracking_env)
    finally:
        tracking_env.close()


def _ppo_training_env(tracking_env: Any, normalize_actions: bool, action_interface: str = DEFAULT_ACTION_INTERFACE) -> Any:
    """Return the environment interface PPO should train against."""
    parsed_interface = envs.actions.parse_action_interface(action_interface)
    if parsed_interface == envs.actions.ActionInterface.DIRECT_RPM:
        return tracking_env
    if not normalize_actions:
        return tracking_env
    return envs.tracking_env.make_normalized_action_env(tracking_env)


def _make_seeded_ppo_tracking_env(
    task: dict[str, Any],
    normalize_actions: bool,
    seed: int,
    action_interface: str = DEFAULT_ACTION_INTERFACE,
    rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE,
    include_dynamics_observation: bool = DEFAULT_INCLUDE_DYNAMICS_OBSERVATION,
    include_previous_action: bool = DEFAULT_INCLUDE_PREVIOUS_ACTION,
    termination_limits: envs.termination.TerminationLimitConfig | Mapping[str, Any] | str | None = None,
    diagnostic_limits: envs.termination.DiagnosticLimitConfig | Mapping[str, Any] | str | None = None,
    task_distribution_settings: envs.task_distribution.TaskDistributionSettings | None = None,
    env_rank: int = 0,
) -> Any:
    """Build one PPO-facing tracking environment and apply a deterministic seed."""
    env_task: dict[str, Any] | envs.task_distribution.TaskDistributionSampler
    if task_distribution_settings is None:
        env_task = dict(task)
    else:
        env_task = envs.task_distribution.TaskDistributionSampler(task_distribution_settings, env_rank=env_rank)
    real_env = envs.tracking_env.make_trajectory_tracking_env(
        env_task,
        gui=False,
        record=False,
        action_interface=action_interface,
        rpm_delta_scale=rpm_delta_scale,
        include_dynamics_observation=include_dynamics_observation,
        include_previous_action=include_previous_action,
        termination_limits=termination_limits,
        diagnostic_limits=diagnostic_limits,
    )
    tracking_env = _ppo_training_env(real_env, normalize_actions=normalize_actions, action_interface=action_interface)
    _seed_tracking_env(tracking_env, seed)
    return tracking_env


def _seed_tracking_env(tracking_env: Any, seed: int) -> None:
    """Seed a Gymnasium tracking environment and its spaces when supported."""
    for space_name in ("action_space", "observation_space"):
        space = getattr(tracking_env, space_name, None)
        seed_space = getattr(space, "seed", None)
        if callable(seed_space):
            seed_space(seed)
    reset = getattr(tracking_env, "reset", None)
    if callable(reset):
        reset(seed=seed)


def _make_ppo_training_env_factory(
    task: dict[str, Any],
    normalize_actions: bool,
    seed: int,
    action_interface: str = DEFAULT_ACTION_INTERFACE,
    rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE,
    include_dynamics_observation: bool = DEFAULT_INCLUDE_DYNAMICS_OBSERVATION,
    include_previous_action: bool = DEFAULT_INCLUDE_PREVIOUS_ACTION,
    termination_limits: envs.termination.TerminationLimitConfig | Mapping[str, Any] | str | None = None,
    diagnostic_limits: envs.termination.DiagnosticLimitConfig | Mapping[str, Any] | str | None = None,
    task_distribution_settings: envs.task_distribution.TaskDistributionSettings | None = None,
    env_rank: int = 0,
) -> Any:
    """Return a lazy factory for one seeded PPO training environment."""
    task_payload = dict(task)

    def make_env() -> Any:
        """Build the actual PyBullet-backed environment lazily."""
        return _make_seeded_ppo_tracking_env(
            task=task_payload,
            normalize_actions=normalize_actions,
            seed=seed,
            action_interface=action_interface,
            rpm_delta_scale=rpm_delta_scale,
            include_dynamics_observation=include_dynamics_observation,
            include_previous_action=include_previous_action,
            termination_limits=termination_limits,
            diagnostic_limits=diagnostic_limits,
            task_distribution_settings=task_distribution_settings,
            env_rank=env_rank,
        )

    return make_env


def _vec_env_classes() -> tuple[type[Any], type[Any]]:
    """Return Stable-Baselines3 VecEnv classes without importing SB3 at module import time."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv  # noqa: PLC0415

    return DummyVecEnv, SubprocVecEnv


def _vec_monitor_class() -> type[Any]:
    """Return the Stable-Baselines3 VecMonitor class without importing SB3 at module import time."""
    from stable_baselines3.common.vec_env import VecMonitor  # noqa: PLC0415

    return VecMonitor


def _monitor_ppo_training_vec_env(vec_env: Any) -> Any:
    """Wrap a PPO training VecEnv so SB3 receives episode reward and length statistics."""
    vec_monitor_cls = _vec_monitor_class()
    return vec_monitor_cls(vec_env)


def _vec_env_type(num_envs: int) -> str:
    """Return the Stable-Baselines3 vector environment type for a training env count."""
    return "DummyVecEnv" if _positive_int_setting(num_envs, "num_envs") == 1 else "SubprocVecEnv"


def _rank_seed(base_seed: int, env_rank: int) -> int:
    """Derive a deterministic per-environment seed from a base seed and rank."""
    return int(base_seed) + int(env_rank)


def _make_ppo_training_vec_env(
    task: dict[str, Any],
    num_envs: int,
    normalize_actions: bool,
    seed: int,
    action_interface: str = DEFAULT_ACTION_INTERFACE,
    rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE,
    include_dynamics_observation: bool = DEFAULT_INCLUDE_DYNAMICS_OBSERVATION,
    include_previous_action: bool = DEFAULT_INCLUDE_PREVIOUS_ACTION,
    termination_limits: envs.termination.TerminationLimitConfig | Mapping[str, Any] | str | None = None,
    diagnostic_limits: envs.termination.DiagnosticLimitConfig | Mapping[str, Any] | str | None = None,
    task_distribution_settings: envs.task_distribution.TaskDistributionSettings | None = None,
) -> Any:
    """Build the vectorized PPO training environment with lazy per-rank factories."""
    resolved_num_envs = _positive_int_setting(num_envs, "num_envs")
    dummy_vec_env_cls, subproc_vec_env_cls = _vec_env_classes()
    env_factories = []
    for env_rank in range(resolved_num_envs):
        factory_kwargs: dict[str, Any] = {
            "task": task,
            "normalize_actions": normalize_actions,
            "seed": _rank_seed(seed, env_rank),
            "action_interface": action_interface,
            "rpm_delta_scale": rpm_delta_scale,
            "include_dynamics_observation": include_dynamics_observation,
            "include_previous_action": include_previous_action,
            "termination_limits": termination_limits,
            "diagnostic_limits": diagnostic_limits,
        }
        if task_distribution_settings is not None:
            factory_kwargs["task_distribution_settings"] = task_distribution_settings
            factory_kwargs["env_rank"] = env_rank
        env_factories.append(_make_ppo_training_env_factory(**factory_kwargs))
    vec_env = dummy_vec_env_cls(env_factories) if resolved_num_envs == 1 else subproc_vec_env_cls(env_factories, start_method="spawn")
    seed_vec_env = getattr(vec_env, "seed", None)
    if callable(seed_vec_env):
        seed_vec_env(seed)
    return _monitor_ppo_training_vec_env(vec_env)


def _positive_int_setting(value: Any, name: str) -> int:
    """Return a strictly positive integer setting value."""
    if isinstance(value, bool):
        message = f"{name} must be a positive integer"
        raise ValueError(message)  # noqa: TRY004 - public config errors are reported as ValueError.
    try:
        resolved = int(value)
    except (TypeError, ValueError) as exc:
        message = f"{name} must be a positive integer"
        raise ValueError(message) from exc
    if isinstance(value, float) and not value.is_integer():
        message = f"{name} must be a positive integer"
        raise ValueError(message)
    if resolved <= 0:
        message = f"{name} must be a positive integer"
        raise ValueError(message)
    return resolved


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
    task_distribution_settings = _resolved_task_distribution_settings(active_settings, task)
    task_distribution_metadata = _task_distribution_metadata(task_distribution_settings)
    resolved_task_shape = str(task.get("shape", "unknown"))
    training_run_name = _run_name(active_settings, resolved_task_shape)
    timesteps_label = _timesteps_label(active_settings.total_timesteps)
    model_path = _resolve_model_path(active_settings, resolved_task_shape)
    metrics_path = _resolve_metrics_path(active_settings, resolved_task_shape)
    manifest_path = _resolve_manifest_path(active_settings, resolved_task_shape)
    run_manifest_path = _resolve_run_manifest_path(active_settings, training_run_name)
    logs_dir = _resolve_artifact_subdir(active_settings, training_run_name, utils.artifacts.LOGS_DIRNAME)
    diagnostics_dir = _resolve_artifact_subdir(active_settings, training_run_name, utils.artifacts.DIAGNOSTICS_DIRNAME)
    wandb_settings = _wandb_settings(active_settings, resolved_task_shape, task_distribution_metadata)
    config_snapshots: dict[str, str | None] = {}
    if active_settings.artifact_root is None:
        utils.artifacts.ensure_run_training_dirs(training_run_name)
        config_snapshots = _write_direct_config_snapshots(
            settings=active_settings,
            run_name=training_run_name,
            task=task,
            selected_task_index=selected_task_index,
            task_source=task_source,
        )

    warnings = [
        *selection_warnings,
        *(
            _check_tracking_env(
                task,
                normalize_actions=active_settings.normalize_actions,
                action_interface=active_settings.action_interface,
                rpm_delta_scale=active_settings.rpm_delta_scale,
                include_dynamics_observation=active_settings.include_dynamics_observation,
                include_previous_action=active_settings.include_previous_action,
                termination_limits=active_settings.termination_limits,
                diagnostic_limits=active_settings.diagnostic_limits,
            )
            if active_settings.check_env
            else ()
        ),
    ]
    diagnostic_steps = min(active_settings.eval_steps, DEFAULT_LIFTOFF_DIAGNOSTIC_STEPS)
    simple_liftoff_diagnostics = run_liftoff_diagnostics(
        task=task,
        max_steps=diagnostic_steps,
        seed=active_settings.seed,
        action_interface=active_settings.action_interface,
        rpm_delta_scale=active_settings.rpm_delta_scale,
        include_dynamics_observation=active_settings.include_dynamics_observation,
        include_previous_action=active_settings.include_previous_action,
        termination_limits=active_settings.termination_limits,
        diagnostic_limits=active_settings.diagnostic_limits,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if run_manifest_path is not None:
        run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO  # noqa: PLC0415

    wandb_run = None
    try:
        vec_env_type = _vec_env_type(active_settings.num_envs)
        effective_rollout_steps = active_settings.ppo_config.effective_rollout_steps(active_settings.num_envs)
        training_env = _make_ppo_training_vec_env(
            task=task,
            num_envs=active_settings.num_envs,
            normalize_actions=active_settings.normalize_actions,
            seed=active_settings.seed,
            action_interface=active_settings.action_interface,
            rpm_delta_scale=active_settings.rpm_delta_scale,
            include_dynamics_observation=active_settings.include_dynamics_observation,
            include_previous_action=active_settings.include_previous_action,
            termination_limits=active_settings.termination_limits,
            diagnostic_limits=active_settings.diagnostic_limits,
            task_distribution_settings=task_distribution_settings,
        )
        try:
            if active_settings.initial_model_path is None:
                sb3_ppo_kwargs = active_settings.ppo_config.to_sb3_kwargs()
                policy = str(sb3_ppo_kwargs.pop("policy"))
                model = PPO(
                    policy,
                    training_env,
                    seed=active_settings.seed,
                    tensorboard_log=str(logs_dir),
                    verbose=0,
                    **sb3_ppo_kwargs,
                )
            else:
                if not active_settings.initial_model_path.exists():
                    message = f"initial_model_path does not exist: {active_settings.initial_model_path}"
                    raise FileNotFoundError(message)
                model = PPO.load(
                    str(active_settings.initial_model_path),
                    env=training_env,
                    device=active_settings.ppo_config.device,
                    tensorboard_log=str(logs_dir),
                )
            wandb_run = utils.wandb.start_wandb_run(
                settings=wandb_settings,
                config=_wandb_config(
                    active_settings,
                    training_run_name,
                    model_path,
                    metrics_path,
                    manifest_path,
                    logs_dir,
                    diagnostics_dir,
                    selected_task_index,
                    task,
                    task_distribution_metadata,
                ),
            )
            learn_kwargs: dict[str, Any] = {
                "total_timesteps": active_settings.total_timesteps,
                "progress_bar": False,
                "tb_log_name": training_run_name,
            }
            callback = _wandb_callback(wandb_run)
            if callback is not None:
                learn_kwargs["callback"] = callback
            model.learn(**learn_kwargs)
            model.save(str(model_path))
            ppo_device = str(model.device)
            eval_env = _make_seeded_ppo_tracking_env(
                task=task,
                normalize_actions=active_settings.normalize_actions,
                seed=active_settings.seed,
                action_interface=active_settings.action_interface,
                rpm_delta_scale=active_settings.rpm_delta_scale,
                include_dynamics_observation=active_settings.include_dynamics_observation,
                include_previous_action=active_settings.include_previous_action,
                termination_limits=envs.termination.default_termination_limits(),
                diagnostic_limits=active_settings.diagnostic_limits,
            )
            try:
                action_metadata = _tracking_env_action_metadata(eval_env)
                eval_diagnostics = evaluation.diagnostics.collect_policy_evaluation_diagnostics(
                    model=model,
                    tracking_env=eval_env,
                    eval_steps=active_settings.eval_steps,
                    seed=active_settings.seed,
                    training_run_name=training_run_name,
                    task_shape=resolved_task_shape,
                    total_timesteps=active_settings.total_timesteps,
                )
                eval_metrics = eval_diagnostics.metrics
            finally:
                eval_env.close()
        finally:
            training_env.close()

        trained_liftoff_diagnostics = run_liftoff_diagnostics(
            task=task,
            max_steps=diagnostic_steps,
            seed=active_settings.seed,
            model=model,
            include_simple_policies=False,
            action_interface=active_settings.action_interface,
            rpm_delta_scale=active_settings.rpm_delta_scale,
            include_dynamics_observation=active_settings.include_dynamics_observation,
            include_previous_action=active_settings.include_previous_action,
            termination_limits=active_settings.termination_limits,
            diagnostic_limits=active_settings.diagnostic_limits,
        )
        liftoff_diagnostics = {
            **simple_liftoff_diagnostics,
            **trained_liftoff_diagnostics,
        }
        diagnostic_artifact_fields = evaluation.diagnostics.write_policy_evaluation_diagnostics(eval_diagnostics, diagnostics_dir)
        warnings.extend(_movement_warnings(eval_metrics=eval_metrics, action_metadata=action_metadata))

        termination_limits = _settings_termination_limits(active_settings)
        diagnostic_limits = _settings_diagnostic_limits(active_settings)
        metrics: dict[str, Any] = {
            "run_type": "training",
            "run_kind": _metadata_text(active_settings.run_metadata, "run_kind", "direct_ppo"),
            "curriculum_kind": active_settings.run_metadata.get("curriculum_kind"),
            "mode": "ppo_smoke",
            "training_run_name": training_run_name,
            "run_name": training_run_name,
            "source_config_path": str(active_settings.training_config_path) if active_settings.training_config_path is not None else None,
            "training_task_shape": resolved_task_shape,
            "task_shape": resolved_task_shape,
            "task_index": selected_task_index,
            "configured_task_index": active_settings.task_index,
            "task_config_path": str(active_settings.task_config_path),
            "training_config_path": str(active_settings.training_config_path) if active_settings.training_config_path is not None else None,
            "training_config_snapshot_path": config_snapshots.get("training_config_snapshot_path"),
            "training_config_snapshot_path_relative": config_snapshots.get("training_config_snapshot_path_relative"),
            "task_config_snapshot_path": config_snapshots.get("task_config_snapshot_path"),
            "task_config_snapshot_path_relative": config_snapshots.get("task_config_snapshot_path_relative"),
            "task_source": task_source,
            "task_shape_requested": active_settings.task_shape,
            **_task_show_metadata(task),
            **task_distribution_metadata,
            "total_timesteps": active_settings.total_timesteps,
            "num_envs": active_settings.num_envs,
            "action_interface": active_settings.action_interface,
            "ppo_action_dim": action_metadata.get("ppo_action_dim"),
            "real_action_type": action_metadata.get("real_action_type"),
            "real_action_space_bounds": action_metadata.get("real_action_space_bounds"),
            "rpm_delta_scale": active_settings.rpm_delta_scale
            if active_settings.action_interface == envs.actions.ActionInterface.DIRECT_RPM.value
            else None,
            "include_dynamics_observation": active_settings.include_dynamics_observation,
            "include_previous_action": active_settings.include_previous_action,
            "observation_dim": action_metadata.get("observation_dim"),
            "observation_components": action_metadata.get("observation_components"),
            "direct_control_limitations": envs.actions.direct_control_limitations(active_settings.action_interface),
            "vec_env_type": vec_env_type,
            "vec_monitor_enabled": VEC_MONITOR_ENABLED,
            "effective_rollout_steps": effective_rollout_steps,
            "ppo_config": active_settings.ppo_config.to_dict(),
            "policy_kwargs": active_settings.ppo_config.to_dict().get("policy_kwargs"),
            **_net_arch_metadata(active_settings),
            "ppo_profile": _ppo_profile(active_settings),
            "timesteps_label": timesteps_label,
            "eval_steps": active_settings.eval_steps,
            "seed": active_settings.seed,
            "model_path": str(model_path),
            "last_model_path": str(model_path),
            "best_model_path": None,
            "best_model_metric": None,
            "best_model_step": None,
            "best_model_source": "not_selected_no_eval_callback",
            "metrics_path": str(metrics_path),
            "manifest_path": str(manifest_path),
            "run_manifest_path": str(run_manifest_path) if run_manifest_path is not None else None,
            "logs_dir": str(logs_dir),
            "diagnostics_dir": str(diagnostics_dir),
            "dependency_available": dependencies,
            "runtime": runtime_info,
            "ppo_device": ppo_device,
            "action_metadata": action_metadata,
            "liftoff_diagnostics": liftoff_diagnostics,
            "warnings": warnings,
            "trained": True,
            "env_checked": active_settings.check_env,
            "normalize_actions": active_settings.normalize_actions,
            "initial_model_path": str(active_settings.initial_model_path) if active_settings.initial_model_path is not None else None,
            "model_transfer_enabled": active_settings.initial_model_path is not None,
            "model_transfer_source": str(active_settings.initial_model_path) if active_settings.initial_model_path is not None else None,
            "run_metadata": dict(active_settings.run_metadata),
            **dict(active_settings.run_metadata),
            "wandb": _wandb_run_metadata(wandb_settings, wandb_run),
            **eval_metrics,
            "evaluation_termination_limits_mode": eval_metrics.get("termination_limits_mode"),
            "evaluation_termination_limits": eval_metrics.get("termination_limits"),
            "evaluation_diagnostic_limits": eval_metrics.get("diagnostic_limits"),
            "termination_limits_mode": termination_limits.mode,
            "termination_limits": termination_limits.to_dict(),
            "diagnostic_limits": diagnostic_limits.to_dict(),
            "base_truncation_policy": termination_limits.base_truncation_policy,
            "terminate_on_base_truncation": termination_limits.terminate_on_base_truncation,
            **diagnostic_artifact_fields,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest = _build_manifest(active_settings, metrics, task_source=task_source, selected_task_index=selected_task_index, task=task)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if run_manifest_path is not None:
            run_manifest = _build_run_manifest(active_settings, metrics, manifest)
            run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        utils.wandb.log_wandb_summary(wandb_run, metrics)
        artifact_paths = {
            f"{training_run_name}_last_model": model_path,
            f"{training_run_name}_metrics": metrics_path,
            f"{training_run_name}_manifest": manifest_path,
        }
        if metrics.get("best_model_path"):
            artifact_paths[f"{training_run_name}_best_model"] = Path(str(metrics["best_model_path"]))
        if run_manifest_path is not None:
            artifact_paths[f"{training_run_name}_run_manifest"] = run_manifest_path
        artifact_paths.update(_diagnostic_artifact_paths(training_run_name, metrics))
        utils.wandb.log_wandb_artifacts(wandb_run, artifact_paths)
        return PPOTrackingSmokeResult(
            model_path=str(model_path),
            metrics_path=str(metrics_path),
            manifest_path=str(manifest_path),
            metrics=metrics,
            last_model_path=str(model_path),
            best_model_path=None,
            best_model_metric=None,
            best_model_step=None,
            best_model_source="not_selected_no_eval_callback",
            warnings=tuple(warnings),
        )
    finally:
        _finish_wandb_run(wandb_run)


def run_ppo_tracking_smoke_from_config(
    config_path: str | Path = DEFAULT_PPO_TRACKING_CONFIG_PATH,
    task_config_path: str | Path | None = None,
    task_index: int | None = None,
    task_shape: str | None = None,
    task_distribution_config_path: str | Path | None = None,
    run_name: str | None = None,
    total_timesteps: int | None = None,
    num_envs: int | None = None,
    eval_steps: int | None = None,
    output_dir: str | Path | None = None,
    artifact_root: str | Path | None = None,
    model_dir: str | Path | None = None,
    seed: int | None = None,
    wandb_mode: str | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_group: str | None = None,
    wandb_name: str | None = None,
    wandb_tags: tuple[str, ...] | None = None,
    wandb_dir: str | Path | None = None,
    normalize_actions: bool | None = None,
    action_interface: str | None = None,
    initial_model_path: str | Path | None = None,
    run_metadata: Mapping[str, Any] | None = None,
) -> PPOTrackingSmokeResult:
    """
    Load settings, apply CLI-style overrides, and run PPO smoke training.

    Parameters
    ----------
    config_path
        YAML settings path.
    task_config_path
        Optional task-list config path override.
    task_index
        Optional task-index override.
    task_shape
        Optional configured task-shape override.
    task_distribution_config_path
        Optional task-distribution config path override.
    run_name
        Optional storage/runs/<run_name> root for generated artifacts.
    total_timesteps
        Optional PPO timestep-budget override.
    num_envs
        Optional parallel training environment count override.
    eval_steps
        Optional evaluation-step override.
    output_dir
        Optional metrics output directory override.
    artifact_root
        Optional canonical training artifact root override used for curriculum stage training.
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
    normalize_actions
        Optional PPO-facing normalized-action override.
    action_interface
        Optional action-interface override.
    initial_model_path
        Optional Stable-Baselines3 PPO model zip used to initialize training.
    run_metadata
        Optional caller-owned metadata copied into metrics, manifests, and W&B config.

    Returns
    -------
    PPOTrackingSmokeResult
        Saved model path, metrics path, metrics payload, and nonfatal warnings.

    """
    settings = load_ppo_tracking_settings(config_path)
    resolved_task_distribution_config_path = settings.task_distribution_config_path
    resolved_task_distribution_settings = settings.task_distribution_settings
    if task_distribution_config_path is not None:
        resolved_task_distribution_config_path = Path(task_distribution_config_path)
        resolved_task_distribution_settings = envs.task_distribution.load_task_distribution_settings(resolved_task_distribution_config_path)
    overridden = PPOTrackingSmokeSettings(
        task_config_path=settings.task_config_path if task_config_path is None else Path(task_config_path),
        task_index=settings.task_index if task_index is None else task_index,
        task_shape=settings.task_shape if task_shape is None else task_shape,
        task_distribution_config_path=resolved_task_distribution_config_path,
        task_distribution_settings=resolved_task_distribution_settings,
        run_name=settings.run_name if run_name is None else run_name,
        total_timesteps=settings.total_timesteps if total_timesteps is None else total_timesteps,
        num_envs=settings.num_envs if num_envs is None else num_envs,
        ppo_config=settings.ppo_config,
        eval_steps=settings.eval_steps if eval_steps is None else eval_steps,
        seed=settings.seed if seed is None else seed,
        output_dir=settings.output_dir if output_dir is None else Path(output_dir),
        artifact_root=settings.artifact_root if artifact_root is None else Path(artifact_root),
        model_dir=settings.model_dir if model_dir is None else Path(model_dir),
        manifest_filename=settings.manifest_filename,
        model_filename=settings.model_filename,
        metrics_filename=settings.metrics_filename,
        check_env=settings.check_env,
        normalize_actions=settings.normalize_actions if normalize_actions is None else normalize_actions,
        action_interface=settings.action_interface if action_interface is None else action_interface,
        rpm_delta_scale=settings.rpm_delta_scale,
        include_dynamics_observation=settings.include_dynamics_observation,
        include_previous_action=settings.include_previous_action,
        termination_limits=settings.termination_limits,
        diagnostic_limits=settings.diagnostic_limits,
        wandb_mode=settings.wandb_mode if wandb_mode is None else wandb_mode,
        training_config_path=settings.training_config_path,
        wandb_project=settings.wandb_project if wandb_project is None else wandb_project,
        wandb_entity=settings.wandb_entity if wandb_entity is None else wandb_entity,
        wandb_group=settings.wandb_group if wandb_group is None else wandb_group,
        wandb_name=_resolved_wandb_name_override(
            base_wandb_name=settings.wandb_name,
            run_name_override=run_name,
            wandb_name_override=wandb_name,
        ),
        wandb_tags=settings.wandb_tags if wandb_tags is None else wandb_tags,
        wandb_dir=settings.wandb_dir if wandb_dir is None else Path(wandb_dir),
        initial_model_path=settings.initial_model_path if initial_model_path is None else Path(initial_model_path),
        run_metadata=settings.run_metadata if run_metadata is None else dict(run_metadata),
    )
    return run_ppo_tracking_smoke(overridden)


def _settings_from_mapping(config: dict[str, Any], training_config_path: Path | None = None) -> PPOTrackingSmokeSettings:
    """Build settings from a loaded YAML mapping."""
    output_dir_value = config.get("output_dir")
    artifact_root_value = config.get("artifact_root")
    model_dir_value = config.get("model_dir")
    wandb_dir_value = config.get("wandb_dir")
    initial_model_path_value = config.get("initial_model_path")
    task_distribution_config_path_value = config.get("task_distribution_config_path")
    task_distribution_settings = _load_configured_task_distribution_settings(config, task_distribution_config_path_value)
    settings_kwargs: dict[str, Any] = {
        "training_config_path": training_config_path,
        "task_config_path": Path(config.get("task_config_path", DEFAULT_TASK_CONFIG_PATH)),
        "task_index": int(config.get("task_index", DEFAULT_TASK_INDEX)),
        "task_shape": config.get("task_shape") or None,
        "task_distribution_config_path": Path(task_distribution_config_path_value) if task_distribution_config_path_value is not None else None,
        "task_distribution_settings": task_distribution_settings,
        "run_name": config.get("run_name") or None,
        "total_timesteps": int(config.get("total_timesteps", DEFAULT_TOTAL_TIMESTEPS)),
        "num_envs": config.get("num_envs", DEFAULT_NUM_ENVS),
        "ppo_config": ppo_config.load_ppo_config_from_mapping(config),
        "eval_steps": int(config.get("eval_steps", DEFAULT_EVAL_STEPS)),
        "seed": int(config.get("seed", DEFAULT_SEED)),
        "output_dir": Path(output_dir_value) if output_dir_value is not None else None,
        "artifact_root": Path(artifact_root_value) if artifact_root_value is not None else None,
        "model_dir": Path(model_dir_value) if model_dir_value is not None else None,
        "manifest_filename": config.get("manifest_filename") or None,
        "model_filename": config.get("model_filename") or None,
        "metrics_filename": config.get("metrics_filename") or None,
        "check_env": bool(config.get("check_env", True)),
        "normalize_actions": bool(config.get("normalize_actions", DEFAULT_NORMALIZE_ACTIONS)),
        "action_interface": str(config.get("action_interface") or DEFAULT_ACTION_INTERFACE),
        "rpm_delta_scale": config.get("rpm_delta_scale", envs.actions.DEFAULT_RPM_DELTA_SCALE),
        "include_dynamics_observation": bool(config.get("include_dynamics_observation", DEFAULT_INCLUDE_DYNAMICS_OBSERVATION)),
        "include_previous_action": bool(config.get("include_previous_action", DEFAULT_INCLUDE_PREVIOUS_ACTION)),
        "termination_limits": config.get("termination_limits"),
        "diagnostic_limits": config.get("diagnostic_limits"),
        "wandb_mode": str(config.get("wandb_mode") or utils.wandb.WANDB_MODE_AUTO),
        "wandb_project": str(config.get("wandb_project") or utils.wandb.DEFAULT_WANDB_PROJECT),
        "wandb_entity": config.get("wandb_entity") or None,
        "wandb_group": config.get("wandb_group") or None,
        "wandb_name": config.get("wandb_name") or None,
        "wandb_tags": utils.wandb.parse_wandb_tags(config.get("wandb_tags")),
    }
    if wandb_dir_value is not None:
        settings_kwargs["wandb_dir"] = Path(wandb_dir_value)
    if initial_model_path_value is not None:
        settings_kwargs["initial_model_path"] = Path(initial_model_path_value)
    return PPOTrackingSmokeSettings(**settings_kwargs)


def _load_task(task_config_path: Path, task_index: int) -> dict[str, Any]:
    """Load and return a copied task from a task config path."""
    task, _, _, _ = _select_task(task_config_path=task_config_path, default_task_index=task_index, task_shape=None)
    return task


def _load_configured_task_distribution_settings(
    config: Mapping[str, Any],
    task_distribution_config_path_value: Any,
) -> envs.task_distribution.TaskDistributionSettings | None:
    """Load optional task-distribution settings from training config."""
    if task_distribution_config_path_value is not None:
        return envs.task_distribution.load_task_distribution_settings(Path(task_distribution_config_path_value))
    if envs.task_distribution.DISTRIBUTION_CONFIG_KEY in config:
        return envs.task_distribution.load_task_distribution_settings(config)
    return None


def _resolved_task_distribution_settings(
    settings: PPOTrackingSmokeSettings,
    task: Mapping[str, Any],
) -> envs.task_distribution.TaskDistributionSettings | None:
    """Return explicit distribution settings or a valid fixed-task distribution."""
    if settings.task_distribution_settings is not None:
        return settings.task_distribution_settings
    try:
        return envs.task_distribution.normalize_fixed_task_to_distribution(
            task,
            seed=settings.seed,
            name="fixed_task",
            config_path=settings.task_config_path,
        )
    except (TypeError, ValueError):
        return None


_TASK_SHOW_METADATA_KEYS = (
    "training_task_kind",
    "task_is_distribution",
    "task_is_show",
    "show_name",
    "scenario_name",
    "segment_count",
    "meaningful_figure_count",
    "segment_shapes",
    "show_is_continuous",
    "continuity_tolerance",
    "difficulty_level",
    "duration_range_sec",
    "move_duration_range_sec",
    "segment_duration_range_sec",
    "path_length_range_m",
    "approx_reference_speed_range_mps",
    "segment_speed_bounds",
    "sampled_per_episode",
    "constant_within_episode",
    "variation_enabled",
    "variation_mode",
    "final_hold_enabled",
    "final_hold_sec",
    "own_task_eval_path",
    "generalization_eval_path",
    "scenario_eval_path",
    "requested_task_family",
    "accepted_task_family",
    "variation_strength",
    "proposed_sampling_bounds",
    "accepted_sampling_bounds",
    "repair_was_applied",
    "repair_reason",
)


def _task_show_metadata(task: Mapping[str, Any]) -> dict[str, Any]:
    """Select compact composed-show metadata from a training task."""
    metadata = {key: copy.deepcopy(task[key]) for key in _TASK_SHOW_METADATA_KEYS if key in task}
    segments = task.get(validation.contracts.FIELD_SEGMENTS)
    if isinstance(segments, list):
        metadata[validation.contracts.FIELD_SEGMENTS] = copy.deepcopy(segments)
        metadata.setdefault("segment_count", len(segments))
        metadata.setdefault(
            "segment_shapes",
            [str(segment.get(validation.contracts.FIELD_SEGMENT_SHAPE, "unknown")) for segment in segments if isinstance(segment, Mapping)],
        )
    return metadata


def _task_show_manifest_fields(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Select compact task-show fields for manifests and run indexes."""
    keys = (*_TASK_SHOW_METADATA_KEYS, validation.contracts.FIELD_SEGMENTS)
    return {key: copy.deepcopy(metrics[key]) for key in keys if key in metrics}


def _task_distribution_metadata(settings: envs.task_distribution.TaskDistributionSettings | None) -> dict[str, Any]:
    """Return compact metadata for task distribution settings."""
    if settings is None:
        return {
            "task_distribution_enabled": False,
            "task_distribution_mode": None,
            "task_distribution_strength": 0.0,
            "task_distribution_sample_on_reset": False,
            "task_distribution_seed": None,
            "task_distribution_config_path": None,
            "task_distribution_supported_families": list(envs.task_distribution.supported_task_families()),
            "task_distribution_family_weights": {},
            "task_distribution_name": None,
        }
    return settings.to_metadata()


def _task_distribution_manifest_fields(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Select task-distribution fields for metrics, manifests, and W&B config."""
    keys = (
        "task_distribution_enabled",
        "task_distribution_mode",
        "task_distribution_strength",
        "task_distribution_sample_on_reset",
        "task_distribution_seed",
        "task_distribution_config_path",
        "task_distribution_supported_families",
        "task_distribution_family_weights",
        "task_distribution_name",
        "task_distribution_base_task_shape",
    )
    return {key: metrics.get(key) for key in keys if key in metrics}


def _select_task(
    task_config_path: Path,
    default_task_index: int,
    task_shape: str | None,
) -> tuple[dict[str, Any], str, int, tuple[str, ...]]:
    """Load one configured training task by index or shape."""
    config = config_loader.load_experiment_config(task_config_path)
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


def _timesteps_label(total_timesteps: int) -> str:
    """Return a compact timestep label for generated training run names."""
    if total_timesteps >= _TIMESTEPS_PER_MILLION_LABEL and total_timesteps % _TIMESTEPS_PER_MILLION_LABEL == 0:
        return f"{total_timesteps // _TIMESTEPS_PER_MILLION_LABEL}m"
    if total_timesteps >= _MIN_TIMESTEPS_FOR_COMPACT_THOUSAND_LABEL and total_timesteps % _TIMESTEPS_PER_THOUSAND_LABEL == 0:
        return f"{total_timesteps // _TIMESTEPS_PER_THOUSAND_LABEL}k"
    return str(total_timesteps)


def _auto_run_name(task_shape: str, seed: int) -> str:
    """Build the default direct-PPO run name from resolved settings."""
    return f"direct_ppo_{task_shape}_seed{seed}"


def _default_training_run_name() -> str:
    """Derive the run name produced by the default training config."""
    settings = load_ppo_tracking_settings(DEFAULT_PPO_TRACKING_CONFIG_PATH)
    task, _, _, _ = _select_task(
        task_config_path=settings.task_config_path,
        default_task_index=settings.task_index,
        task_shape=settings.task_shape,
    )
    return _run_name(settings, str(task.get("shape", "unknown")))


def _run_name(settings: PPOTrackingSmokeSettings, task_shape: str | None = None) -> str:
    """Return the explicit or derived training run name."""
    if settings.run_name is not None:
        return settings.run_name
    resolved_shape = task_shape or settings.task_shape
    if resolved_shape is None:
        message = "task_shape is required to derive a training run name"
        raise ValueError(message)
    return _auto_run_name(resolved_shape, settings.seed)


def _resolve_model_path(settings: PPOTrackingSmokeSettings, task_shape: str | None = None) -> Path:
    """Resolve the trained model output path."""
    run_name = _run_name(settings, task_shape)
    default_dir = _default_artifact_subdir(settings, run_name, utils.artifacts.MODELS_DIRNAME)
    filename = settings.model_filename or f"{run_name}.zip"
    return _resolve_directory(settings.model_dir, default_dir) / filename


def _resolve_metrics_path(settings: PPOTrackingSmokeSettings, task_shape: str | None = None) -> Path:
    """Resolve the metrics output path."""
    run_name = _run_name(settings, task_shape)
    filename = settings.metrics_filename or f"{run_name}_metrics.json"
    if settings.output_dir is None:
        return _default_artifact_subdir(settings, run_name, utils.artifacts.METRICS_DIRNAME) / filename
    output_dir = settings.output_dir.expanduser().resolve(strict=False)
    if "results" in output_dir.parts or output_dir.name == utils.artifacts.METRICS_DIRNAME:
        return output_dir / filename
    return output_dir / utils.artifacts.METRICS_DIRNAME / filename


def _resolve_manifest_path(settings: PPOTrackingSmokeSettings, task_shape: str | None = None) -> Path:
    """Resolve the training manifest output path."""
    run_name = _run_name(settings, task_shape)
    canonical_filename = settings.manifest_filename or utils.artifacts.MANIFEST_FILENAME
    legacy_override_filename = settings.manifest_filename or f"{run_name}_manifest.json"
    if settings.output_dir is None:
        return (_default_artifact_root(settings, run_name) / canonical_filename).expanduser().resolve(strict=False)
    output_dir = settings.output_dir.expanduser().resolve(strict=False)
    if "results" in output_dir.parts or output_dir.name == utils.artifacts.MANIFESTS_DIRNAME:
        return output_dir / legacy_override_filename
    return output_dir / utils.artifacts.MANIFESTS_DIRNAME / legacy_override_filename


def _resolve_run_manifest_path(settings: PPOTrackingSmokeSettings, run_name: str) -> Path | None:
    """Resolve the root run manifest path for a direct canonical training run."""
    if settings.artifact_root is not None:
        return None
    return utils.artifacts.get_run_manifest_path(run_name).expanduser().resolve(strict=False)


def _resolve_artifact_subdir(settings: PPOTrackingSmokeSettings, run_name: str, subdir: str) -> Path:
    """Resolve a standard artifact subdirectory for this training run."""
    return _default_artifact_subdir(settings, run_name, subdir).expanduser().resolve(strict=False)


def _default_artifact_subdir(settings: PPOTrackingSmokeSettings, run_name: str, subdir: str) -> Path:
    """Return a canonical training artifact subdirectory."""
    return _default_artifact_root(settings, run_name) / subdir


def _default_artifact_root(settings: PPOTrackingSmokeSettings, run_name: str) -> Path:
    """Return the canonical training artifact root for metrics, models, and logs."""
    if settings.artifact_root is not None:
        return settings.artifact_root
    return utils.artifacts.get_run_training_dir(run_name)


def _resolve_directory(path: Path | None, default: Path) -> Path:
    """Resolve a configured directory or its storage-backed default."""
    directory = default if path is None else path
    return directory.expanduser().resolve(strict=False)


def _wandb_settings(
    settings: PPOTrackingSmokeSettings,
    task_shape: str | None = None,
    task_distribution_metadata: Mapping[str, Any] | None = None,
) -> utils.wandb.WandbTrackingSettings:
    """Build W&B settings from PPO smoke settings and resolved task metadata."""
    run_name = _run_name(settings, task_shape)
    resolved_shape = task_shape or settings.task_shape or "unknown"
    return utils.wandb.WandbTrackingSettings(
        mode=settings.wandb_mode,
        project=settings.wandb_project,
        entity=settings.wandb_entity,
        group=_wandb_group(settings, run_name, resolved_shape, task_distribution_metadata),
        name=settings.wandb_name or run_name,
        tags=_wandb_tags(settings, resolved_shape, task_distribution_metadata),
        dir=settings.wandb_dir or _resolve_artifact_subdir(settings, run_name, utils.artifacts.WANDB_DIRNAME),
    )


def _wandb_group(
    settings: PPOTrackingSmokeSettings,
    run_name: str,
    task_shape: str,
    task_distribution_metadata: Mapping[str, Any] | None,
) -> str:
    """Return the W&B group for direct runs or curriculum stages."""
    if _is_curriculum_stage(settings):
        curriculum_kind = _metadata_text(settings.run_metadata, "curriculum_kind", "curriculum")
        curriculum_run_name = _metadata_text(settings.run_metadata, "curriculum_run_name", settings.wandb_group or run_name)
        return settings.wandb_group or f"curriculum/{curriculum_kind}/{curriculum_run_name}"
    if settings.wandb_group and not settings.wandb_group.startswith("direct_ppo/"):
        return settings.wandb_group
    task_distribution = _task_distribution_tag_value(task_distribution_metadata)
    variant = _direct_ppo_variant_label(run_name=run_name, task_shape=task_shape, seed=settings.seed)
    return f"direct_ppo/{settings.action_interface}/{task_distribution}/{variant}/seed{settings.seed}"


def _wandb_tags(
    settings: PPOTrackingSmokeSettings,
    task_shape: str,
    task_distribution_metadata: Mapping[str, Any] | None = None,
) -> tuple[str, ...]:
    """Return derived and user-provided W&B tags without duplicates."""
    config_stem = settings.training_config_path.stem if settings.training_config_path is not None else "direct"
    curriculum_kind = settings.run_metadata.get("curriculum_kind")
    run_kind = _metadata_text(settings.run_metadata, "run_kind", "direct_ppo")
    run_kind_tags = ("curriculum", str(curriculum_kind)) if run_kind == "curriculum_stage" and curriculum_kind else ("direct_ppo",)
    observation_tags = (("observation:dynamics",) if settings.include_dynamics_observation else ()) + (
        ("observation:previous_action",) if settings.include_previous_action else ()
    )
    derived = (
        "ppo",
        *run_kind_tags,
        "training",
        f"task:{task_shape}",
        f"action_interface:{settings.action_interface}",
        *observation_tags,
        f"task_distribution:{_task_distribution_tag_value(task_distribution_metadata)}",
        f"net:{_net_arch_label(settings)}",
        f"ppo_profile:{_ppo_profile(settings)}",
        f"steps:{_timesteps_label(settings.total_timesteps)}",
        f"seed:{settings.seed}",
        f"config:{config_stem}",
    )
    return _dedupe_tags((*derived, *settings.wandb_tags))


def _resolved_wandb_name_override(
    *,
    base_wandb_name: str | None,
    run_name_override: str | None,
    wandb_name_override: str | None,
) -> str | None:
    """Return the effective W&B name override for config-loaded runs."""
    if wandb_name_override is not None:
        return wandb_name_override
    if run_name_override is not None:
        return run_name_override
    return base_wandb_name


def _is_curriculum_stage(settings: PPOTrackingSmokeSettings) -> bool:
    """Return whether caller metadata marks this PPO run as a curriculum stage."""
    return settings.run_metadata.get("run_kind") == "curriculum_stage"


def _metadata_text(metadata: Mapping[str, Any], key: str, default: str) -> str:
    """Return a non-empty metadata string with a fallback."""
    value = metadata.get(key)
    if value is None or not str(value).strip():
        return default
    return str(value)


def _task_distribution_tag_value(task_distribution_metadata: Mapping[str, Any] | None) -> str:
    """Return a compact task-distribution identity for tags and groups."""
    if not task_distribution_metadata:
        return "fixed"
    name = task_distribution_metadata.get("task_distribution_name")
    if isinstance(name, str) and name.strip() and name != "fixed_task":
        return name
    mode = task_distribution_metadata.get("task_distribution_mode")
    if mode == "fixed":
        return "fixed"
    return str(mode or "fixed")


def _net_arch_label(settings: PPOTrackingSmokeSettings) -> str:
    """Return a compact network-architecture tag label."""
    net_arch = _policy_net_arch(settings)
    net128_flat_arch = [128, 128]
    net128_policy_arch = {"pi": net128_flat_arch, "vf": net128_flat_arch}
    net256_flat_arch = [256, 256]
    net256_policy_arch = {"pi": net256_flat_arch, "vf": net256_flat_arch}
    if net_arch in (net128_flat_arch, net128_policy_arch):
        return "net128_default"
    if net_arch in (net256_flat_arch, net256_policy_arch):
        return "net256_large"
    return "custom"


def _policy_net_arch(settings: PPOTrackingSmokeSettings) -> Any:
    """Return a JSON-ready copy of the resolved PPO net architecture."""
    policy_kwargs = settings.ppo_config.policy_kwargs or {}
    net_arch = policy_kwargs.get("net_arch")
    if isinstance(net_arch, dict):
        return {key: list(value) for key, value in net_arch.items()}
    if isinstance(net_arch, list):
        return list(net_arch)
    return net_arch


def _net_arch_metadata(settings: PPOTrackingSmokeSettings) -> dict[str, Any]:
    """Return network-architecture metadata for metrics, manifests, and W&B config."""
    label = _net_arch_label(settings)
    return {
        "net_arch_label": label,
        "policy_net_arch": _policy_net_arch(settings),
        "uses_default_net128_arch": label == "net128_default",
        "uses_large_net_variant": label == "net256_large",
    }


def _ppo_profile(settings: PPOTrackingSmokeSettings) -> str:
    """Infer the PPO profile represented by the resolved hyperparameters."""
    if settings.ppo_config.ent_coef == _ENT005_COEFFICIENT:
        return "ent005"
    if settings.ppo_config.learning_rate == _LOW_LR_LEARNING_RATE:
        return "low_lr"
    return "default"


def _direct_ppo_variant_label(run_name: str, task_shape: str, seed: int) -> str:
    """Return the run variant segment used inside direct-PPO W&B groups."""
    prefix = "direct_ppo_"
    suffix = f"_seed{seed}"
    if run_name.startswith(prefix) and run_name.endswith(suffix):
        variant = run_name[len(prefix) : -len(suffix)]
        return variant or task_shape
    return task_shape


def _dedupe_tags(tags: tuple[str, ...]) -> tuple[str, ...]:
    """Deduplicate tags while preserving their first occurrence."""
    unique: list[str] = []
    seen: set[str] = set()
    for tag in tags:
        clean = tag.strip()
        if not clean or clean in seen:
            continue
        unique.append(clean)
        seen.add(clean)
    return tuple(unique)


def _wandb_run_metadata(settings: utils.wandb.WandbTrackingSettings, run: Any | None) -> dict[str, Any]:
    """Return resolved W&B metadata for metrics and manifests."""
    resolved_mode = utils.wandb.resolve_wandb_mode(settings.mode)
    return {
        "enabled": resolved_mode != utils.wandb.WANDB_MODE_DISABLED,
        "mode": settings.mode,
        "resolved_mode": resolved_mode,
        "project": settings.project,
        "entity": settings.entity,
        "group": settings.group,
        "name": settings.name,
        "tags": list(settings.tags),
        "dir": str(settings.dir) if settings.dir is not None else None,
        "run_id": _optional_wandb_string(getattr(run, "id", None) if run is not None else None),
        "url": _optional_wandb_string(getattr(run, "url", None) if run is not None else None),
    }


def _optional_wandb_string(value: Any | None) -> str | None:
    """Return a non-empty W&B metadata string or ``None``."""
    if value is None:
        return None
    text = str(value)
    return text if text and text != "None" else None


def _wandb_callback(run: Any | None) -> Any | None:
    """Return the official SB3/W&B callback when tracking and integration are available."""
    if run is None:
        return None
    try:
        from wandb.integration.sb3 import WandbCallback  # noqa: PLC0415
    except ImportError:
        return None
    return WandbCallback(verbose=0)


def _finish_wandb_run(run: Any | None) -> None:
    """Finish an active W&B run without masking an already-raising exception."""
    if run is None:
        return
    has_active_exception = sys.exc_info()[0] is not None
    try:
        run.finish()
    except Exception:
        if not has_active_exception:
            raise


def _build_manifest(
    settings: PPOTrackingSmokeSettings,
    metrics: dict[str, Any],
    task_source: str,
    selected_task_index: int,
    task: dict[str, Any],
) -> dict[str, Any]:
    """Build a manifest payload for a PPO training run."""
    run_name = str(metrics["training_run_name"])
    diagnostics = _diagnostic_manifest_fields(metrics)
    return {
        "run_type": metrics.get("run_type", "training"),
        "run_kind": metrics.get("run_kind", "direct_ppo"),
        "curriculum_kind": metrics.get("curriculum_kind"),
        "mode": "ppo_smoke",
        "training_run_name": run_name,
        "run_name": run_name,
        "training_config_path": str(settings.training_config_path) if settings.training_config_path is not None else None,
        "source_config_path": metrics.get("source_config_path"),
        "task_config_path": str(settings.task_config_path),
        "training_config_snapshot_path": metrics.get("training_config_snapshot_path"),
        "training_config_snapshot_path_relative": metrics.get("training_config_snapshot_path_relative"),
        "task_config_snapshot_path": metrics.get("task_config_snapshot_path"),
        "task_config_snapshot_path_relative": metrics.get("task_config_snapshot_path_relative"),
        "task_source": task_source,
        "task_index": selected_task_index,
        "task_shape": str(task.get("shape", "unknown")),
        "training_task_shape": str(task.get("shape", "unknown")),
        "task_shape_requested": settings.task_shape,
        **_task_show_metadata(task),
        **_task_show_manifest_fields(metrics),
        **_task_distribution_manifest_fields(metrics),
        "total_timesteps": settings.total_timesteps,
        "num_envs": metrics.get("num_envs", settings.num_envs),
        "action_interface": metrics.get("action_interface", settings.action_interface),
        "ppo_action_dim": metrics.get("ppo_action_dim"),
        "real_action_type": metrics.get("real_action_type"),
        "real_action_space_bounds": metrics.get("real_action_space_bounds"),
        "rpm_delta_scale": metrics.get("rpm_delta_scale"),
        "include_dynamics_observation": metrics.get("include_dynamics_observation", settings.include_dynamics_observation),
        "include_previous_action": metrics.get("include_previous_action", settings.include_previous_action),
        "observation_dim": metrics.get("observation_dim"),
        "observation_components": metrics.get("observation_components"),
        "direct_control_limitations": list(metrics.get("direct_control_limitations", [])),
        "termination_limits_mode": metrics.get("termination_limits_mode"),
        "termination_limits": metrics.get("termination_limits"),
        "diagnostic_limits": metrics.get("diagnostic_limits"),
        "base_truncation_policy": metrics.get("base_truncation_policy"),
        "terminate_on_base_truncation": metrics.get("terminate_on_base_truncation"),
        "evaluation_termination_limits_mode": metrics.get("evaluation_termination_limits_mode"),
        "evaluation_termination_limits": metrics.get("evaluation_termination_limits"),
        "evaluation_diagnostic_limits": metrics.get("evaluation_diagnostic_limits"),
        "vec_env_type": metrics.get("vec_env_type", _vec_env_type(settings.num_envs)),
        "vec_monitor_enabled": metrics.get("vec_monitor_enabled", VEC_MONITOR_ENABLED),
        "effective_rollout_steps": metrics.get(
            "effective_rollout_steps",
            settings.ppo_config.effective_rollout_steps(settings.num_envs),
        ),
        "ppo_config": metrics.get("ppo_config", settings.ppo_config.to_dict()),
        "policy_kwargs": metrics.get("policy_kwargs", settings.ppo_config.to_dict().get("policy_kwargs")),
        "net_arch_label": metrics.get("net_arch_label", _net_arch_label(settings)),
        "policy_net_arch": metrics.get("policy_net_arch", _policy_net_arch(settings)),
        "uses_default_net128_arch": metrics.get("uses_default_net128_arch", _net_arch_label(settings) == "net128_default"),
        "uses_large_net_variant": metrics.get("uses_large_net_variant", _net_arch_label(settings) == "net256_large"),
        "ppo_profile": metrics.get("ppo_profile", _ppo_profile(settings)),
        "eval_steps": settings.eval_steps,
        "seed": settings.seed,
        "normalize_actions": settings.normalize_actions,
        "initial_model_path": metrics.get("initial_model_path"),
        "model_transfer_enabled": metrics.get("model_transfer_enabled", False),
        "model_transfer_source": metrics.get("model_transfer_source"),
        "model_path": metrics["model_path"],
        "model_path_relative": utils.artifacts.path_relative_to(metrics["model_path"], _manifest_relative_base(settings, run_name)),
        "last_model_path": metrics.get("last_model_path", metrics["model_path"]),
        "last_model_path_relative": utils.artifacts.path_relative_to(
            metrics.get("last_model_path", metrics["model_path"]), _manifest_relative_base(settings, run_name)
        ),
        "best_model_path": metrics.get("best_model_path"),
        "best_model_path_relative": utils.artifacts.path_relative_to(metrics.get("best_model_path"), _manifest_relative_base(settings, run_name)),
        "best_model_metric": metrics.get("best_model_metric"),
        "best_model_step": metrics.get("best_model_step"),
        "best_model_source": metrics.get("best_model_source"),
        "metrics_path": metrics["metrics_path"],
        "metrics_path_relative": utils.artifacts.path_relative_to(metrics["metrics_path"], _manifest_relative_base(settings, run_name)),
        "manifest_path": metrics["manifest_path"],
        "manifest_path_relative": utils.artifacts.path_relative_to(metrics["manifest_path"], _manifest_relative_base(settings, run_name)),
        "run_manifest_path": metrics.get("run_manifest_path"),
        "output_dir": str(settings.output_dir or _default_artifact_root(settings, run_name)),
        "logs_dir": metrics["logs_dir"],
        "logs_dir_relative": utils.artifacts.path_relative_to(metrics["logs_dir"], _manifest_relative_base(settings, run_name)),
        "diagnostics_dir": metrics.get("diagnostics_dir"),
        "diagnostics_dir_relative": utils.artifacts.path_relative_to(metrics.get("diagnostics_dir"), _manifest_relative_base(settings, run_name)),
        "warnings": list(metrics.get("warnings", [])),
        "run_metadata": dict(metrics.get("run_metadata", {})),
        **_run_metadata_manifest_fields(metrics),
        "wandb": metrics.get("wandb", {}),
        "diagnostics": diagnostics,
        **diagnostics,
    }


def _build_run_manifest(
    settings: PPOTrackingSmokeSettings,
    metrics: dict[str, Any],
    training_manifest: dict[str, Any],
) -> dict[str, Any]:
    """Build the root manifest payload for a unified storage/runs training run."""
    run_name = str(metrics["training_run_name"])
    return {
        "run_type": metrics.get("run_type", "training"),
        "run_kind": metrics.get("run_kind", "direct_ppo"),
        "curriculum_kind": metrics.get("curriculum_kind"),
        "mode": "ppo_smoke",
        "run_name": run_name,
        "training_run_name": run_name,
        "source_config_path": metrics.get("source_config_path"),
        "run_manifest_path": metrics.get("run_manifest_path"),
        "training": {
            "manifest_path": metrics["manifest_path"],
            "manifest_path_relative": utils.artifacts.path_relative_to_run(metrics["manifest_path"], run_name),
            "model_path": metrics["model_path"],
            "model_path_relative": utils.artifacts.path_relative_to_run(metrics["model_path"], run_name),
            "last_model_path": metrics.get("last_model_path", metrics["model_path"]),
            "last_model_path_relative": utils.artifacts.path_relative_to_run(metrics.get("last_model_path", metrics["model_path"]), run_name),
            "best_model_path": metrics.get("best_model_path"),
            "best_model_path_relative": utils.artifacts.path_relative_to_run(metrics.get("best_model_path"), run_name),
            "best_model_metric": metrics.get("best_model_metric"),
            "best_model_step": metrics.get("best_model_step"),
            "best_model_source": metrics.get("best_model_source"),
            "metrics_path": metrics["metrics_path"],
            "metrics_path_relative": utils.artifacts.path_relative_to_run(metrics["metrics_path"], run_name),
            "logs_dir": metrics["logs_dir"],
            "logs_dir_relative": utils.artifacts.path_relative_to_run(metrics["logs_dir"], run_name),
            "diagnostics_dir": metrics.get("diagnostics_dir"),
            "diagnostics_dir_relative": utils.artifacts.path_relative_to_run(metrics.get("diagnostics_dir"), run_name),
        },
        "config": {
            "training_config_path": str(settings.training_config_path) if settings.training_config_path is not None else None,
            "source_config_path": metrics.get("source_config_path"),
            "task_config_path": str(settings.task_config_path),
            "training_config_snapshot_path": metrics.get("training_config_snapshot_path"),
            "training_config_snapshot_path_relative": metrics.get("training_config_snapshot_path_relative"),
            "task_config_snapshot_path": metrics.get("task_config_snapshot_path"),
            "task_config_snapshot_path_relative": metrics.get("task_config_snapshot_path_relative"),
            "task_shape": training_manifest["task_shape"],
            "task_shape_requested": settings.task_shape,
            "task_index": training_manifest["task_index"],
            "task_source": training_manifest["task_source"],
            **_task_show_manifest_fields(metrics),
            **_task_distribution_manifest_fields(metrics),
            "ppo_config": metrics.get("ppo_config", settings.ppo_config.to_dict()),
            "policy_kwargs": metrics.get("policy_kwargs", settings.ppo_config.to_dict().get("policy_kwargs")),
            "net_arch_label": metrics.get("net_arch_label", _net_arch_label(settings)),
            "policy_net_arch": metrics.get("policy_net_arch", _policy_net_arch(settings)),
            "uses_default_net128_arch": metrics.get("uses_default_net128_arch", _net_arch_label(settings) == "net128_default"),
            "uses_large_net_variant": metrics.get("uses_large_net_variant", _net_arch_label(settings) == "net256_large"),
            "ppo_profile": metrics.get("ppo_profile", _ppo_profile(settings)),
            "num_envs": metrics.get("num_envs", settings.num_envs),
            "action_interface": metrics.get("action_interface", settings.action_interface),
            "ppo_action_dim": metrics.get("ppo_action_dim"),
            "real_action_type": metrics.get("real_action_type"),
            "real_action_space_bounds": metrics.get("real_action_space_bounds"),
            "rpm_delta_scale": metrics.get("rpm_delta_scale"),
            "include_dynamics_observation": metrics.get("include_dynamics_observation", settings.include_dynamics_observation),
            "include_previous_action": metrics.get("include_previous_action", settings.include_previous_action),
            "observation_dim": metrics.get("observation_dim"),
            "observation_components": metrics.get("observation_components"),
            "direct_control_limitations": list(metrics.get("direct_control_limitations", [])),
            "termination_limits_mode": metrics.get("termination_limits_mode"),
            "termination_limits": metrics.get("termination_limits"),
            "diagnostic_limits": metrics.get("diagnostic_limits"),
            "base_truncation_policy": metrics.get("base_truncation_policy"),
            "terminate_on_base_truncation": metrics.get("terminate_on_base_truncation"),
            "evaluation_termination_limits_mode": metrics.get("evaluation_termination_limits_mode"),
            "evaluation_termination_limits": metrics.get("evaluation_termination_limits"),
            "evaluation_diagnostic_limits": metrics.get("evaluation_diagnostic_limits"),
            "vec_env_type": metrics.get("vec_env_type", _vec_env_type(settings.num_envs)),
            "vec_monitor_enabled": metrics.get("vec_monitor_enabled", VEC_MONITOR_ENABLED),
            "effective_rollout_steps": metrics.get(
                "effective_rollout_steps",
                settings.ppo_config.effective_rollout_steps(settings.num_envs),
            ),
        },
        **_task_distribution_manifest_fields(metrics),
        "total_timesteps": settings.total_timesteps,
        "policy_kwargs": metrics.get("policy_kwargs", settings.ppo_config.to_dict().get("policy_kwargs")),
        "net_arch_label": metrics.get("net_arch_label", _net_arch_label(settings)),
        "policy_net_arch": metrics.get("policy_net_arch", _policy_net_arch(settings)),
        "uses_default_net128_arch": metrics.get("uses_default_net128_arch", _net_arch_label(settings) == "net128_default"),
        "uses_large_net_variant": metrics.get("uses_large_net_variant", _net_arch_label(settings) == "net256_large"),
        "ppo_profile": metrics.get("ppo_profile", _ppo_profile(settings)),
        "num_envs": metrics.get("num_envs", settings.num_envs),
        "action_interface": metrics.get("action_interface", settings.action_interface),
        "ppo_action_dim": metrics.get("ppo_action_dim"),
        "real_action_type": metrics.get("real_action_type"),
        "real_action_space_bounds": metrics.get("real_action_space_bounds"),
        "rpm_delta_scale": metrics.get("rpm_delta_scale"),
        "include_dynamics_observation": metrics.get("include_dynamics_observation", settings.include_dynamics_observation),
        "include_previous_action": metrics.get("include_previous_action", settings.include_previous_action),
        "observation_dim": metrics.get("observation_dim"),
        "observation_components": metrics.get("observation_components"),
        "direct_control_limitations": list(metrics.get("direct_control_limitations", [])),
        "termination_limits_mode": metrics.get("termination_limits_mode"),
        "termination_limits": metrics.get("termination_limits"),
        "diagnostic_limits": metrics.get("diagnostic_limits"),
        "base_truncation_policy": metrics.get("base_truncation_policy"),
        "terminate_on_base_truncation": metrics.get("terminate_on_base_truncation"),
        "evaluation_termination_limits_mode": metrics.get("evaluation_termination_limits_mode"),
        "evaluation_termination_limits": metrics.get("evaluation_termination_limits"),
        "evaluation_diagnostic_limits": metrics.get("evaluation_diagnostic_limits"),
        "vec_env_type": metrics.get("vec_env_type", _vec_env_type(settings.num_envs)),
        "vec_monitor_enabled": metrics.get("vec_monitor_enabled", VEC_MONITOR_ENABLED),
        "effective_rollout_steps": metrics.get(
            "effective_rollout_steps",
            settings.ppo_config.effective_rollout_steps(settings.num_envs),
        ),
        "eval_steps": settings.eval_steps,
        "seed": settings.seed,
        "normalize_actions": settings.normalize_actions,
        "initial_model_path": metrics.get("initial_model_path"),
        "model_transfer_enabled": metrics.get("model_transfer_enabled", False),
        "model_transfer_source": metrics.get("model_transfer_source"),
        "model_path": metrics["model_path"],
        "model_path_relative": utils.artifacts.path_relative_to_run(metrics["model_path"], run_name),
        "last_model_path": metrics.get("last_model_path", metrics["model_path"]),
        "last_model_path_relative": utils.artifacts.path_relative_to_run(metrics.get("last_model_path", metrics["model_path"]), run_name),
        "best_model_path": metrics.get("best_model_path"),
        "best_model_path_relative": utils.artifacts.path_relative_to_run(metrics.get("best_model_path"), run_name),
        "best_model_metric": metrics.get("best_model_metric"),
        "best_model_step": metrics.get("best_model_step"),
        "best_model_source": metrics.get("best_model_source"),
        "wandb": metrics.get("wandb", {}),
        "run_metadata": dict(metrics.get("run_metadata", {})),
        **_run_metadata_manifest_fields(metrics),
        "warnings": list(metrics.get("warnings", [])),
        "evaluation_index": _evaluation_index_manifest(run_name),
    }


def _write_direct_config_snapshots(
    settings: PPOTrackingSmokeSettings,
    run_name: str,
    task: dict[str, Any],
    selected_task_index: int,
    task_source: str,
) -> dict[str, str | None]:
    """Write direct PPO config snapshots and return manifest path fields."""
    training_snapshot_path = utils.artifacts.get_run_training_config_snapshot_path(run_name)
    task_snapshot_path = utils.artifacts.get_run_task_config_snapshot_path(run_name)
    training_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    _write_config_snapshot(
        source_path=settings.training_config_path,
        destination_path=training_snapshot_path,
        fallback_payload=_training_config_snapshot_payload(settings),
    )
    _write_config_snapshot(
        source_path=settings.task_config_path,
        destination_path=task_snapshot_path,
        fallback_payload={
            "name": f"{run_name}_selected_task",
            "task_source": task_source,
            "task_index": selected_task_index,
            "tasks": [task],
        },
    )
    return {
        "training_config_snapshot_path": str(training_snapshot_path),
        "training_config_snapshot_path_relative": str(utils.artifacts.path_relative_to_run(training_snapshot_path, run_name)),
        "task_config_snapshot_path": str(task_snapshot_path),
        "task_config_snapshot_path_relative": str(utils.artifacts.path_relative_to_run(task_snapshot_path, run_name)),
    }


def _write_config_snapshot(source_path: Path | None, destination_path: Path, fallback_payload: dict[str, Any]) -> None:
    """Copy a YAML source config exactly when available, otherwise materialize a fallback."""
    if source_path is not None and source_path.is_file():
        destination_path.write_text(source_path.read_text(encoding="utf-8"), encoding="utf-8")
        return
    destination_path.write_text(_to_yaml(fallback_payload), encoding="utf-8")


def _training_config_snapshot_payload(settings: PPOTrackingSmokeSettings) -> dict[str, Any]:
    """Build a fallback training config snapshot from resolved settings."""
    termination_limits = _settings_termination_limits(settings)
    diagnostic_limits = _settings_diagnostic_limits(settings)
    return {
        "task_config_path": str(settings.task_config_path),
        "task_index": settings.task_index,
        "task_shape": settings.task_shape,
        "task_distribution_config_path": None if settings.task_distribution_config_path is None else str(settings.task_distribution_config_path),
        "task_distribution": None if settings.task_distribution_settings is None else settings.task_distribution_settings.to_metadata(),
        "run_name": settings.run_name,
        "total_timesteps": settings.total_timesteps,
        "num_envs": settings.num_envs,
        "eval_steps": settings.eval_steps,
        "seed": settings.seed,
        "check_env": settings.check_env,
        "normalize_actions": settings.normalize_actions,
        "action_interface": settings.action_interface,
        "rpm_delta_scale": settings.rpm_delta_scale,
        "include_dynamics_observation": settings.include_dynamics_observation,
        "include_previous_action": settings.include_previous_action,
        "termination_limits": termination_limits.to_dict(),
        "diagnostic_limits": diagnostic_limits.to_dict(),
        "ppo": settings.ppo_config.to_dict(),
        "wandb_mode": settings.wandb_mode,
        "wandb_project": settings.wandb_project,
        "wandb_entity": settings.wandb_entity,
        "wandb_group": settings.wandb_group,
        "wandb_name": settings.wandb_name,
        "wandb_tags": list(settings.wandb_tags),
        "run_metadata": dict(settings.run_metadata),
    }


def _to_yaml(payload: dict[str, Any]) -> str:
    """Serialize a small config snapshot payload to YAML."""
    import yaml  # noqa: PLC0415

    return yaml.safe_dump(payload, sort_keys=False)


def _manifest_relative_base(settings: PPOTrackingSmokeSettings, run_name: str) -> Path:
    """Return the manifest-relative base for direct runs or curriculum stages."""
    if settings.artifact_root is not None:
        return settings.artifact_root.expanduser().resolve(strict=False).parent
    return utils.artifacts.get_run_dir(run_name)


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


def _diagnostic_artifact_paths(training_run_name: str, metrics: dict[str, Any]) -> dict[str, Path]:
    """Return diagnostic JSON paths that should be preserved as W&B artifacts."""
    artifact_keys = {
        "failure_report": "failure_report_path",
        "curriculum_feedback": "curriculum_feedback_path",
        "episode_summaries": "episode_summaries_path",
        "evaluation_trace": "evaluation_trace_path",
    }
    paths: dict[str, Path] = {}
    for artifact_label, metric_key in artifact_keys.items():
        metric_value = metrics.get(metric_key)
        if isinstance(metric_value, str) and metric_value:
            paths[f"{training_run_name}_{artifact_label}"] = Path(metric_value)
    return paths


def _run_metadata_manifest_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    """Select caller-owned identity metadata for top-level manifests."""
    metadata_keys = (
        "experiment_id",
        "source_config_path",
        "curriculum_run_name",
        "curriculum_stage_index",
        "curriculum_stage_name",
        "curriculum_stage_count",
        "curriculum_stage_run_name",
        "previous_stage_model_path",
        "stage_budget_profile",
        "requested_stage_budget_profile",
        "selected_stage_budget_profile",
        "stage_total_timesteps",
        "cumulative_budget_timesteps",
        "cumulative_llm_budget_timesteps",
        "llm_budget_cap_timesteps",
        "budget_was_clipped",
        "budget_fallback_reason",
        "budget_rationale",
        "proposal_type",
        "original_proposal",
        "accepted_task",
        "stage_display_name",
        "proposal_fallback_used",
        "proposal_failure_reason",
        "task_distribution_reference",
        "resolved_task",
        "resolved_task_shape",
        "resolved_task_sample_metadata",
        "llm_provider",
        "llm_model",
    )
    return {key: metrics.get(key) for key in metadata_keys if key in metrics}


def _diagnostic_manifest_fields(metrics: dict[str, Any]) -> dict[str, Any]:
    """Select diagnostic fields that should be duplicated into the manifest."""
    diagnostic_keys = (
        "action_mean",
        "action_std",
        "action_min",
        "action_max",
        "action_saturation_fraction",
        "real_action_mean",
        "real_action_std",
        "real_action_min",
        "real_action_max",
        "real_action_saturation_fraction",
        "actions_normalized",
        "action_interface",
        "ppo_action_dim",
        "real_action_type",
        "real_action_space_bounds",
        "rpm_delta_scale",
        "include_dynamics_observation",
        "include_previous_action",
        "observation_dim",
        "observation_components",
        "policy_kwargs",
        "direct_control_limitations",
        "termination_limits_mode",
        "termination_limits",
        "diagnostic_limits",
        "base_truncation_policy",
        "terminate_on_base_truncation",
        "evaluation_termination_limits_mode",
        "evaluation_termination_limits",
        "evaluation_diagnostic_limits",
        "strict_limit_violation_count",
        "strict_limit_violation_causes",
        "base_truncation_causes",
        "project_truncation_causes",
        "direct_rpm_clipping_fraction",
        "direct_rpm_saturation_fraction",
        "mean_abs_x_error",
        "mean_abs_y_error",
        "mean_abs_z_error",
        "final_abs_x_error",
        "final_abs_y_error",
        "final_abs_z_error",
        "reference_xy_span_m",
        "actual_xy_span_m",
        "xy_tracking_ratio",
        "eval_terminated_count",
        "eval_truncated_count",
        "diagnostics_dir",
        "evaluation_trace_path",
        "episode_summaries_path",
        "failure_report_path",
        "curriculum_feedback_path",
        "failure_primary_mode",
        "failure_modes",
        "failure_overall_status",
        "curriculum_readiness_level",
        "curriculum_recommended_next_tasks",
        "curriculum_avoid_next_tasks",
    )
    return {key: metrics[key] for key in diagnostic_keys if key in metrics}


def _settings_ppo_action_dim(settings: PPOTrackingSmokeSettings) -> int:
    """Return the flattened PPO action dimension implied by settings."""
    if settings.action_interface == envs.actions.ActionInterface.DIRECT_RPM.value:
        return 4
    return 3


def _settings_real_action_type(settings: PPOTrackingSmokeSettings) -> str:
    """Return the real action type implied by settings."""
    if settings.action_interface == envs.actions.ActionInterface.DIRECT_RPM.value:
        return "motor_rpm"
    return "pid_target_position"


def _settings_real_action_space_bounds(settings: PPOTrackingSmokeSettings) -> dict[str, Any]:
    """Return static real-action bounds when they are known before env construction."""
    if settings.action_interface == envs.actions.ActionInterface.DIRECT_RPM.value:
        return {"low": None, "high": None, "units": "rpm"}
    return {"low": None, "high": None, "units": "meters"}


def _settings_observation_components(settings: PPOTrackingSmokeSettings) -> list[dict[str, int | str]]:
    """Return observation-component metadata implied by settings."""
    components: list[dict[str, int | str]] = [
        {"name": "current_position", "dim": 3},
        {"name": "reference_position", "dim": 3},
        {"name": "position_error", "dim": 3},
        {"name": "trajectory_progress", "dim": 1},
    ]
    if settings.include_dynamics_observation:
        components.extend(
            [
                {"name": "linear_velocity", "dim": 3},
                {"name": "attitude_rpy", "dim": 3},
                {"name": "angular_velocity", "dim": 3},
            ]
        )
    if settings.include_previous_action:
        components.append({"name": "previous_action", "dim": _settings_ppo_action_dim(settings)})
    return components


def _settings_observation_dim(settings: PPOTrackingSmokeSettings) -> int:
    """Return the observation dimension implied by settings."""
    return int(sum(int(component["dim"]) for component in _settings_observation_components(settings)))


def _wandb_config(
    settings: PPOTrackingSmokeSettings,
    run_name: str,
    model_path: Path,
    metrics_path: Path,
    manifest_path: Path,
    logs_dir: Path,
    diagnostics_dir: Path,
    selected_task_index: int,
    task: dict[str, Any],
    task_distribution_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a compact W&B config payload for PPO smoke training."""
    termination_limits = _settings_termination_limits(settings)
    diagnostic_limits = _settings_diagnostic_limits(settings)
    return {
        "run_type": _metadata_text(settings.run_metadata, "run_type", "training"),
        "run_kind": _metadata_text(settings.run_metadata, "run_kind", "direct_ppo"),
        "curriculum_kind": settings.run_metadata.get("curriculum_kind"),
        "run_name": run_name,
        "training_run_name": run_name,
        "training_config_path": str(settings.training_config_path) if settings.training_config_path is not None else None,
        "source_config_path": str(settings.training_config_path) if settings.training_config_path is not None else None,
        "task_config_path": str(settings.task_config_path),
        "task_index": selected_task_index,
        "task_shape": str(task.get("shape", "unknown")),
        "task_shape_requested": settings.task_shape,
        **_task_show_metadata(task),
        **dict(task_distribution_metadata or {}),
        "total_timesteps": settings.total_timesteps,
        "num_envs": settings.num_envs,
        "action_interface": settings.action_interface,
        "ppo_action_dim": _settings_ppo_action_dim(settings),
        "real_action_type": _settings_real_action_type(settings),
        "real_action_space_bounds": _settings_real_action_space_bounds(settings),
        "rpm_delta_scale": settings.rpm_delta_scale if settings.action_interface == envs.actions.ActionInterface.DIRECT_RPM.value else None,
        "include_dynamics_observation": settings.include_dynamics_observation,
        "include_previous_action": settings.include_previous_action,
        "observation_dim": _settings_observation_dim(settings),
        "observation_components": _settings_observation_components(settings),
        "direct_control_limitations": envs.actions.direct_control_limitations(settings.action_interface),
        "termination_limits_mode": termination_limits.mode,
        "termination_limits": termination_limits.to_dict(),
        "diagnostic_limits": diagnostic_limits.to_dict(),
        "base_truncation_policy": termination_limits.base_truncation_policy,
        "terminate_on_base_truncation": termination_limits.terminate_on_base_truncation,
        "vec_env_type": _vec_env_type(settings.num_envs),
        "vec_monitor_enabled": VEC_MONITOR_ENABLED,
        "effective_rollout_steps": settings.ppo_config.effective_rollout_steps(settings.num_envs),
        "ppo": settings.ppo_config.to_dict(),
        "policy_kwargs": settings.ppo_config.to_dict().get("policy_kwargs"),
        **_net_arch_metadata(settings),
        "ppo_profile": _ppo_profile(settings),
        "eval_steps": settings.eval_steps,
        "seed": settings.seed,
        "normalize_actions": settings.normalize_actions,
        "initial_model_path": str(settings.initial_model_path) if settings.initial_model_path is not None else None,
        "model_transfer_enabled": settings.initial_model_path is not None,
        "model_transfer_source": str(settings.initial_model_path) if settings.initial_model_path is not None else None,
        "run_metadata": dict(settings.run_metadata),
        **dict(settings.run_metadata),
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "logs_dir": str(logs_dir),
        "diagnostics_dir": str(diagnostics_dir),
    }


def _require_training_dependencies(dependencies: dict[str, bool]) -> None:
    """Raise if a dependency required for real PPO training is unavailable."""
    required = ("stable_baselines3", "gymnasium", "gym_pybullet_drones", "torch")
    missing = [name for name in required if not dependencies.get(name, False)]
    if missing:
        message = f"PPO smoke training requires missing dependencies: {', '.join(sorted(missing))}"
        raise RuntimeError(message)


def _check_tracking_env(
    task: dict[str, Any],
    normalize_actions: bool = DEFAULT_NORMALIZE_ACTIONS,
    action_interface: str = DEFAULT_ACTION_INTERFACE,
    rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE,
    include_dynamics_observation: bool = DEFAULT_INCLUDE_DYNAMICS_OBSERVATION,
    include_previous_action: bool = DEFAULT_INCLUDE_PREVIOUS_ACTION,
    termination_limits: envs.termination.TerminationLimitConfig | Mapping[str, Any] | str | None = None,
    diagnostic_limits: envs.termination.DiagnosticLimitConfig | Mapping[str, Any] | str | None = None,
) -> tuple[str, ...]:
    """Run Stable-Baselines3's environment checker and return captured warnings."""
    try:
        from stable_baselines3.common.env_checker import check_env  # noqa: PLC0415
    except ImportError as exc:
        return (f"stable_baselines3 env checker unavailable: {exc}",)

    real_checker_env = envs.tracking_env.make_trajectory_tracking_env(
        task,
        gui=False,
        record=False,
        action_interface=action_interface,
        rpm_delta_scale=rpm_delta_scale,
        include_dynamics_observation=include_dynamics_observation,
        include_previous_action=include_previous_action,
        termination_limits=termination_limits,
        diagnostic_limits=diagnostic_limits,
    )
    checker_env = _ppo_training_env(real_checker_env, normalize_actions=normalize_actions, action_interface=action_interface)
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


def _evaluate_model(model: Any, tracking_env: Any, settings: PPOTrackingSmokeSettings) -> dict[str, Any]:
    """Evaluate a trained model for a deterministic rollout with tracking diagnostics."""
    task_shape = str(getattr(getattr(tracking_env, "reference", None), "shape", settings.task_shape or "unknown"))
    diagnostics = evaluation.diagnostics.collect_policy_evaluation_diagnostics(
        model=model,
        tracking_env=tracking_env,
        eval_steps=settings.eval_steps,
        seed=settings.seed,
        training_run_name=_run_name(settings, task_shape),
        task_shape=task_shape,
        total_timesteps=settings.total_timesteps,
    )
    return diagnostics.metrics


def _tracking_error_metrics(positions: list[np.ndarray], reference_positions: list[np.ndarray]) -> dict[str, float]:
    """Return per-axis absolute tracking-error diagnostics."""
    if not positions or not reference_positions:
        return {
            "mean_abs_x_error": 0.0,
            "mean_abs_y_error": 0.0,
            "mean_abs_z_error": 0.0,
            "final_abs_x_error": 0.0,
            "final_abs_y_error": 0.0,
            "final_abs_z_error": 0.0,
        }
    position_array = np.asarray(positions, dtype=float).reshape(len(positions), -1)
    reference_array = np.asarray(reference_positions, dtype=float).reshape(len(reference_positions), -1)
    absolute_errors = np.abs(position_array[:, :3] - reference_array[:, :3])
    mean_errors = np.mean(absolute_errors, axis=0)
    final_errors = absolute_errors[-1]
    return {
        "mean_abs_x_error": float(mean_errors[0]),
        "mean_abs_y_error": float(mean_errors[1]),
        "mean_abs_z_error": float(mean_errors[2]),
        "final_abs_x_error": float(final_errors[0]),
        "final_abs_y_error": float(final_errors[1]),
        "final_abs_z_error": float(final_errors[2]),
    }


def _action_distribution_metrics(actions: list[np.ndarray], action_space: Any) -> dict[str, list[float]]:
    """Return per-dimension action distribution and saturation diagnostics."""
    if not actions:
        return {
            "action_mean": [],
            "action_std": [],
            "action_min": [],
            "action_max": [],
            "action_saturation_fraction": [],
        }
    action_array = np.asarray(actions, dtype=float).reshape(len(actions), -1)
    low = np.asarray(getattr(action_space, "low", []), dtype=float).reshape(-1)
    high = np.asarray(getattr(action_space, "high", []), dtype=float).reshape(-1)
    if low.size != action_array.shape[1] or high.size != action_array.shape[1]:
        saturation_fraction = np.zeros(action_array.shape[1], dtype=float)
    else:
        near_low = np.isclose(action_array, low, atol=_ACTION_SATURATION_TOLERANCE, rtol=0.0)
        near_high = np.isclose(action_array, high, atol=_ACTION_SATURATION_TOLERANCE, rtol=0.0)
        saturation_fraction = np.mean(np.logical_or(near_low, near_high), axis=0)
    return {
        "action_mean": [float(value) for value in np.mean(action_array, axis=0)],
        "action_std": [float(value) for value in np.std(action_array, axis=0)],
        "action_min": [float(value) for value in np.min(action_array, axis=0)],
        "action_max": [float(value) for value in np.max(action_array, axis=0)],
        "action_saturation_fraction": [float(value) for value in saturation_fraction],
    }


def _xy_tracking_ratio(actual_xy_span_m: float, reference_xy_span_m: float) -> float | None:
    """Return actual/reference XY span ratio when the reference span is nonzero."""
    if reference_xy_span_m <= _XY_TRACKING_RATIO_MIN_REFERENCE_SPAN_M:
        return None
    return float(actual_xy_span_m / reference_xy_span_m)


def run_liftoff_diagnostics(
    task: dict[str, Any],
    max_steps: int = DEFAULT_LIFTOFF_DIAGNOSTIC_STEPS,
    seed: int = DEFAULT_SEED,
    model: Any | None = None,
    include_simple_policies: bool = True,
    action_interface: str = DEFAULT_ACTION_INTERFACE,
    rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE,
    include_dynamics_observation: bool = DEFAULT_INCLUDE_DYNAMICS_OBSERVATION,
    include_previous_action: bool = DEFAULT_INCLUDE_PREVIOUS_ACTION,
    termination_limits: envs.termination.TerminationLimitConfig | Mapping[str, Any] | str | None = None,
    diagnostic_limits: envs.termination.DiagnosticLimitConfig | Mapping[str, Any] | str | None = None,
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
    action_interface
        Explicit action interface, either ``pid_position`` or ``direct_rpm``.
    rpm_delta_scale
        Fractional RPM delta around hover used by ``direct_rpm``.
    include_dynamics_observation
        Whether observations append velocity, attitude, and angular velocity.
    include_previous_action
        Whether observations append the previous PPO-facing action.
    termination_limits
        Optional hard episode-control safety limits.
    diagnostic_limits
        Optional strict diagnostic thresholds.

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
        diagnostics["zero_action"] = _run_liftoff_rollout(
            diagnostic_task,
            "zero_action",
            _zero_action,
            max_steps=max_steps,
            seed=seed,
            action_interface=action_interface,
            rpm_delta_scale=rpm_delta_scale,
            include_dynamics_observation=include_dynamics_observation,
            include_previous_action=include_previous_action,
            termination_limits=termination_limits,
            diagnostic_limits=diagnostic_limits,
        )
        diagnostics["sampled_action"] = _run_liftoff_rollout(
            diagnostic_task,
            "sampled_action",
            _sampled_action,
            max_steps=max_steps,
            seed=seed,
            action_interface=action_interface,
            rpm_delta_scale=rpm_delta_scale,
            include_dynamics_observation=include_dynamics_observation,
            include_previous_action=include_previous_action,
            termination_limits=termination_limits,
            diagnostic_limits=diagnostic_limits,
        )
        diagnostics["middle_action"] = _run_liftoff_rollout(
            diagnostic_task,
            "middle_action",
            _middle_action,
            max_steps=max_steps,
            seed=seed,
            action_interface=action_interface,
            rpm_delta_scale=rpm_delta_scale,
            include_dynamics_observation=include_dynamics_observation,
            include_previous_action=include_previous_action,
            termination_limits=termination_limits,
            diagnostic_limits=diagnostic_limits,
        )
        diagnostics["high_action"] = _run_liftoff_rollout(
            diagnostic_task,
            "high_action",
            _high_action,
            max_steps=max_steps,
            seed=seed,
            action_interface=action_interface,
            rpm_delta_scale=rpm_delta_scale,
            include_dynamics_observation=include_dynamics_observation,
            include_previous_action=include_previous_action,
            termination_limits=termination_limits,
            diagnostic_limits=diagnostic_limits,
        )
    if model is not None:
        diagnostics["trained_policy"] = _run_liftoff_rollout(
            diagnostic_task,
            "trained_policy",
            lambda _env, observation, _step: model.predict(observation, deterministic=True)[0],
            max_steps=max_steps,
            seed=seed,
            action_interface=action_interface,
            rpm_delta_scale=rpm_delta_scale,
            include_dynamics_observation=include_dynamics_observation,
            include_previous_action=include_previous_action,
            termination_limits=termination_limits,
            diagnostic_limits=diagnostic_limits,
        )
    return diagnostics


def _task_with_minimum_reference_samples(task: dict[str, Any], required_steps: int) -> tuple[dict[str, Any], int, tuple[str, ...]]:
    """Return a copied task with enough reference samples for diagnostics."""
    reference = envs.task_adapter.make_task_reference(task)
    reference_samples = int(reference.positions.shape[0])
    required_samples = required_steps + 1
    if reference_samples >= required_samples:
        return dict(task), reference_samples, ()

    sample_rate_value = task.get("sample_rate_hz")
    if sample_rate_value is None:
        warning = "diagnostic task has too few reference samples and cannot be extended because sample_rate_hz is missing"
        return dict(task), reference_samples, (warning,)

    try:
        sample_rate_hz = float(sample_rate_value)
    except (TypeError, ValueError):
        warning = "diagnostic task has non-numeric sample_rate_hz and cannot be safely extended"
        return dict(task), reference_samples, (warning,)
    if sample_rate_hz <= 0.0:
        warning = "diagnostic task has non-positive sample_rate_hz and cannot be safely extended"
        return dict(task), reference_samples, (warning,)

    extended_task, extension_warning = _duration_extended_diagnostic_task(
        task=task,
        sample_rate_hz=sample_rate_hz,
        required_samples=required_samples,
    )
    if extended_task is None:
        return dict(task), reference_samples, (extension_warning,)

    try:
        extended_reference = envs.task_adapter.make_task_reference(extended_task)
    except ValueError as exc:
        warning = f"{extension_warning}; duration extension failed validation: {exc}; using original diagnostic task"
        return dict(task), reference_samples, (warning,)

    extended_samples = int(extended_reference.positions.shape[0])
    warnings = [extension_warning]
    if extended_samples < required_samples:
        warnings.append("extended diagnostic task still has too few reference samples; rollout may end early")
    return extended_task, extended_samples, tuple(warnings)


def _duration_extended_diagnostic_task(
    task: dict[str, Any],
    sample_rate_hz: float,
    required_samples: int,
) -> tuple[dict[str, Any] | None, str]:
    """Return a diagnostic task extended in time without changing sample spacing."""
    required_duration_sec = (required_samples - 1) / sample_rate_hz
    duration_value = task.get("duration_sec")
    if duration_value is not None:
        try:
            duration_sec = float(duration_value)
        except (TypeError, ValueError):
            return None, "diagnostic task has non-numeric duration_sec and cannot be safely extended"
        if duration_sec <= 0.0:
            return None, "diagnostic task has non-positive duration_sec and cannot be safely extended"

        extended_duration_sec = max(duration_sec, required_duration_sec)
        extended_task = dict(task)
        # Preserve sample spacing; increasing sample_rate_hz can violate acceleration checks.
        extended_task["duration_sec"] = float(extended_duration_sec)
        warning = (
            f"extended diagnostic task duration_sec from {duration_sec} to {extended_duration_sec} "
            f"at sample_rate_hz {sample_rate_hz} for {required_samples - 1} steps"
        )
        return extended_task, warning

    hold_duration_value = task.get("hold_duration_sec")
    move_duration_value = task.get("move_duration_sec")
    if hold_duration_value is None or move_duration_value is None:
        warning = (
            "diagnostic task has too few reference samples and cannot be extended because "
            "duration_sec or hold_duration_sec/move_duration_sec are missing"
        )
        return None, warning

    try:
        hold_duration_sec = float(hold_duration_value)
        move_duration_sec = float(move_duration_value)
    except (TypeError, ValueError):
        warning = "diagnostic task has non-numeric hold_duration_sec or move_duration_sec and cannot be safely extended"
        return None, warning
    if hold_duration_sec <= 0.0 or move_duration_sec <= 0.0:
        warning = "diagnostic task has non-positive hold_duration_sec or move_duration_sec and cannot be safely extended"
        return None, warning

    extended_move_duration_sec = max(move_duration_sec, required_duration_sec - hold_duration_sec)
    extended_task = dict(task)
    extended_task["move_duration_sec"] = float(extended_move_duration_sec)
    warning = (
        f"extended diagnostic task move_duration_sec from {move_duration_sec} to {extended_move_duration_sec} "
        f"at sample_rate_hz {sample_rate_hz} for {required_samples - 1} steps"
    )
    return extended_task, warning


def _run_liftoff_rollout(
    task: dict[str, Any],
    name: str,
    action_factory: Any,
    max_steps: int,
    seed: int,
    action_interface: str = DEFAULT_ACTION_INTERFACE,
    rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE,
    include_dynamics_observation: bool = DEFAULT_INCLUDE_DYNAMICS_OBSERVATION,
    include_previous_action: bool = DEFAULT_INCLUDE_PREVIOUS_ACTION,
    termination_limits: envs.termination.TerminationLimitConfig | Mapping[str, Any] | str | None = None,
    diagnostic_limits: envs.termination.DiagnosticLimitConfig | Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Run one diagnostic rollout and return bounds and termination metadata."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        task,
        gui=False,
        record=False,
        max_steps=max_steps,
        action_interface=action_interface,
        rpm_delta_scale=rpm_delta_scale,
        include_dynamics_observation=include_dynamics_observation,
        include_previous_action=include_previous_action,
        termination_limits=termination_limits,
        diagnostic_limits=diagnostic_limits,
    )
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
            "project_truncation_causes": list(final_info.get("project_truncation_causes", [])),
            "strict_limit_violations": list(final_info.get("strict_limit_violations", [])),
            "strict_limit_violation_count": int(final_info.get("strict_limit_violation_count", 0)),
            "termination_limits_mode": str(final_info.get("termination_limits_mode", "")),
            "base_truncation_policy": str(final_info.get("base_truncation_policy", "")),
            "terminate_on_base_truncation": bool(final_info.get("terminate_on_base_truncation", True)),
            "recovery_allowed_after_limit_violation": bool(final_info.get("recovery_allowed_after_limit_violation", False)),
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
            "action_interface": str(final_info.get("action_interface", action_interface)),
            "real_action_type": str(final_info.get("real_action_type", "")),
            "ppo_action_dim": int(final_info.get("ppo_action_dim", 0)),
            "hover_rpm": final_info.get("hover_rpm"),
            "rpm_delta_scale": final_info.get("rpm_delta_scale"),
            "include_dynamics_observation": bool(final_info.get("include_dynamics_observation", include_dynamics_observation)),
            "include_previous_action": bool(final_info.get("include_previous_action", include_previous_action)),
            "observation_dim": int(final_info.get("observation_dim", 0)),
            "observation_components": [dict(component) for component in final_info.get("observation_components", [])],
            "real_motor_rpms": _array_to_jsonable(final_info.get("real_motor_rpms", [])),
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
    observation_space = tracking_env.observation_space
    real_action_space = getattr(tracking_env, "real_action_space", action_space)
    sample = action_space.sample()
    tracking_core = getattr(tracking_env, "unwrapped", tracking_env)
    base_env = getattr(tracking_core, "base_env", None)
    action_type = getattr(base_env, "ACT_TYPE", None)
    action_type_value = _enum_value(action_type)
    action_interface = str(getattr(tracking_env, "action_interface", getattr(tracking_core, "action_interface", DEFAULT_ACTION_INTERFACE)))
    action_interface = envs.actions.parse_action_interface(action_interface).value
    direct_rpm = action_interface == envs.actions.ActionInterface.DIRECT_RPM.value
    if direct_rpm:
        rpm_min = 0.0
        rpm_max = float(getattr(base_env, "MAX_RPM", 0.0)) if base_env is not None else 0.0
        hover_rpm = float(getattr(base_env, "HOVER_RPM", 0.0)) if base_env is not None else 0.0
        rpm_delta_scale = float(getattr(tracking_core, "rpm_delta_scale", envs.actions.DEFAULT_RPM_DELTA_SCALE))
        real_action_space_low = np.full(getattr(action_space, "shape", ()), rpm_min, dtype=float)
        real_action_space_high = np.full(getattr(action_space, "shape", ()), rpm_max, dtype=float)
        command_low = envs.tracking_env.normalized_direct_rpm_to_motor_rpms(
            np.full(getattr(action_space, "shape", ()), -1.0, dtype=np.float32),
            hover_rpm=hover_rpm,
            rpm_delta_scale=rpm_delta_scale,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
        )
        command_high = envs.tracking_env.normalized_direct_rpm_to_motor_rpms(
            np.full(getattr(action_space, "shape", ()), 1.0, dtype=np.float32),
            hover_rpm=hover_rpm,
            rpm_delta_scale=rpm_delta_scale,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
        )
        real_action_type = "motor_rpm"
    else:
        rpm_delta_scale = None
        hover_rpm = None
        rpm_min = None
        rpm_max = None
        command_low = None
        command_high = None
        real_action_space_low = np.asarray(getattr(real_action_space, "low", []), dtype=float)
        real_action_space_high = np.asarray(getattr(real_action_space, "high", []), dtype=float)
        real_action_type = "pid_target_position"
    return {
        "action_interface": action_interface,
        "action_space": str(action_space),
        "action_space_shape": _shape_list(getattr(action_space, "shape", ())),
        "action_space_dtype": str(getattr(action_space, "dtype", "")),
        "action_space_low": _array_to_jsonable(getattr(action_space, "low", [])),
        "action_space_high": _array_to_jsonable(getattr(action_space, "high", [])),
        "actions_normalized": bool(direct_rpm or getattr(tracking_env, "real_action_space", None) is not None),
        "ppo_action_dim": int(np.prod(tuple(getattr(action_space, "shape", ())))),
        "ppo_action_space": str(action_space),
        "real_action_type": real_action_type,
        "real_action_space": str(real_action_space),
        "real_action_space_low": _array_to_jsonable(real_action_space_low),
        "real_action_space_high": _array_to_jsonable(real_action_space_high),
        "real_action_space_bounds": {
            "low": _array_to_jsonable(real_action_space_low),
            "high": _array_to_jsonable(real_action_space_high),
            "units": "rpm" if direct_rpm else "meters",
        },
        "hover_rpm": hover_rpm,
        "rpm_delta_scale": rpm_delta_scale,
        "rpm_min": rpm_min,
        "rpm_max": rpm_max,
        "rpm_command_space_low": None if command_low is None else _array_to_jsonable(command_low),
        "rpm_command_space_high": None if command_high is None else _array_to_jsonable(command_high),
        "include_dynamics_observation": bool(getattr(tracking_core, "include_dynamics_observation", False)),
        "include_previous_action": bool(getattr(tracking_core, "include_previous_action", False)),
        "observation_space": str(observation_space),
        "observation_space_shape": _shape_list(getattr(observation_space, "shape", ())),
        "observation_dim": int(np.prod(tuple(getattr(observation_space, "shape", ())))),
        "observation_components": [dict(component) for component in getattr(tracking_core, "observation_components", [])],
        "direct_control_limitations": envs.actions.direct_control_limitations(action_interface),
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
    if action_metadata.get("action_interface") == envs.actions.ActionInterface.DIRECT_RPM.value and not bool(
        action_metadata.get("include_dynamics_observation", False)
    ):
        warnings.append("direct_rpm is under-observed without include_dynamics_observation=true")
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
    if action_type_value == "rpm":
        return (
            "four-dimensional normalized per-motor command shaped (num_drones, 4); "
            "tracking wrapper maps each motor command to clipped real RPMs before PyBullet physics"
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
