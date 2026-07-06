"""
===============================================================================
experiments_ppo_tracking.py
===============================================================================
Run tiny Stable-Baselines3 PPO smoke training on TrajectoryTrackingEnv.

Responsibilities:
  - Load a configured validated trajectory task for PPO smoke training
  - Verify the Gymnasium wrapper with Stable-Baselines3 when available
  - Train, save, and evaluate a tiny deterministic PPO model headlessly
  - Write compact JSON metrics under approved storage results directories

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

DEFAULT_PPO_TRACKING_CONFIG_PATH = Path("configs/smoke/ppo_tracking_smoke.yaml")
DEFAULT_TASK_CONFIG_PATH = Path("configs/smoke/trajectory_validation.yaml")
DEFAULT_TASK_INDEX = 2
DEFAULT_TOTAL_TIMESTEPS = 128
DEFAULT_EVAL_STEPS = 32
DEFAULT_SEED = 0
DEFAULT_MODEL_FILENAME = "ppo_tracking_smoke.zip"
DEFAULT_METRICS_FILENAME = "ppo_tracking_smoke_metrics.json"
_MIN_PPO_ROLLOUT_STEPS = 2
_MAX_PPO_ROLLOUT_STEPS = 32


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
    model_filename
        Trained model filename within ``model_dir``.
    metrics_filename
        Metrics JSON filename within ``output_dir``.
    check_env
        Whether to run the Stable-Baselines3 environment checker before training.

    """

    task_config_path: Path = DEFAULT_TASK_CONFIG_PATH
    task_index: int = DEFAULT_TASK_INDEX
    total_timesteps: int = DEFAULT_TOTAL_TIMESTEPS
    eval_steps: int = DEFAULT_EVAL_STEPS
    seed: int = DEFAULT_SEED
    output_dir: Path | None = None
    model_dir: Path | None = None
    model_filename: str = DEFAULT_MODEL_FILENAME
    metrics_filename: str = DEFAULT_METRICS_FILENAME
    check_env: bool = True

    def __post_init__(self) -> None:
        """Validate PPO smoke-run settings."""
        if self.task_index < 0:
            message = "task_index must be nonnegative"
            raise ValueError(message)
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
    metrics
        JSON-serializable metrics proving PPO trained and evaluated.
    warnings
        Nonfatal compatibility or checker warnings captured during the run.

    """

    model_path: str
    metrics_path: str
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
    """Return the default PPO tracking smoke output directory under storage results."""
    return utils.paths.get_results_root() / "ppo_tracking_smoke"


def default_model_dir() -> Path:
    """Return the default PPO tracking smoke model directory under storage models."""
    return utils.paths.get_models_root() / "ppo_tracking_smoke"


def detect_ppo_tracking_dependencies() -> dict[str, bool]:
    """Return availability flags for PPO trajectory-tracking runtime dependencies."""
    return {
        "stable_baselines3": importlib.util.find_spec("stable_baselines3") is not None,
        "gymnasium": importlib.util.find_spec("gymnasium") is not None,
        "gym_pybullet_drones": importlib.util.find_spec("gym_pybullet_drones") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }


def run_ppo_tracking_smoke(settings: PPOTrackingSmokeSettings | None = None) -> PPOTrackingSmokeResult:
    """
    Train and evaluate a tiny Stable-Baselines3 PPO model on TrajectoryTrackingEnv.

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
    _require_training_dependencies(dependencies)
    task = _load_task(active_settings.task_config_path, active_settings.task_index)

    warnings = list(_check_tracking_env(task) if active_settings.check_env else ())
    model_path = _resolve_model_path(active_settings)
    metrics_path = _resolve_metrics_path(active_settings)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    from stable_baselines3 import PPO  # noqa: PLC0415

    training_env = envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False)
    try:
        rollout_steps = _ppo_rollout_steps(active_settings.total_timesteps)
        model = PPO(
            "MlpPolicy",
            training_env,
            batch_size=rollout_steps,
            device="cpu",
            gamma=0.95,
            n_epochs=1,
            n_steps=rollout_steps,
            seed=active_settings.seed,
            verbose=0,
        )
        model.learn(total_timesteps=active_settings.total_timesteps, progress_bar=False)
        model.save(str(model_path))
        eval_metrics = _evaluate_model(model, training_env, active_settings)
    finally:
        training_env.close()

    metrics: dict[str, Any] = {
        "mode": "ppo_smoke",
        "task_config_path": str(active_settings.task_config_path),
        "task_index": active_settings.task_index,
        "task_shape": str(task.get("shape", "unknown")),
        "total_timesteps": active_settings.total_timesteps,
        "eval_steps": active_settings.eval_steps,
        "seed": active_settings.seed,
        "model_path": str(model_path),
        "metrics_path": str(metrics_path),
        "dependency_available": dependencies,
        "warnings": warnings,
        "trained": True,
        "env_checked": active_settings.check_env,
        **eval_metrics,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return PPOTrackingSmokeResult(model_path=str(model_path), metrics_path=str(metrics_path), metrics=metrics, warnings=tuple(warnings))


def run_ppo_tracking_smoke_from_config(
    config_path: str | Path = DEFAULT_PPO_TRACKING_CONFIG_PATH,
    task_index: int | None = None,
    total_timesteps: int | None = None,
    eval_steps: int | None = None,
    output_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
    seed: int | None = None,
) -> PPOTrackingSmokeResult:
    """
    Load settings, apply CLI-style overrides, and run PPO smoke training.

    Parameters
    ----------
    config_path
        YAML settings path.
    task_index
        Optional task-index override.
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

    Returns
    -------
    PPOTrackingSmokeResult
        Saved model path, metrics path, metrics payload, and nonfatal warnings.

    """
    settings = load_ppo_tracking_settings(config_path)
    overridden = PPOTrackingSmokeSettings(
        task_config_path=settings.task_config_path,
        task_index=settings.task_index if task_index is None else task_index,
        total_timesteps=settings.total_timesteps if total_timesteps is None else total_timesteps,
        eval_steps=settings.eval_steps if eval_steps is None else eval_steps,
        seed=settings.seed if seed is None else seed,
        output_dir=settings.output_dir if output_dir is None else Path(output_dir),
        model_dir=settings.model_dir if model_dir is None else Path(model_dir),
        model_filename=settings.model_filename,
        metrics_filename=settings.metrics_filename,
        check_env=settings.check_env,
    )
    return run_ppo_tracking_smoke(overridden)


def _settings_from_mapping(config: dict[str, Any]) -> PPOTrackingSmokeSettings:
    """Build settings from a loaded YAML mapping."""
    output_dir_value = config.get("output_dir")
    model_dir_value = config.get("model_dir")
    return PPOTrackingSmokeSettings(
        task_config_path=Path(config.get("task_config_path", DEFAULT_TASK_CONFIG_PATH)),
        task_index=int(config.get("task_index", DEFAULT_TASK_INDEX)),
        total_timesteps=int(config.get("total_timesteps", DEFAULT_TOTAL_TIMESTEPS)),
        eval_steps=int(config.get("eval_steps", DEFAULT_EVAL_STEPS)),
        seed=int(config.get("seed", DEFAULT_SEED)),
        output_dir=Path(output_dir_value) if output_dir_value is not None else None,
        model_dir=Path(model_dir_value) if model_dir_value is not None else None,
        model_filename=str(config.get("model_filename", DEFAULT_MODEL_FILENAME)),
        metrics_filename=str(config.get("metrics_filename", DEFAULT_METRICS_FILENAME)),
        check_env=bool(config.get("check_env", True)),
    )


def _load_task(task_config_path: Path, task_index: int) -> dict[str, Any]:
    """Load and return a copied task from a task config path."""
    config = experiments.config.load_experiment_config(task_config_path)
    tasks = config.get("tasks")
    if not isinstance(tasks, list):
        message = "task config must contain a top-level tasks list"
        raise ValueError(message)  # noqa: TRY004 - public config contract reports config errors as ValueError.
    if task_index >= len(tasks):
        message = "task_index is outside the configured task list"
        raise ValueError(message)
    task = tasks[task_index]
    if not isinstance(task, dict):
        message = "selected task must be a mapping"
        raise ValueError(message)  # noqa: TRY004 - public config contract reports config errors as ValueError.
    return dict(task)


def _resolve_model_path(settings: PPOTrackingSmokeSettings) -> Path:
    """Resolve the trained model output path."""
    return _resolve_directory(settings.model_dir, default_model_dir()) / settings.model_filename


def _resolve_metrics_path(settings: PPOTrackingSmokeSettings) -> Path:
    """Resolve the metrics output path."""
    return _resolve_directory(settings.output_dir, default_output_dir()) / settings.metrics_filename


def _resolve_directory(path: Path | None, default: Path) -> Path:
    """Resolve a configured directory or its storage-backed default."""
    directory = default if path is None else path
    return directory.expanduser().resolve(strict=False)


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


def _evaluate_model(model: Any, tracking_env: Any, settings: PPOTrackingSmokeSettings) -> dict[str, float | int]:
    """Evaluate a trained model for a tiny deterministic rollout."""
    observation, _ = tracking_env.reset(seed=settings.seed)
    rewards: list[float] = []
    errors: list[float] = []
    reset_count = 0
    for _ in range(settings.eval_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = tracking_env.step(action)
        rewards.append(float(reward))
        errors.append(float(info["position_error_m"]))
        if terminated or truncated:
            reset_count += 1
            observation, _ = tracking_env.reset(seed=settings.seed + reset_count)

    return {
        "actual_eval_steps": len(rewards),
        "eval_resets": reset_count,
        "mean_eval_reward": float(np.mean(rewards)),
        "final_eval_reward": float(rewards[-1]),
        "mean_position_error_m": float(np.mean(errors)),
        "final_position_error_m": float(errors[-1]),
    }


__all__ = [
    "DEFAULT_PPO_TRACKING_CONFIG_PATH",
    "DEFAULT_TASK_CONFIG_PATH",
    "PPOTrackingSmokeResult",
    "PPOTrackingSmokeSettings",
    "default_model_dir",
    "default_output_dir",
    "detect_ppo_tracking_dependencies",
    "load_ppo_tracking_settings",
    "run_ppo_tracking_smoke",
    "run_ppo_tracking_smoke_from_config",
]
