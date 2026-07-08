"""
===============================================================================
experiments_training_smoke.py
===============================================================================
Run tiny deterministic MVP training-smoke loops for validated trajectory tasks.

Responsibilities:
  - Load smoke training settings and configured trajectory tasks
  - Run a bounded deterministic baseline loop against validated references
  - Write small JSON metrics artifacts under approved results directories

Design principles:
  - Keep defaults fast, deterministic, headless, and simulator-independent
  - Report optional RL dependency availability without requiring it for smoke tests

Boundaries:
  - Full RL training and PyBullet environment wrappers belong in later experiments
  - Rollout evaluation and plotting belong in evaluation modules
===============================================================================

"""

from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src import envs, utils
from src.experiments import experiments_config as config_loader

DEFAULT_TRAINING_CONFIG_PATH = Path("configs/smoke/training_smoke.yaml")
DEFAULT_TASK_CONFIG_PATH = Path("configs/smoke/trajectory_validation.yaml")
DEFAULT_OUTPUT_FILENAME = "training_smoke_metrics.json"
DEFAULT_INITIAL_ERROR_M = 0.2
DEFAULT_MAX_STEPS = 16


@dataclass(frozen=True)
class TrainingSmokeSettings:
    """
    Settings for a tiny deterministic training-smoke run.

    Parameters
    ----------
    task_config_path
        YAML config containing a top-level list of trajectory tasks.
    task_index
        Zero-based task index selected from the task config.
    max_steps
        Maximum number of synthetic baseline steps to execute.
    output_dir
        Directory where the metrics JSON artifact is written.
    output_filename
        Metrics JSON filename within ``output_dir``.
    initial_error_m
        Initial deterministic XYZ offset magnitude used by the baseline.
    mode
        Training smoke mode. The MVP implementation supports ``"deterministic"``.

    """

    task_config_path: Path = DEFAULT_TASK_CONFIG_PATH
    task_index: int = 0
    max_steps: int = DEFAULT_MAX_STEPS
    output_dir: Path | None = None
    output_filename: str = DEFAULT_OUTPUT_FILENAME
    initial_error_m: float = DEFAULT_INITIAL_ERROR_M
    mode: str = "deterministic"

    def __post_init__(self) -> None:
        """Validate smoke-run settings."""
        if self.task_index < 0:
            message = "task_index must be nonnegative"
            raise ValueError(message)
        if self.max_steps <= 0:
            message = "max_steps must be positive"
            raise ValueError(message)
        if not np.isfinite(self.initial_error_m) or self.initial_error_m < 0.0:
            message = "initial_error_m must be finite and nonnegative"
            raise ValueError(message)
        if self.mode != "deterministic":
            message = "only deterministic smoke mode is supported by the MVP baseline"
            raise ValueError(message)
        if not self.output_filename.endswith(".json"):
            message = "output_filename must end with .json"
            raise ValueError(message)


@dataclass(frozen=True)
class TrainingSmokeResult:
    """
    Summary returned by a training-smoke run.

    Parameters
    ----------
    output_path
        Path to the written metrics JSON artifact.
    metrics
        JSON-serializable metrics proving the smoke loop executed.
    warnings
        Nonfatal warnings, including optional dependency fallback notes.

    """

    output_path: str
    metrics: dict[str, Any]
    warnings: tuple[str, ...] = ()


def load_training_smoke_settings(path: str | Path) -> TrainingSmokeSettings:
    """
    Load smoke training settings from YAML.

    Parameters
    ----------
    path
        Path to a YAML settings file.

    Returns
    -------
    TrainingSmokeSettings
        Validated settings with paths expanded as ``Path`` objects.

    """
    config = config_loader.load_experiment_config(path)
    return _settings_from_mapping(config)


def default_output_dir() -> Path:
    """Return the default MVP smoke metrics directory under the run layout."""
    return utils.artifacts.get_training_metrics_dir("mvp_smoke")


def detect_optional_training_dependencies() -> dict[str, bool]:
    """Return availability flags for optional runtime training dependencies."""
    return {
        "stable_baselines3": importlib.util.find_spec("stable_baselines3") is not None,
        "gymnasium": importlib.util.find_spec("gymnasium") is not None,
        "gym_pybullet_drones": importlib.util.find_spec("gym_pybullet_drones") is not None,
        "torch": importlib.util.find_spec("torch") is not None,
    }


def run_training_smoke(settings: TrainingSmokeSettings | None = None) -> TrainingSmokeResult:
    """
    Run the deterministic MVP training-smoke loop and write metrics JSON.

    Parameters
    ----------
    settings
        Optional smoke-run settings. Defaults are used when omitted.

    Returns
    -------
    TrainingSmokeResult
        Output path, metrics, and nonfatal warnings.

    """
    active_settings = settings or TrainingSmokeSettings()
    task = _load_task(active_settings.task_config_path, active_settings.task_index)
    reference = envs.task_adapter.make_task_reference(task)
    step_count = min(active_settings.max_steps, int(reference.positions.shape[0]))
    if step_count <= 0:
        message = "selected task reference contains no samples"
        raise ValueError(message)

    rewards: list[float] = []
    errors: list[float] = []
    for step_index in range(step_count):
        progress = step_index / max(step_count - 1, 1)
        offset = np.array([active_settings.initial_error_m * (1.0 - progress), 0.0, 0.0], dtype=float)
        actual_position = reference.positions[step_index] + offset
        step = envs.tracking_reward.step_tracking_episode(
            reference=reference,
            actual_position=actual_position,
            step_index=step_index,
            action=offset,
            config=envs.tracking_reward.TrackingRewardConfig(max_steps=step_count),
        )
        rewards.append(step.reward)
        errors.append(step.position_error_m)

    dependencies = detect_optional_training_dependencies()
    warnings = _fallback_warnings(dependencies)
    metrics: dict[str, Any] = {
        "mode": active_settings.mode,
        "baseline": "deterministic_offset_decay",
        "task_config_path": str(active_settings.task_config_path),
        "task_index": active_settings.task_index,
        "task_shape": reference.shape,
        "validated": True,
        "step_count": step_count,
        "max_steps": active_settings.max_steps,
        "initial_error_m": active_settings.initial_error_m,
        "mean_reward": float(np.mean(rewards)),
        "final_reward": float(rewards[-1]),
        "mean_position_error_m": float(np.mean(errors)),
        "final_position_error_m": float(errors[-1]),
        "dependency_available": dependencies,
        "warnings": list(warnings),
    }
    output_path = _resolve_output_path(active_settings)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return TrainingSmokeResult(output_path=str(output_path), metrics=metrics, warnings=warnings)


def run_training_smoke_from_config(
    config_path: str | Path = DEFAULT_TRAINING_CONFIG_PATH,
    output_dir: str | Path | None = None,
    max_steps: int | None = None,
    task_index: int | None = None,
) -> TrainingSmokeResult:
    """
    Load settings, apply CLI-style overrides, and run the smoke loop.

    Parameters
    ----------
    config_path
        YAML settings path.
    output_dir
        Optional output directory override.
    max_steps
        Optional max-step override.
    task_index
        Optional task-index override.

    Returns
    -------
    TrainingSmokeResult
        Output path, metrics, and nonfatal warnings.

    """
    settings = load_training_smoke_settings(config_path)
    overridden = TrainingSmokeSettings(
        task_config_path=settings.task_config_path,
        task_index=settings.task_index if task_index is None else task_index,
        max_steps=settings.max_steps if max_steps is None else max_steps,
        output_dir=settings.output_dir if output_dir is None else Path(output_dir),
        output_filename=settings.output_filename,
        initial_error_m=settings.initial_error_m,
        mode=settings.mode,
    )
    return run_training_smoke(overridden)


def _settings_from_mapping(config: dict[str, Any]) -> TrainingSmokeSettings:
    """Build settings from a loaded YAML mapping."""
    output_dir_value = config.get("output_dir")
    return TrainingSmokeSettings(
        task_config_path=Path(config.get("task_config_path", DEFAULT_TASK_CONFIG_PATH)),
        task_index=int(config.get("task_index", 0)),
        max_steps=int(config.get("max_steps", DEFAULT_MAX_STEPS)),
        output_dir=Path(output_dir_value) if output_dir_value is not None else None,
        output_filename=str(config.get("output_filename", DEFAULT_OUTPUT_FILENAME)),
        initial_error_m=float(config.get("initial_error_m", DEFAULT_INITIAL_ERROR_M)),
        mode=str(config.get("mode", "deterministic")),
    )


def _load_task(task_config_path: Path, task_index: int) -> dict[str, Any]:
    """Load and return a copied task from a task config path."""
    config = config_loader.load_experiment_config(task_config_path)
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


def _resolve_output_path(settings: TrainingSmokeSettings) -> Path:
    """Resolve the metrics output path."""
    output_dir = settings.output_dir or default_output_dir()
    return output_dir.expanduser().resolve(strict=False) / settings.output_filename


def _fallback_warnings(dependencies: dict[str, bool]) -> tuple[str, ...]:
    """Return deterministic fallback warnings for the current MVP state."""
    if all(dependencies.values()):
        return ("deterministic fallback used because no trajectory-tracking Gym wrapper exists yet",)
    missing = sorted(name for name, available in dependencies.items() if not available)
    return (f"deterministic fallback used; missing optional dependencies: {', '.join(missing)}",)


__all__ = [
    "DEFAULT_TASK_CONFIG_PATH",
    "DEFAULT_TRAINING_CONFIG_PATH",
    "TrainingSmokeResult",
    "TrainingSmokeSettings",
    "default_output_dir",
    "detect_optional_training_dependencies",
    "load_training_smoke_settings",
    "run_training_smoke",
    "run_training_smoke_from_config",
]
