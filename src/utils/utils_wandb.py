"""
===============================================================================
utils_wandb.py
===============================================================================
Provide optional Weights & Biases tracking for bounded experiment smoke runs.

Responsibilities:
  - Resolve W&B auto/online/offline/disabled modes without storing secrets
  - Lazily initialize W&B with run-specific directories and non-secret metadata
  - Safely load WANDB_API_KEY from the environment or an optional home key file
  - Map final PPO diagnostics into grouped run summary fields

Design principles:
  - Never store or print secrets
  - Keep disabled mode a no-op for tests and explicit local opt-out
  - Fall back to offline tracking when auto mode lacks credentials

Boundaries:
  - Training modules decide when final metrics are ready
  - This module does not own Stable-Baselines3 callbacks or long-running tracking
===============================================================================

"""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

from . import utils_artifacts as artifacts

WANDB_MODE_AUTO = "auto"
WANDB_MODE_DISABLED = "disabled"
WANDB_MODE_OFFLINE = "offline"
WANDB_MODE_ONLINE = "online"
WANDB_MODES = (WANDB_MODE_AUTO, WANDB_MODE_ONLINE, WANDB_MODE_OFFLINE, WANDB_MODE_DISABLED)
DEFAULT_WANDB_PROJECT = "drone-rl-llm-curriculum"


@dataclass(frozen=True)
class WandbTrackingSettings:
    """
    Settings for optional W&B run tracking.

    Parameters
    ----------
    mode
        Tracking mode. ``auto`` uses online when credentials are available and offline otherwise.
    project
        W&B project name used when tracking is enabled.
    entity
        Optional W&B entity/team.
    group
        Optional run group.
    name
        Optional run name.
    tags
        Optional run tags.
    dir
        Directory where W&B local/offline files are written.

    """

    mode: str = WANDB_MODE_AUTO
    project: str = DEFAULT_WANDB_PROJECT
    entity: str | None = None
    group: str | None = None
    name: str | None = None
    tags: tuple[str, ...] = ()
    dir: Path | None = None

    def __post_init__(self) -> None:
        """Validate W&B settings."""
        if self.mode not in WANDB_MODES:
            message = f"wandb mode must be one of: {', '.join(WANDB_MODES)}"
            raise ValueError(message)
        if not self.project.strip():
            message = "wandb project must be non-empty"
            raise ValueError(message)


def default_wandb_dir(training_run_name: str) -> Path:
    """Return the W&B directory for an explicit training run name."""
    return artifacts.get_training_wandb_dir(training_run_name)


def resolve_wandb_mode(mode: str) -> str:
    """Resolve ``auto`` W&B mode to ``online`` or ``offline`` based on credentials."""
    if mode not in WANDB_MODES:
        message = f"wandb mode must be one of: {', '.join(WANDB_MODES)}"
        raise ValueError(message)
    if mode != WANDB_MODE_AUTO:
        return mode
    _load_wandb_api_key_from_home_file()
    return WANDB_MODE_ONLINE if os.environ.get("WANDB_API_KEY") else WANDB_MODE_OFFLINE


def parse_wandb_tags(value: str | Sequence[str] | None) -> tuple[str, ...]:
    """Parse comma-separated or sequence W&B tags into a clean tuple."""
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(tag.strip() for tag in value.split(",") if tag.strip())
    return tuple(str(tag).strip() for tag in value if str(tag).strip())


@contextmanager
def wandb_run(settings: WandbTrackingSettings, config: dict[str, Any]) -> Iterator[Any | None]:
    """
    Create an optional W&B run context.

    Parameters
    ----------
    settings
        W&B tracking settings.
    config
        JSON-serializable run configuration sent to W&B when tracking is enabled.

    Yields
    ------
    Any | None
        Active W&B run object, or ``None`` when tracking is disabled.

    Raises
    ------
    RuntimeError
        If W&B is requested but unavailable, or online mode lacks credentials.

    """
    run = start_wandb_run(settings=settings, config=config)
    try:
        yield run
    finally:
        if run is not None:
            run.finish()


def start_wandb_run(settings: WandbTrackingSettings, config: dict[str, Any]) -> Any | None:
    """
    Start a W&B run when enabled and return the run object.

    Parameters
    ----------
    settings
        W&B tracking settings.
    config
        JSON-serializable run configuration sent to W&B.

    Returns
    -------
    Any | None
        Active W&B run object, or ``None`` in disabled mode.

    Raises
    ------
    RuntimeError
        If W&B is requested but cannot be initialized safely.

    """
    resolved_mode = resolve_wandb_mode(settings.mode)
    if resolved_mode == WANDB_MODE_DISABLED:
        return None

    _load_wandb_api_key_from_home_file()
    if resolved_mode == WANDB_MODE_ONLINE and not os.environ.get("WANDB_API_KEY"):
        message = "W&B online mode requires WANDB_API_KEY in the environment or ${HOME}/wandb_key.txt"
        raise RuntimeError(message)

    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        message = f"W&B tracking requested with mode={settings.mode!r}, but wandb is not installed"
        raise RuntimeError(message) from exc

    if settings.dir is None and settings.name is None:
        message = "W&B tracking requires settings.dir or settings.name to resolve a run-scoped directory"
        raise RuntimeError(message)
    wandb_dir = settings.dir or default_wandb_dir(str(settings.name))
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ["WANDB_MODE"] = resolved_mode
    wandb_mode = cast("Literal['disabled', 'offline', 'online']", resolved_mode)
    return wandb.init(
        project=settings.project,
        entity=settings.entity,
        group=settings.group,
        name=settings.name,
        tags=list(settings.tags),
        config=config,
        dir=str(wandb_dir),
        mode=wandb_mode,
        sync_tensorboard=True,
    )


TRACKING_SUMMARY_KEYS = {
    "mean_position_error_m": "tracking/mean_position_error_m",
    "final_position_error_m": "tracking/final_position_error_m",
    "max_position_error_m": "tracking/max_position_error_m",
    "mean_abs_x_error": "tracking/mean_abs_x_error_m",
    "mean_abs_y_error": "tracking/mean_abs_y_error_m",
    "mean_abs_z_error": "tracking/mean_abs_z_error_m",
    "final_abs_x_error": "tracking/final_abs_x_error_m",
    "final_abs_y_error": "tracking/final_abs_y_error_m",
    "final_abs_z_error": "tracking/final_abs_z_error_m",
    "reference_xy_span_m": "tracking/reference_xy_span_m",
    "actual_xy_span_m": "tracking/actual_xy_span_m",
    "xy_tracking_ratio": "tracking/xy_tracking_ratio",
}
ACTION_VECTOR_SUMMARY_KEYS = {
    "action_mean": "actions/mean",
    "action_std": "actions/std",
    "action_min": "actions/min",
    "action_max": "actions/max",
    "action_saturation_fraction": "actions/saturation_fraction",
}
EVALUATION_SUMMARY_KEYS = {
    "eval_steps": "evaluation/eval_steps",
    "actual_eval_steps": "evaluation/actual_eval_steps",
    "eval_terminated_count": "evaluation/terminated_count",
    "eval_truncated_count": "evaluation/truncated_count",
    "eval_reset_count": "evaluation/reset_count",
    "episode_count": "evaluation/episode_count",
    "mean_eval_reward": "evaluation/mean_reward",
    "final_eval_reward": "evaluation/final_reward",
}
RUN_SUMMARY_KEYS = {
    "task_shape": "run/task_shape",
    "task_index": "run/task_index",
    "seed": "run/seed",
    "total_timesteps": "run/total_timesteps",
    "training_run_name": "run/training_run_name",
}
CURRICULUM_SUMMARY_KEYS = {
    "curriculum_readiness_level": "curriculum/readiness_level",
    "curriculum_recommended_next_tasks": "curriculum/recommended_next_tasks",
    "curriculum_avoid_next_tasks": "curriculum/avoid_next_tasks",
}
FAILURE_SUMMARY_MODES = (
    "hover_lock",
    "insufficient_xy_motion",
    "action_saturation",
    "overshoot",
    "z_instability",
    "attitude_instability",
    "early_termination",
    "repeated_truncation",
    "reference_too_fast_or_too_hard",
    "no_failure_detected",
)


def log_wandb_metrics(run: Any | None, metrics: dict[str, Any]) -> None:
    """Write final run-level metrics to W&B summary fields when tracking is enabled."""
    log_wandb_summary(run, metrics)


def log_wandb_summary(run: Any | None, metrics: dict[str, Any]) -> None:
    """Write grouped final diagnostics to ``run.summary`` without creating history charts."""
    if run is None:
        return
    summary_metrics = build_wandb_summary_metrics(metrics)
    for key, value in summary_metrics.items():
        run.summary[key] = value


def build_wandb_summary_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """
    Build grouped W&B summary metrics from flat PPO diagnostic metrics.

    Parameters
    ----------
    metrics
        Final training metrics payload written to the local metrics JSON.

    Returns
    -------
    dict[str, Any]
        Compact summary payload with grouped W&B keys. Large nested fields, paths,
        full diagnostics dictionaries, traces and unsupported objects are omitted.

    """
    summary: dict[str, Any] = {}
    _copy_scalar_summary(metrics, TRACKING_SUMMARY_KEYS, summary)
    _copy_scalar_summary(metrics, EVALUATION_SUMMARY_KEYS, summary)
    _copy_scalar_summary(metrics, RUN_SUMMARY_KEYS, summary)
    _copy_text_summary(metrics, CURRICULUM_SUMMARY_KEYS, summary)
    for source_key, key_prefix in ACTION_VECTOR_SUMMARY_KEYS.items():
        _copy_vector_summary(metrics.get(source_key), key_prefix, summary)
    _copy_failure_summary(metrics, summary)
    return summary


def log_wandb_artifacts(run: Any | None, paths: dict[str, Path]) -> None:
    """Log small final training artifacts to W&B when tracking is enabled."""
    if run is None:
        return
    try:
        import wandb  # noqa: PLC0415
    except ImportError:
        return
    for name, path in paths.items():
        if not path.is_file():
            continue
        artifact = wandb.Artifact(name=name, type="training-artifact")
        artifact.add_file(str(path))
        run.log_artifact(artifact)


def _load_wandb_api_key_from_home_file() -> None:
    """Populate WANDB_API_KEY from ${HOME}/wandb_key.txt when present."""
    if os.environ.get("WANDB_API_KEY"):
        return
    home = os.environ.get("HOME")
    if not home:
        return
    key_path = Path(home) / "wandb_key.txt"
    if not key_path.is_file():
        return
    key = key_path.read_text(encoding="utf-8").strip()
    if key:
        os.environ["WANDB_API_KEY"] = key


def _copy_scalar_summary(source: dict[str, Any], key_map: dict[str, str], target: dict[str, Any]) -> None:
    """Copy scalar summary values from source to grouped target keys."""
    for source_key, target_key in key_map.items():
        if source_key not in source:
            continue
        value = source[source_key]
        if _is_summary_scalar(value):
            target[target_key] = value


def _copy_text_summary(source: dict[str, Any], key_map: dict[str, str], target: dict[str, Any]) -> None:
    """Copy string or string-list summary values from source to grouped target keys."""
    for source_key, target_key in key_map.items():
        value = source.get(source_key)
        if isinstance(value, str) or (isinstance(value, list) and all(isinstance(item, str) for item in value)):
            target[target_key] = value


def _copy_vector_summary(value: Any, key_prefix: str, target: dict[str, Any]) -> None:
    """Copy a short numeric vector to indexed grouped summary keys."""
    if not isinstance(value, (list, tuple)):
        return
    for index, item in enumerate(value):
        if _is_summary_scalar(item) and not isinstance(item, str):
            target[f"{key_prefix}_{index}"] = item


def _copy_failure_summary(metrics: dict[str, Any], target: dict[str, Any]) -> None:
    """Convert detected failure modes into grouped 0/1 indicator fields."""
    modes = metrics.get("failure_modes")
    active_modes = {str(mode) for mode in modes} if isinstance(modes, list) else set()
    primary_mode = metrics.get("failure_primary_mode")
    if isinstance(primary_mode, str):
        active_modes.add(primary_mode)
    for mode in FAILURE_SUMMARY_MODES:
        target[f"failure/{mode}"] = int(mode in active_modes)


def _is_summary_scalar(value: Any) -> bool:
    """Return whether a value is safe to store as a compact W&B summary scalar."""
    return isinstance(value, (int, float, bool)) or value is None or isinstance(value, str)


__all__ = [
    "DEFAULT_WANDB_PROJECT",
    "WANDB_MODES",
    "WANDB_MODE_AUTO",
    "WANDB_MODE_DISABLED",
    "WANDB_MODE_OFFLINE",
    "WANDB_MODE_ONLINE",
    "WandbTrackingSettings",
    "build_wandb_summary_metrics",
    "default_wandb_dir",
    "log_wandb_artifacts",
    "log_wandb_metrics",
    "log_wandb_summary",
    "parse_wandb_tags",
    "resolve_wandb_mode",
    "start_wandb_run",
    "wandb_run",
]
