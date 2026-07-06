"""
===============================================================================
utils_wandb.py
===============================================================================
Provide optional Weights & Biases tracking for bounded experiment smoke runs.

Responsibilities:
  - Keep W&B disabled and dependency-free unless explicitly requested
  - Lazily initialize W&B with run-specific directories and non-secret metadata
  - Safely load WANDB_API_KEY from the environment or an optional home key file

Design principles:
  - Never store or print secrets
  - Keep online tracking opt-in and fail clearly when credentials are missing
  - Make disabled mode a no-op for tests, Docker, and local smoke runs

Boundaries:
  - Training modules decide which metrics to log
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

WANDB_MODE_DISABLED = "disabled"
WANDB_MODE_OFFLINE = "offline"
WANDB_MODE_ONLINE = "online"
WANDB_MODES = (WANDB_MODE_DISABLED, WANDB_MODE_OFFLINE, WANDB_MODE_ONLINE)
DEFAULT_WANDB_PROJECT = "drone-rl-llm-curriculum"
DEFAULT_WANDB_RUN_NAME = "ppo_hover_smoke"


@dataclass(frozen=True)
class WandbTrackingSettings:
    """
    Settings for optional W&B run tracking.

    Parameters
    ----------
    mode
        Tracking mode. ``disabled`` performs no import or logging.
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

    mode: str = WANDB_MODE_DISABLED
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


def default_wandb_dir() -> Path:
    """Return the default run-specific W&B directory for PPO tracking smoke."""
    return artifacts.get_training_wandb_dir(DEFAULT_WANDB_RUN_NAME)


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
    if settings.mode == WANDB_MODE_DISABLED:
        return None

    _load_wandb_api_key_from_home_file()
    if settings.mode == WANDB_MODE_ONLINE and not os.environ.get("WANDB_API_KEY"):
        message = "W&B online mode requires WANDB_API_KEY in the environment or ${HOME}/wandb_key.txt"
        raise RuntimeError(message)

    try:
        import wandb  # noqa: PLC0415
    except ImportError as exc:
        message = f"W&B tracking requested with mode={settings.mode!r}, but wandb is not installed"
        raise RuntimeError(message) from exc

    wandb_dir = settings.dir or default_wandb_dir()
    wandb_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("WANDB_MODE", settings.mode)
    wandb_mode = cast("Literal['disabled', 'offline', 'online']", settings.mode)
    return wandb.init(
        project=settings.project,
        entity=settings.entity,
        group=settings.group,
        name=settings.name,
        tags=list(settings.tags),
        config=config,
        dir=str(wandb_dir),
        mode=wandb_mode,
    )


def log_wandb_metrics(run: Any | None, metrics: dict[str, Any]) -> None:
    """Log final metrics when W&B tracking is enabled."""
    if run is None:
        return
    run.log(_flatten_wandb_metrics(metrics))


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


def _flatten_wandb_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    """Return scalar metrics suitable for a compact W&B summary log."""
    return {key: value for key, value in metrics.items() if isinstance(value, (str, int, float, bool)) or value is None}


__all__ = [
    "DEFAULT_WANDB_PROJECT",
    "DEFAULT_WANDB_RUN_NAME",
    "WANDB_MODES",
    "WANDB_MODE_DISABLED",
    "WANDB_MODE_OFFLINE",
    "WANDB_MODE_ONLINE",
    "WandbTrackingSettings",
    "default_wandb_dir",
    "log_wandb_metrics",
    "parse_wandb_tags",
    "start_wandb_run",
    "wandb_run",
]
