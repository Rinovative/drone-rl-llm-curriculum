"""
===============================================================================
experiments_training_ppo_config.py
===============================================================================
Validate Stable-Baselines3 PPO hyperparameters loaded from experiment configs.

Responsibilities:
  - Represent the explicit PPO hyperparameter contract for tracking training
  - Load nested PPO settings from resolved experiment mappings
  - Convert validated settings into Stable-Baselines3 PPO keyword arguments

Design principles:
  - Keep validation deterministic and independent of Stable-Baselines3 imports
  - Make tiny smoke-training rollout sizes explicit in config rather than hidden

Boundaries:
  - Training modules own task selection, environment construction, and artifacts
  - This module does not start PPO training or inspect generated model files
===============================================================================

"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from math import isfinite
from typing import Any

PPO_CONFIG_KEYS = (
    "policy",
    "device",
    "learning_rate",
    "gamma",
    "gae_lambda",
    "n_steps",
    "batch_size",
    "n_epochs",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "target_kl",
)


@dataclass(frozen=True)
class PPOConfig:
    """
    Stable-Baselines3 PPO hyperparameters for trajectory-tracking training.

    Parameters
    ----------
    policy
        SB3 policy identifier or policy class name.
    device
        Torch device selector passed to SB3, such as ``cpu``, ``cuda``, or ``auto``.
    learning_rate
        Positive optimizer learning rate.
    gamma
        Discount factor in the interval ``(0, 1]``.
    gae_lambda
        Generalized advantage estimation lambda in the interval ``(0, 1]``.
    n_steps
        Number of rollout steps per environment for each PPO update.
    batch_size
        Minibatch size used for PPO updates. Rollout-size consistency is
        validated separately against ``n_steps * num_envs`` by the training settings.
    n_epochs
        Number of optimization epochs per rollout.
    clip_range
        Positive PPO policy clipping range.
    ent_coef
        Nonnegative entropy coefficient.
    vf_coef
        Nonnegative value-function coefficient.
    max_grad_norm
        Positive gradient clipping norm.
    target_kl
        Optional positive target KL early-stopping threshold.

    """

    policy: str = "MlpPolicy"
    device: str = "cpu"
    learning_rate: float = 0.0003
    gamma: float = 0.99
    gae_lambda: float = 0.95
    n_steps: int = 256
    batch_size: int = 64
    n_epochs: int = 5
    clip_range: float = 0.2
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float | None = 0.03

    def __post_init__(self) -> None:
        """Normalize scalar values and validate the PPO hyperparameter contract."""
        object.__setattr__(self, "policy", _required_text(self.policy, "ppo.policy"))
        object.__setattr__(self, "device", _required_text(self.device, "ppo.device"))
        object.__setattr__(self, "learning_rate", _positive_float(self.learning_rate, "ppo.learning_rate"))
        object.__setattr__(self, "gamma", _positive_float(self.gamma, "ppo.gamma", upper=1.0))
        object.__setattr__(self, "gae_lambda", _positive_float(self.gae_lambda, "ppo.gae_lambda", upper=1.0))
        object.__setattr__(self, "n_steps", _positive_int(self.n_steps, "ppo.n_steps"))
        object.__setattr__(self, "batch_size", _positive_int(self.batch_size, "ppo.batch_size"))
        object.__setattr__(self, "n_epochs", _positive_int(self.n_epochs, "ppo.n_epochs"))
        object.__setattr__(self, "clip_range", _positive_float(self.clip_range, "ppo.clip_range"))
        object.__setattr__(self, "ent_coef", _nonnegative_float(self.ent_coef, "ppo.ent_coef"))
        object.__setattr__(self, "vf_coef", _nonnegative_float(self.vf_coef, "ppo.vf_coef"))
        object.__setattr__(self, "max_grad_norm", _positive_float(self.max_grad_norm, "ppo.max_grad_norm"))
        object.__setattr__(self, "target_kl", _optional_positive_float(self.target_kl, "ppo.target_kl"))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None, *, section_name: str = "ppo") -> PPOConfig:
        """
        Build a PPO config from a mapping while rejecting unsupported nested keys.

        Parameters
        ----------
        values
            Mapping containing PPO hyperparameter values. Missing keys use dataclass defaults.
        section_name
            Human-readable config section prefix used in validation errors.

        Returns
        -------
        PPOConfig
            Validated PPO configuration.

        Raises
        ------
        ValueError
            If the mapping contains unsupported keys or invalid hyperparameter values.

        """
        if values is None:
            return cls()
        unknown_keys = sorted(set(values) - set(PPO_CONFIG_KEYS))
        if unknown_keys:
            message = f"{section_name} contains unsupported keys: {', '.join(unknown_keys)}"
            raise ValueError(message)
        return cls(**{key: values[key] for key in PPO_CONFIG_KEYS if key in values})

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable resolved PPO config."""
        return asdict(self)

    def to_sb3_kwargs(self) -> dict[str, Any]:
        """Return keyword arguments accepted by ``stable_baselines3.PPO``."""
        return self.to_dict()

    def validate_total_timesteps(self, total_timesteps: int) -> None:
        """
        Validate that a training budget can collect at least one configured rollout.

        Parameters
        ----------
        total_timesteps
            PPO learning budget requested for the training run.

        Raises
        ------
        ValueError
            If the total timestep budget is smaller than ``ppo.n_steps``.

        """
        resolved_total_timesteps = _positive_int(total_timesteps, "total_timesteps")
        if resolved_total_timesteps < self.n_steps:
            message = "total_timesteps must be greater than or equal to ppo.n_steps; set an explicit smaller ppo.n_steps for tiny smoke tests"
            raise ValueError(message)

    def effective_rollout_steps(self, num_envs: int) -> int:
        """
        Return PPO rollout samples collected per update across all environments.

        Parameters
        ----------
        num_envs
            Number of parallel training environments.

        Returns
        -------
        int
            Effective rollout sample count, ``ppo.n_steps * num_envs``.

        """
        return self.n_steps * _positive_int(num_envs, "num_envs")

    def validate_rollout_consistency(self, num_envs: int) -> None:
        """
        Validate minibatch size against the vectorized PPO rollout size.

        Parameters
        ----------
        num_envs
            Number of parallel training environments used for rollout collection.

        Raises
        ------
        ValueError
            If ``ppo.batch_size`` exceeds ``ppo.n_steps * num_envs``.

        """
        effective_rollout_steps = self.effective_rollout_steps(num_envs)
        if self.batch_size > effective_rollout_steps:
            message = "ppo.batch_size must be less than or equal to ppo.n_steps * num_envs"
            raise ValueError(message)


def load_ppo_config_from_mapping(config: Mapping[str, Any]) -> PPOConfig:
    """
    Load the resolved PPO config from an experiment config mapping.

    Parameters
    ----------
    config
        Loaded experiment config. PPO hyperparameters must live under a nested ``ppo`` block.

    Returns
    -------
    PPOConfig
        Validated resolved PPO configuration.

    Raises
    ------
    ValueError
        If ``ppo`` is missing, if top-level PPO keys are present, or if PPO values are invalid.

    """
    flat_keys = sorted(set(config) & set(PPO_CONFIG_KEYS))
    if flat_keys:
        message = f"top-level PPO keys are not supported; move under ppo: {', '.join(flat_keys)}"
        raise ValueError(message)
    nested_config = config.get("ppo")
    if nested_config is None:
        message = "ppo config section is required"
        raise ValueError(message)
    if not isinstance(nested_config, Mapping):
        message = "ppo config section must be a mapping"
        raise ValueError(message)  # noqa: TRY004 - public config errors are reported as ValueError.
    return PPOConfig.from_mapping(dict(nested_config))


def _required_text(value: Any, name: str) -> str:
    """Return a stripped non-empty text value."""
    if value is None:
        message = f"{name} must be non-empty"
        raise ValueError(message)
    text = str(value).strip()
    if not text:
        message = f"{name} must be non-empty"
        raise ValueError(message)
    return text


def _positive_int(value: Any, name: str) -> int:
    """Return a strictly positive integer."""
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


def _positive_float(value: Any, name: str, *, upper: float | None = None) -> float:
    """Return a finite float greater than zero and optionally below an upper bound."""
    resolved = _finite_float(value, name)
    if resolved <= 0.0:
        message = f"{name} must be greater than 0"
        raise ValueError(message)
    if upper is not None and resolved > upper:
        message = f"{name} must be less than or equal to {upper:g}"
        raise ValueError(message)
    return resolved


def _nonnegative_float(value: Any, name: str) -> float:
    """Return a finite float greater than or equal to zero."""
    resolved = _finite_float(value, name)
    if resolved < 0.0:
        message = f"{name} must be greater than or equal to 0"
        raise ValueError(message)
    return resolved


def _optional_positive_float(value: Any, name: str) -> float | None:
    """Return ``None`` or a finite float greater than zero."""
    if value is None:
        return None
    return _positive_float(value, name)


def _finite_float(value: Any, name: str) -> float:
    """Return a finite float and reject booleans as numeric config values."""
    if isinstance(value, bool):
        message = f"{name} must be a finite number"
        raise ValueError(message)  # noqa: TRY004 - public config errors are reported as ValueError.
    try:
        resolved = float(value)
    except (TypeError, ValueError) as exc:
        message = f"{name} must be a finite number"
        raise ValueError(message) from exc
    if not isfinite(resolved):
        message = f"{name} must be a finite number"
        raise ValueError(message)
    return resolved


__all__ = [
    "PPO_CONFIG_KEYS",
    "PPOConfig",
    "load_ppo_config_from_mapping",
]
