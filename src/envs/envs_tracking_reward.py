"""
===============================================================================
envs_tracking_reward.py
===============================================================================
Compute deterministic MVP trajectory-tracking rewards and episode steps.

Responsibilities:
  - Select reference samples from validated environment task references
  - Compute simple position-error and action-cost tracking rewards
  - Package per-step reward, error, and termination diagnostics for smoke loops

Design principles:
  - Keep reward logic pure, deterministic, and independent of PyBullet
  - Make invalid reference, position, and configuration data fail loudly

Boundaries:
  - Simulator environment construction stays in envs_builders.py
  - Training loops, rollout evaluation, plotting, and artifact writing belong elsewhere
===============================================================================

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from src import envs

XYZ_DIMENSIONS = 3
POSITION_ARRAY_NDIM = 2
MIN_REFERENCE_SAMPLES = 1


@dataclass(frozen=True)
class TrackingRewardConfig:
    """
    Configuration for the conservative MVP tracking reward.

    Parameters
    ----------
    position_error_weight
        Nonnegative multiplier applied to Euclidean position error.
    action_cost_weight
        Nonnegative multiplier applied to squared action magnitude when an action is supplied.
    success_tolerance_m
        Nonnegative distance threshold used by callers that need a success indicator.
    max_steps
        Optional maximum number of episode steps. When set, step indices at ``max_steps - 1`` or later are terminal.

    Notes
    -----
    This reward is intentionally simple. It is an MVP smoke-test contract, not the final research reward.

    """

    position_error_weight: float = 1.0
    action_cost_weight: float = 0.01
    success_tolerance_m: float = 0.1
    max_steps: int | None = None

    def __post_init__(self) -> None:
        """Validate reward configuration values."""
        _ensure_nonnegative_finite(self.position_error_weight, name="position_error_weight")
        _ensure_nonnegative_finite(self.action_cost_weight, name="action_cost_weight")
        _ensure_nonnegative_finite(self.success_tolerance_m, name="success_tolerance_m")
        if self.max_steps is not None and self.max_steps <= 0:
            message = "max_steps must be positive when provided"
            raise ValueError(message)


@dataclass(frozen=True)
class TrackingStepResult:
    """
    Per-step deterministic tracking reward output.

    Parameters
    ----------
    step_index
        Zero-based reference step index.
    reference_position
        Copied XYZ reference position for this step.
    actual_position
        Copied XYZ actual position supplied by the caller.
    position_error_m
        Euclidean XYZ tracking error in meters.
    action_cost
        Squared action magnitude used in the reward calculation.
    reward
        Scalar reward after position-error and action-cost penalties.
    done
        Whether the episode is terminal at this step.
    success
        Whether the position error is within ``TrackingRewardConfig.success_tolerance_m``.

    """

    step_index: int
    reference_position: np.ndarray
    actual_position: np.ndarray
    position_error_m: float
    action_cost: float
    reward: float
    done: bool
    success: bool


def select_reference_position(reference: envs.task_adapter.EnvironmentTaskReference, step_index: int) -> np.ndarray:
    """
    Return a copied XYZ reference position for a step index.

    Parameters
    ----------
    reference
        Validated environment task reference containing sampled positions.
    step_index
        Zero-based trajectory sample index.

    Returns
    -------
    np.ndarray
        Copied XYZ reference position.

    Raises
    ------
    ValueError
        If the reference arrays or step index are invalid.

    """
    positions = _validated_reference_positions(reference)
    if step_index < 0:
        message = "step_index must be nonnegative"
        raise ValueError(message)
    if step_index >= positions.shape[0]:
        message = "step_index is outside the reference trajectory"
        raise ValueError(message)
    return np.array(positions[step_index], dtype=float, copy=True)


def compute_tracking_reward(
    actual_position: object,
    reference_position: object,
    action: object | None = None,
    config: TrackingRewardConfig | None = None,
) -> float:
    """
    Compute the conservative MVP tracking reward for one sample.

    Parameters
    ----------
    actual_position
        Actual XYZ position supplied by a policy, baseline, or simulator.
    reference_position
        Reference XYZ position sampled from a validated task.
    action
        Optional action vector. When supplied, squared magnitude is penalized.
    config
        Optional reward configuration. Defaults are used when omitted.

    Returns
    -------
    float
        Negative weighted position error and action cost.

    """
    active_config = config or TrackingRewardConfig()
    actual = _as_xyz_position(actual_position, name="actual_position")
    reference = _as_xyz_position(reference_position, name="reference_position")
    action_cost = _action_cost(action)
    position_error_m = float(np.linalg.norm(actual - reference))
    return -active_config.position_error_weight * position_error_m - active_config.action_cost_weight * action_cost


def step_tracking_episode(
    reference: envs.task_adapter.EnvironmentTaskReference,
    actual_position: object,
    step_index: int,
    action: object | None = None,
    config: TrackingRewardConfig | None = None,
) -> TrackingStepResult:
    """
    Compute reward and terminal diagnostics for one trajectory-tracking step.

    Parameters
    ----------
    reference
        Validated environment task reference containing sampled positions.
    actual_position
        Actual XYZ position for the current step.
    step_index
        Zero-based trajectory sample index.
    action
        Optional action vector used for action-cost penalty.
    config
        Optional reward configuration. Defaults are used when omitted.

    Returns
    -------
    TrackingStepResult
        Copied reference/actual positions, reward, error, and terminal flags.

    """
    active_config = config or TrackingRewardConfig()
    positions = _validated_reference_positions(reference)
    reference_position = select_reference_position(reference=reference, step_index=step_index)
    actual = _as_xyz_position(actual_position, name="actual_position")
    action_cost = _action_cost(action)
    position_error_m = float(np.linalg.norm(actual - reference_position))
    reward = -active_config.position_error_weight * position_error_m - active_config.action_cost_weight * action_cost
    done_by_reference = step_index >= positions.shape[0] - 1
    done_by_max_steps = active_config.max_steps is not None and step_index >= active_config.max_steps - 1
    return TrackingStepResult(
        step_index=step_index,
        reference_position=reference_position,
        actual_position=np.array(actual, dtype=float, copy=True),
        position_error_m=position_error_m,
        action_cost=action_cost,
        reward=float(reward),
        done=bool(done_by_reference or done_by_max_steps),
        success=bool(position_error_m <= active_config.success_tolerance_m),
    )


def _validated_reference_positions(reference: envs.task_adapter.EnvironmentTaskReference) -> np.ndarray:
    """Return validated reference positions as a float array."""
    positions = np.asarray(reference.positions, dtype=float)
    if positions.ndim != POSITION_ARRAY_NDIM or positions.shape[1:] != (XYZ_DIMENSIONS,):
        message = "reference positions must have shape (num_samples, 3)"
        raise ValueError(message)
    if positions.shape[0] < MIN_REFERENCE_SAMPLES:
        message = "reference positions must contain at least one sample"
        raise ValueError(message)
    if not np.all(np.isfinite(positions)):
        message = "reference positions must contain only finite values"
        raise ValueError(message)
    return positions


def _as_xyz_position(value: object, name: str) -> np.ndarray:
    """Convert an object to a finite XYZ position array."""
    position = np.asarray(value, dtype=float)
    if position.shape != (XYZ_DIMENSIONS,):
        message = f"{name} must have shape (3,)"
        raise ValueError(message)
    if not np.all(np.isfinite(position)):
        message = f"{name} must contain only finite values"
        raise ValueError(message)
    return position


def _action_cost(action: object | None) -> float:
    """Return squared action magnitude for a finite action vector."""
    if action is None:
        return 0.0
    action_array = np.asarray(action, dtype=float)
    if action_array.ndim == 0:
        message = "action must be an array-like vector"
        raise ValueError(message)
    if not np.all(np.isfinite(action_array)):
        message = "action must contain only finite values"
        raise ValueError(message)
    return float(np.sum(np.square(action_array)))


def _ensure_nonnegative_finite(value: float, name: str) -> None:
    """Raise ValueError unless a scalar value is finite and nonnegative."""
    if not np.isfinite(value) or value < 0.0:
        message = f"{name} must be finite and nonnegative"
        raise ValueError(message)


__all__ = [
    "TrackingRewardConfig",
    "TrackingStepResult",
    "compute_tracking_reward",
    "select_reference_position",
    "step_tracking_episode",
]
