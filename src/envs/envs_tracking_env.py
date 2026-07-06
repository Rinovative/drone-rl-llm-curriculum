"""
===============================================================================
envs_tracking_env.py
===============================================================================
Provide a minimal Gymnasium trajectory-tracking wrapper around HoverAviary.

Responsibilities:
  - Adapt validated trajectory tasks into a compact single-drone RL environment
  - Expose deterministic tracking observations for later PPO smoke training
  - Delegate low-level simulation, action semantics, and physics to HoverAviary

Design principles:
  - Keep the wrapper small and compatible with Gymnasium reset and step APIs
  - Use deterministic task validation before constructing reference trajectories
  - Reuse existing tracking reward helpers rather than duplicating reward logic

Boundaries:
  - PPO, model persistence, curriculum loops, and rollout evaluation belong elsewhere
  - This module must not create videos, training artifacts, or GUI requirements

Notes:
  Position extraction
  The MVP wrapper reads XYZ position from HoverAviary._getDroneStateVector(0).
  In gym-pybullet-drones this helper returns the simulator state vector with
  position in the first three entries, in meters. Using the upstream helper keeps
  this wrapper independent of the raw observation layout chosen by HoverAviary.
===============================================================================

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src import envs, validation

if TYPE_CHECKING:
    from collections.abc import Mapping

OBSERVATION_DIMENSIONS = 10
XYZ_DIMENSIONS = 3
OBSERVATION_BOUND = 1.0e6


class TrajectoryTrackingEnv(gym.Env[np.ndarray, Any]):
    """
    Minimal single-drone trajectory-tracking environment.

    Parameters
    ----------
    task
        Valid trajectory task mapping or prebuilt environment task reference.
    gui
        Whether the wrapped HoverAviary should open a GUI.
    record
        Whether the wrapped HoverAviary should record frames.
    limits
        Optional validation limits used when ``task`` is a mapping.
    max_steps
        Optional maximum number of wrapper steps before trajectory termination.

    Notes
    -----
    The action space is the wrapped HoverAviary action space. The observation is
    a compact float32 vector containing current XYZ position, reference XYZ
    position, XYZ position error, and normalized trajectory progress.

    """

    def __init__(
        self,
        task: Mapping[str, Any] | envs.task_adapter.EnvironmentTaskReference,
        gui: bool = False,
        record: bool = False,
        limits: validation.tasks.ValidationLimits | None = None,
        max_steps: int | None = None,
    ) -> None:
        """Initialize the tracking wrapper and its base HoverAviary environment."""
        super().__init__()
        self.metadata = {"render_modes": []}
        self.reference = _coerce_task_reference(task=task, limits=limits)
        self.base_env = envs.builders.make_hover_aviary_env(gui=gui, record=record)
        self.action_space = self.base_env.action_space
        self.observation_space = spaces.Box(
            low=np.full(OBSERVATION_DIMENSIONS, -OBSERVATION_BOUND, dtype=np.float32),
            high=np.full(OBSERVATION_DIMENSIONS, OBSERVATION_BOUND, dtype=np.float32),
            dtype=np.float32,
        )
        self.reward_config = envs.tracking_reward.TrackingRewardConfig(max_steps=max_steps)
        self._step_index = 0
        self._closed = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the base simulator and return the compact tracking observation.

        Parameters
        ----------
        seed
            Optional random seed forwarded to HoverAviary.
        options
            Optional reset options forwarded to HoverAviary.

        Returns
        -------
        tuple[np.ndarray, dict[str, Any]]
            Compact observation and reset diagnostics.

        """
        super().reset(seed=seed)
        _, base_info = self.base_env.reset(seed=seed, options=options)
        self._step_index = 0
        current_position = self._current_position()
        reference_position = envs.tracking_reward.select_reference_position(self.reference, self._step_index)
        observation = self._make_observation(
            current_position=current_position,
            reference_position=reference_position,
            step_index=self._step_index,
        )
        info = self._make_info(
            current_position=current_position,
            reference_position=reference_position,
            position_error_m=float(np.linalg.norm(current_position - reference_position)),
            tracking_success=False,
            base_info=base_info,
        )
        return observation, info

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Advance HoverAviary once and compute deterministic tracking reward.

        Parameters
        ----------
        action
            Action passed directly to the wrapped HoverAviary environment.

        Returns
        -------
        tuple[np.ndarray, float, bool, bool, dict[str, Any]]
            Observation, tracking reward, termination flag, truncation flag, and
            diagnostics for the completed tracking step.

        """
        _, base_reward, base_terminated, base_truncated, base_info = self.base_env.step(action)
        current_position = self._current_position()
        tracking_result = envs.tracking_reward.step_tracking_episode(
            reference=self.reference,
            actual_position=current_position,
            step_index=self._step_index,
            action=action,
            config=self.reward_config,
        )
        next_step_index = min(self._step_index + 1, self.reference.positions.shape[0] - 1)
        next_reference_position = envs.tracking_reward.select_reference_position(self.reference, next_step_index)
        observation = self._make_observation(
            current_position=current_position,
            reference_position=next_reference_position,
            step_index=next_step_index,
        )
        self._step_index = next_step_index
        info = self._make_info(
            current_position=tracking_result.actual_position,
            reference_position=tracking_result.reference_position,
            position_error_m=tracking_result.position_error_m,
            tracking_success=tracking_result.success,
            base_info=base_info,
        )
        info["base_reward"] = float(base_reward)
        terminated = bool(tracking_result.done or base_terminated)
        truncated = bool(base_truncated)
        return observation, float(tracking_result.reward), terminated, truncated, info

    def close(self) -> None:
        """Close the wrapped HoverAviary environment safely."""
        if self._closed:
            return
        close = getattr(self.base_env, "close", None)
        if close is None:
            self._closed = True
            return

        import pybullet as pybullet_client  # noqa: PLC0415

        try:
            close()
        except pybullet_client.error as exc:
            if "Not connected to physics server" not in str(exc):
                raise
        finally:
            self._closed = True

    def _current_position(self) -> np.ndarray:
        """Extract the current drone XYZ position from the HoverAviary state vector."""
        state_getter = getattr(self.base_env, "_getDroneStateVector", None)
        if state_getter is None:
            message = "HoverAviary does not expose _getDroneStateVector for XYZ extraction"
            raise RuntimeError(message)
        state = np.asarray(state_getter(0), dtype=float)
        if state.shape[0] < XYZ_DIMENSIONS:
            message = "HoverAviary state vector is too short to contain XYZ position"
            raise RuntimeError(message)
        position = np.array(state[:XYZ_DIMENSIONS], dtype=float, copy=True)
        if position.shape != (XYZ_DIMENSIONS,) or not np.all(np.isfinite(position)):
            message = "HoverAviary XYZ position must be a finite shape-(3,) vector"
            raise RuntimeError(message)
        return position

    def _make_observation(
        self,
        current_position: np.ndarray,
        reference_position: np.ndarray,
        step_index: int,
    ) -> np.ndarray:
        """Build the compact float32 observation vector."""
        position_error = current_position - reference_position
        progress = self._normalized_progress(step_index)
        observation = np.concatenate(
            [
                current_position,
                reference_position,
                position_error,
                np.array([progress], dtype=float),
            ]
        )
        return observation.astype(np.float32, copy=False)

    def _normalized_progress(self, step_index: int) -> float:
        """Return trajectory progress normalized to ``[0, 1]``."""
        final_index = self.reference.positions.shape[0] - 1
        if final_index <= 0:
            return 1.0
        return float(np.clip(step_index / final_index, 0.0, 1.0))

    def _make_info(
        self,
        current_position: np.ndarray,
        reference_position: np.ndarray,
        position_error_m: float,
        tracking_success: bool,
        base_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Package tracking diagnostics with copied simulator metadata."""
        return {
            "reference_position": np.array(reference_position, dtype=float, copy=True),
            "current_position": np.array(current_position, dtype=float, copy=True),
            "position_error_m": float(position_error_m),
            "task_shape": self.reference.shape,
            "tracking_success": bool(tracking_success),
            "base_info": dict(base_info),
        }


def make_trajectory_tracking_env(
    task: Mapping[str, Any] | envs.task_adapter.EnvironmentTaskReference,
    gui: bool = False,
    record: bool = False,
    limits: validation.tasks.ValidationLimits | None = None,
    max_steps: int | None = None,
) -> TrajectoryTrackingEnv:
    """
    Build a ready-to-use minimal trajectory-tracking environment.

    Parameters
    ----------
    task
        Valid trajectory task mapping or prebuilt environment task reference.
    gui
        Whether the wrapped HoverAviary should open a GUI.
    record
        Whether the wrapped HoverAviary should record frames.
    limits
        Optional validation limits used when ``task`` is a mapping.
    max_steps
        Optional maximum number of wrapper steps before trajectory termination.

    Returns
    -------
    TrajectoryTrackingEnv
        Ready-to-use Gymnasium-compatible tracking environment.

    """
    return TrajectoryTrackingEnv(task=task, gui=gui, record=record, limits=limits, max_steps=max_steps)


def _coerce_task_reference(
    task: Mapping[str, Any] | envs.task_adapter.EnvironmentTaskReference,
    limits: validation.tasks.ValidationLimits | None,
) -> envs.task_adapter.EnvironmentTaskReference:
    """Return an environment task reference, validating mappings as needed."""
    if isinstance(task, envs.task_adapter.EnvironmentTaskReference):
        return task
    return envs.task_adapter.make_task_reference(task=task, limits=limits)


__all__ = [
    "TrajectoryTrackingEnv",
    "make_trajectory_tracking_env",
]
