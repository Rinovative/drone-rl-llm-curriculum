"""
===============================================================================
envs_tracking_env.py
===============================================================================
Provide a minimal Gymnasium trajectory-tracking wrapper around HoverAviary.

Responsibilities:
  - Adapt validated trajectory tasks into a compact single-drone RL environment
  - Expose deterministic tracking observations for later PPO smoke training
  - Delegate low-level simulation, PID action semantics, and physics to HoverAviary

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

from typing import TYPE_CHECKING, Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src import envs, validation

if TYPE_CHECKING:
    from collections.abc import Mapping

OBSERVATION_DIMENSIONS = 10
XYZ_DIMENSIONS = 3
STATE_VECTOR_MIN_DIMENSIONS = 20
OBSERVATION_BOUND = 1.0e6
BASE_XY_TRUNCATION_LIMIT_M = 1.5
BASE_Z_TRUNCATION_LIMIT_M = 2.0
BASE_ATTITUDE_TRUNCATION_LIMIT_RAD = 0.4
TRACKING_ACTION_XY_MARGIN_M = 0.2
TRACKING_ACTION_Z_MARGIN_M = 0.5
NORMALIZED_ACTION_LOW = -1.0
NORMALIZED_ACTION_HIGH = 1.0


class NormalizedActionWrapper(gym.Wrapper[np.ndarray, Any, np.ndarray, Any]):
    """
    Map symmetric PPO actions in ``[-1, 1]`` to real PID target-position actions.

    Parameters
    ----------
    env
        Trajectory tracking environment exposing the real PID target-position action space.

    Notes
    -----
    The wrapped environment physics and PID semantics are unchanged. Only the
    action interface seen by PPO is normalized; every step forwards the mapped
    real action to the underlying environment and records both forms in ``info``.

    """

    def __init__(self, env: gym.Env[np.ndarray, Any]) -> None:
        """Initialize the normalized action interface around a tracking environment."""
        super().__init__(env)
        self.real_action_space = cast("spaces.Box", env.action_space)
        self.normalized_action_space = spaces.Box(
            low=np.full(self.real_action_space.shape, NORMALIZED_ACTION_LOW, dtype=np.float32),
            high=np.full(self.real_action_space.shape, NORMALIZED_ACTION_HIGH, dtype=np.float32),
            dtype=np.float32,
        )
        self.action_space = self.normalized_action_space

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Map a normalized PPO action to a real PID action before stepping."""
        normalized_action = _clip_action_to_space(np.asarray(action), self.normalized_action_space)
        real_action = normalized_to_real_action(normalized_action, self.real_action_space)
        observation, reward, terminated, truncated, info = self.env.step(real_action)
        info = dict(info)
        info["normalized_action"] = np.array(normalized_action, dtype=float, copy=True)
        info["real_action"] = np.array(real_action, dtype=float, copy=True)
        info["action_normalized"] = True
        info["real_action_space_low"] = np.array(self.real_action_space.low, dtype=float, copy=True)
        info["real_action_space_high"] = np.array(self.real_action_space.high, dtype=float, copy=True)
        return observation, float(reward), terminated, truncated, info

    def normalized_to_real_action(self, action: Any) -> np.ndarray:
        """Map a normalized action from ``[-1, 1]`` into the real PID action bounds."""
        return normalized_to_real_action(action, self.real_action_space)

    def real_to_normalized_action(self, action: Any) -> np.ndarray:
        """Map a real PID target-position action into normalized PPO coordinates."""
        return real_to_normalized_action(action, self.real_action_space)


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
    episode_len_sec
        Optional upstream HoverAviary episode duration in seconds.

    Notes
    -----
    The action space is HoverAviary's PID target-position action space with
    shape ``(1, 3)``. The observation is a compact float32 vector containing current XYZ position, reference XYZ
    position, XYZ position error, and normalized trajectory progress.

    """

    def __init__(
        self,
        task: Mapping[str, Any] | envs.task_adapter.EnvironmentTaskReference,
        gui: bool = False,
        record: bool = False,
        limits: validation.tasks.ValidationLimits | None = None,
        max_steps: int | None = None,
        episode_len_sec: float | None = None,
    ) -> None:
        """Initialize the tracking wrapper and its base HoverAviary environment."""
        super().__init__()
        self.metadata = {"render_modes": []}
        self.reference = _coerce_task_reference(task=task, limits=limits)
        self.base_env = _make_tracking_base_env(gui=gui, record=record, episode_len_sec=episode_len_sec)
        self._tracking_action_space = _make_tracking_action_space(self.reference, self.base_env.action_space)
        self.action_space = self._tracking_action_space
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
        requested_action_array = np.asarray(action)
        base_action_array = _clip_action_to_space(requested_action_array, self._tracking_action_space)
        _, base_reward, base_terminated, base_truncated, base_info = self.base_env.step(base_action_array)
        state = self._current_state_vector()
        current_position = _state_position(state)
        tracking_result = envs.tracking_reward.step_tracking_episode(
            reference=self.reference,
            actual_position=current_position,
            step_index=self._step_index,
            action=base_action_array,
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
        terminated = bool(tracking_result.done or base_terminated)
        truncated = bool(base_truncated)
        info = self._make_info(
            current_position=tracking_result.actual_position,
            reference_position=tracking_result.reference_position,
            position_error_m=tracking_result.position_error_m,
            tracking_success=tracking_result.success,
            base_info=base_info,
            state=state,
            base_terminated=bool(base_terminated),
            base_truncated=bool(base_truncated),
            termination_reason=_termination_reason(
                tracking_done=tracking_result.done,
                base_terminated=bool(base_terminated),
                base_truncated=bool(base_truncated),
                base_truncation_causes=_base_truncation_causes(self.base_env, state),
                step_index=tracking_result.step_index,
                max_steps=self.reward_config.max_steps,
                reference_sample_count=int(self.reference.positions.shape[0]),
            ),
            requested_action=requested_action_array,
            applied_action=base_action_array,
        )
        info["base_reward"] = float(base_reward)
        info["base_action_shape"] = tuple(int(dimension) for dimension in base_action_array.shape)
        info["base_action_dtype"] = str(base_action_array.dtype)
        info["action_shape"] = info["base_action_shape"]
        info["action_dtype"] = info["base_action_dtype"]
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
        return _state_position(self._current_state_vector())

    def _current_state_vector(self) -> np.ndarray:
        """Return the current upstream HoverAviary state vector for diagnostics."""
        state_getter = getattr(self.base_env, "_getDroneStateVector", None)
        if state_getter is None:
            message = "HoverAviary does not expose _getDroneStateVector for diagnostics"
            raise RuntimeError(message)
        state = np.asarray(state_getter(0), dtype=float)
        if state.shape[0] < STATE_VECTOR_MIN_DIMENSIONS:
            message = "HoverAviary state vector is too short for tracking diagnostics"
            raise RuntimeError(message)
        if not np.all(np.isfinite(state)):
            message = "HoverAviary state vector must contain only finite values"
            raise RuntimeError(message)
        return np.array(state, dtype=float, copy=True)

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
        state: np.ndarray | None = None,
        base_terminated: bool = False,
        base_truncated: bool = False,
        termination_reason: str = "reset",
        requested_action: np.ndarray | None = None,
        applied_action: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Package tracking diagnostics with copied simulator metadata."""
        active_state = self._current_state_vector() if state is None else np.asarray(state, dtype=float)
        attitude = np.array(active_state[7:10], dtype=float, copy=True)
        velocity = np.array(active_state[10:13], dtype=float, copy=True)
        angular_velocity = np.array(active_state[13:16], dtype=float, copy=True)
        last_action = np.array(active_state[16:20], dtype=float, copy=True)
        current = np.array(current_position, dtype=float, copy=True)
        reference = np.array(reference_position, dtype=float, copy=True)
        return {
            "reference_position": reference,
            "reference_xyz": reference,
            "current_position": current,
            "current_xyz": current,
            "position_error_m": float(position_error_m),
            "task_shape": self.reference.shape,
            "tracking_success": bool(tracking_success),
            "roll_pitch_yaw": attitude,
            "velocity": velocity,
            "angular_velocity": angular_velocity,
            "last_action": last_action,
            "requested_action": None if requested_action is None else np.array(requested_action, dtype=float, copy=True),
            "applied_action": None if applied_action is None else np.array(applied_action, dtype=float, copy=True),
            "base_terminated": bool(base_terminated),
            "base_truncated": bool(base_truncated),
            "base_truncation_causes": _base_truncation_causes(self.base_env, active_state),
            "base_info": dict(base_info),
            "base_info_keys": sorted(str(key) for key in base_info),
            "base_reason_fields": _base_reason_fields(base_info),
            "termination_reason": termination_reason,
        }


def make_trajectory_tracking_env(
    task: Mapping[str, Any] | envs.task_adapter.EnvironmentTaskReference,
    gui: bool = False,
    record: bool = False,
    limits: validation.tasks.ValidationLimits | None = None,
    max_steps: int | None = None,
    episode_len_sec: float | None = None,
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
    episode_len_sec
        Optional upstream HoverAviary episode duration in seconds.

    Returns
    -------
    TrajectoryTrackingEnv
        Ready-to-use Gymnasium-compatible tracking environment.

    """
    return TrajectoryTrackingEnv(
        task=task,
        gui=gui,
        record=record,
        limits=limits,
        max_steps=max_steps,
        episode_len_sec=episode_len_sec,
    )


def make_normalized_action_env(env: gym.Env[np.ndarray, Any]) -> NormalizedActionWrapper:
    """Wrap a tracking environment with a symmetric normalized action interface."""
    return NormalizedActionWrapper(env)


def normalized_to_real_action(action: Any, real_action_space: spaces.Box) -> np.ndarray:
    """
    Map normalized ``[-1, 1]`` actions to real PID target-position bounds.

    Parameters
    ----------
    action
        PPO-facing normalized action. Values are clipped to ``[-1, 1]`` before mapping.
    real_action_space
        Underlying tracking environment action space with real PID target-position bounds.

    Returns
    -------
    np.ndarray
        Real action computed as ``low + (normalized + 1) * 0.5 * (high - low)``.

    """
    normalized = np.asarray(action, dtype=np.float32)
    normalized = np.clip(normalized, NORMALIZED_ACTION_LOW, NORMALIZED_ACTION_HIGH)
    low = np.asarray(real_action_space.low, dtype=np.float32)
    high = np.asarray(real_action_space.high, dtype=np.float32)
    return (low + (normalized + 1.0) * 0.5 * (high - low)).astype(real_action_space.dtype, copy=False)


def real_to_normalized_action(action: Any, real_action_space: spaces.Box) -> np.ndarray:
    """
    Map real PID target-position actions back to normalized PPO coordinates.

    Parameters
    ----------
    action
        Real PID target-position action.
    real_action_space
        Underlying tracking environment action space with real PID target-position bounds.

    Returns
    -------
    np.ndarray
        Normalized action computed as ``2 * (real - low) / (high - low) - 1``.

    """
    real = np.asarray(action, dtype=np.float32)
    low = np.asarray(real_action_space.low, dtype=np.float32)
    high = np.asarray(real_action_space.high, dtype=np.float32)
    normalized = 2.0 * (real - low) / (high - low) - 1.0
    return np.clip(normalized, NORMALIZED_ACTION_LOW, NORMALIZED_ACTION_HIGH).astype(np.float32, copy=False)


def _make_tracking_action_space(
    reference: envs.task_adapter.EnvironmentTaskReference,
    base_action_space: spaces.Box,
) -> spaces.Box:
    """Build conservative PID target-position bounds around the reference path."""
    base_low = np.asarray(base_action_space.low, dtype=np.float32)
    base_high = np.asarray(base_action_space.high, dtype=np.float32)
    positions = np.asarray(reference.positions, dtype=np.float32)
    reference_min = np.min(positions, axis=0)
    reference_max = np.max(positions, axis=0)
    margin = np.array([TRACKING_ACTION_XY_MARGIN_M, TRACKING_ACTION_XY_MARGIN_M, TRACKING_ACTION_Z_MARGIN_M], dtype=np.float32)
    task_low = (reference_min - margin).reshape(base_low.shape)
    task_high = (reference_max + margin).reshape(base_high.shape)
    low = np.maximum(base_low, task_low).astype(np.float32, copy=False)
    high = np.minimum(base_high, task_high).astype(np.float32, copy=False)
    high = np.maximum(high, low + np.finfo(np.float32).eps)
    return spaces.Box(low=low, high=high, dtype=np.float32)


def _clip_action_to_space(action: np.ndarray, action_space: spaces.Box) -> np.ndarray:
    """Clip an incoming PID target to the wrapper's conservative action bounds."""
    action_array = np.asarray(action, dtype=action_space.dtype)
    return np.clip(action_array, action_space.low, action_space.high).astype(action_space.dtype, copy=False)


def _state_position(state: np.ndarray) -> np.ndarray:
    """Extract a finite XYZ position from an upstream HoverAviary state vector."""
    position = np.array(state[:XYZ_DIMENSIONS], dtype=float, copy=True)
    if position.shape != (XYZ_DIMENSIONS,) or not np.all(np.isfinite(position)):
        message = "HoverAviary XYZ position must be a finite shape-(3,) vector"
        raise RuntimeError(message)
    return position


def _base_reason_fields(base_info: dict[str, Any]) -> dict[str, Any]:
    """Return upstream info fields whose names look like termination reasons."""
    reason_fields: dict[str, Any] = {}
    for key, value in base_info.items():
        key_text = str(key).lower()
        if "reason" in key_text or "termin" in key_text or "trunc" in key_text or "done" in key_text:
            reason_fields[str(key)] = value
    return reason_fields


def _base_truncation_causes(base_env: Any, state: np.ndarray) -> list[str]:
    """Return HoverAviary truncation conditions currently visible in the state."""
    causes: list[str] = []
    x_position, y_position, z_position = (float(value) for value in state[:3])
    roll, pitch = (float(value) for value in state[7:9])
    if abs(x_position) > BASE_XY_TRUNCATION_LIMIT_M:
        causes.append("x_position_out_of_bounds")
    if abs(y_position) > BASE_XY_TRUNCATION_LIMIT_M:
        causes.append("y_position_out_of_bounds")
    if z_position > BASE_Z_TRUNCATION_LIMIT_M:
        causes.append("z_position_above_limit")
    if abs(roll) > BASE_ATTITUDE_TRUNCATION_LIMIT_RAD:
        causes.append("roll_above_limit")
    if abs(pitch) > BASE_ATTITUDE_TRUNCATION_LIMIT_RAD:
        causes.append("pitch_above_limit")

    step_counter = float(getattr(base_env, "step_counter", 0.0))
    pyb_steps_per_ctrl = float(getattr(base_env, "PYB_STEPS_PER_CTRL", 0.0))
    pyb_frequency = float(getattr(base_env, "PYB_FREQ", 0.0))
    episode_len_sec = float(getattr(base_env, "EPISODE_LEN_SEC", np.inf))
    compute_step_counter = max(step_counter - pyb_steps_per_ctrl, 0.0)
    if pyb_frequency > 0.0 and compute_step_counter / pyb_frequency > episode_len_sec:
        causes.append("episode_time_limit")
    return causes


def _termination_reason(
    tracking_done: bool,
    base_terminated: bool,
    base_truncated: bool,
    base_truncation_causes: list[str],
    step_index: int,
    max_steps: int | None,
    reference_sample_count: int,
) -> str:
    """Return the most specific rollout termination reason available."""
    if base_truncated:
        if base_truncation_causes:
            return "base_truncated:" + ",".join(base_truncation_causes)
        return "base_truncated:upstream_unclassified"
    if base_terminated:
        return "base_terminated:hover_target_reached"
    if tracking_done and max_steps is not None and step_index >= max_steps - 1:
        return "tracking_max_steps_reached"
    if tracking_done and step_index >= reference_sample_count - 1:
        return "tracking_reference_complete"
    if tracking_done:
        return "tracking_done"
    return "running"


def _make_tracking_base_env(gui: bool, record: bool, episode_len_sec: float | None = None) -> Any:
    """Build the HoverAviary instance used by trajectory tracking."""
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary  # noqa: PLC0415
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType  # noqa: PLC0415

    base_env = HoverAviary(gui=gui, record=record, obs=ObservationType.KIN, act=ActionType.PID)
    if episode_len_sec is not None:
        if not np.isfinite(float(episode_len_sec)) or float(episode_len_sec) <= 0.0:
            message = "episode_len_sec must be finite and positive when provided"
            raise ValueError(message)
        base_env.EPISODE_LEN_SEC = float(episode_len_sec)
    return base_env


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
