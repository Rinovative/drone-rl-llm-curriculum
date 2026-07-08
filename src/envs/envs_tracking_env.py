"""
===============================================================================
envs_tracking_env.py
===============================================================================
Provide a Gymnasium trajectory-tracking wrapper around HoverAviary.

Responsibilities:
  - Adapt validated trajectory tasks into a compact single-drone RL environment
  - Expose deterministic tracking observations for later PPO smoke training
  - Support explicit PID target-position and direct motor-RPM action interfaces
  - Delegate low-level simulation and physics to HoverAviary

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

from collections.abc import Mapping
from typing import Any, cast

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from src import envs, validation

OBSERVATION_DIMENSIONS = 10
DYNAMICS_OBSERVATION_DIMENSIONS = 9
XYZ_DIMENSIONS = 3
PROGRESS_DIMENSIONS = 1
RPM_MOTOR_COUNT = 4
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
        self.action_interface = envs.actions.ActionInterface.PID_POSITION.value
        self.ppo_action_dim = _action_dimension(self.action_space)

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Map a normalized PPO action to a real PID action before stepping."""
        normalized_action = _clip_action_to_space(np.asarray(action), self.normalized_action_space)
        real_action = normalized_to_real_action(normalized_action, self.real_action_space)
        set_previous_action = getattr(self.env, "_set_previous_ppo_action_for_next_observation", None)
        if callable(set_previous_action):
            set_previous_action(normalized_action)
        observation, reward, terminated, truncated, info = self.env.step(real_action)
        info = dict(info)
        info["normalized_action"] = np.array(normalized_action, dtype=float, copy=True)
        info["real_action"] = np.array(real_action, dtype=float, copy=True)
        info["action_normalized"] = True
        info["action_interface"] = self.action_interface
        info["real_action_type"] = "pid_target_position"
        info["ppo_action_dim"] = self.ppo_action_dim
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
        Valid trajectory task mapping, task-distribution sampler/settings, or prebuilt environment task reference.
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
    action_interface
        Explicit action interface, either ``pid_position`` or ``direct_rpm``.
    rpm_delta_scale
        Fractional RPM delta around hover used only by ``direct_rpm``.
    include_dynamics_observation
        Whether observations append velocity, attitude, and angular velocity.
    include_previous_action
        Whether observations append the previous PPO-facing action.
    termination_limits
        Optional hard episode-control safety limits. Defaults preserve upstream truncation behavior.
    diagnostic_limits
        Optional strict diagnostic thresholds reported independently of episode truncation.

    Notes
    -----
    ``pid_position`` preserves the existing target-position PID behavior. The
    experimental ``direct_rpm`` interface exposes four normalized motor commands
    for one drone and maps them to real RPMs before PyBullet physics.

    """

    def __init__(
        self,
        task: Mapping[str, Any]
        | envs.task_adapter.EnvironmentTaskReference
        | envs.task_distribution.TaskDistributionSettings
        | envs.task_distribution.TaskDistributionSampler,
        gui: bool = False,
        record: bool = False,
        limits: validation.tasks.ValidationLimits | None = None,
        max_steps: int | None = None,
        episode_len_sec: float | None = None,
        action_interface: envs.actions.ActionInterface | str = envs.actions.DEFAULT_ACTION_INTERFACE,
        rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE,
        include_dynamics_observation: bool = envs.actions.DEFAULT_INCLUDE_DYNAMICS_OBSERVATION,
        include_previous_action: bool = envs.actions.DEFAULT_INCLUDE_PREVIOUS_ACTION,
        termination_limits: envs.termination.TerminationLimitConfig | Mapping[str, Any] | str | None = None,
        diagnostic_limits: envs.termination.DiagnosticLimitConfig | Mapping[str, Any] | str | None = None,
    ) -> None:
        """Initialize the tracking wrapper and its base HoverAviary environment."""
        super().__init__()
        self.metadata = {"render_modes": []}
        self.task_distribution_sampler = _coerce_task_distribution_sampler(task)
        self._task_reference_limits = _task_reference_limits(limits, self.task_distribution_sampler)
        self.reference = _initial_task_reference(task=task, sampler=self.task_distribution_sampler, limits=self._task_reference_limits)
        self.task_distribution_metadata = _task_distribution_metadata(self.task_distribution_sampler)
        self.action_config = envs.actions.ActionInterfaceConfig(
            action_interface=action_interface,
            rpm_delta_scale=rpm_delta_scale,
            include_dynamics_observation=include_dynamics_observation,
            include_previous_action=include_previous_action,
        )
        self.action_interface = self.action_config.parsed_action_interface.value
        self.termination_limits = envs.termination.parse_termination_limits(
            termination_limits,
            self.action_config.parsed_action_interface,
        )
        self.diagnostic_limits = envs.termination.parse_diagnostic_limits(diagnostic_limits)
        self.rpm_delta_scale = self.action_config.rpm_delta_scale
        self.include_dynamics_observation = self.action_config.include_dynamics_observation
        self.include_previous_action = self.action_config.include_previous_action
        self.base_env = _make_tracking_base_env(
            gui=gui,
            record=record,
            episode_len_sec=episode_len_sec,
            action_interface=self.action_config.parsed_action_interface,
            rpm_delta_scale=self.rpm_delta_scale,
        )
        self._use_distribution_action_space = _use_distribution_action_space(self.task_distribution_sampler)
        self._tracking_action_space = _make_tracking_action_space(
            self.reference,
            self.base_env.action_space,
            self.action_config.parsed_action_interface,
            use_base_bounds=self._use_distribution_action_space,
        )
        self.action_space = self._tracking_action_space
        self.ppo_action_dim = _action_dimension(self.action_space)
        self.observation_components = _observation_components(
            include_dynamics_observation=self.include_dynamics_observation,
            include_previous_action=self.include_previous_action,
            previous_action_dim=self.ppo_action_dim,
        )
        self.observation_dim = _observation_dimensions(self.observation_components)
        self.observation_space = spaces.Box(
            low=np.full(self.observation_dim, -OBSERVATION_BOUND, dtype=np.float32),
            high=np.full(self.observation_dim, OBSERVATION_BOUND, dtype=np.float32),
            dtype=np.float32,
        )
        self.reward_config = envs.tracking_reward.TrackingRewardConfig(max_steps=max_steps)
        self._step_index = 0
        self._strict_limit_violation_count = 0
        self._termination_limit_violation_steps = 0
        self._previous_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self._previous_action_override: np.ndarray | None = None
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
        self._refresh_task_reference_for_reset()
        _, base_info = self.base_env.reset(seed=seed, options=options)
        self._step_index = 0
        self._strict_limit_violation_count = 0
        self._termination_limit_violation_steps = 0
        self._reset_previous_action()
        state = self._current_state_vector()
        current_position = _state_position(state)
        reference_position = envs.tracking_reward.select_reference_position(self.reference, self._step_index)
        observation = self._make_observation(
            current_position=current_position,
            reference_position=reference_position,
            step_index=self._step_index,
            state=state,
        )
        info = self._make_info(
            current_position=current_position,
            reference_position=reference_position,
            position_error_m=float(np.linalg.norm(current_position - reference_position)),
            tracking_success=False,
            base_info=base_info,
            state=state,
            reference_step_index=self._step_index,
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
        direct_rpm_metadata = self._direct_rpm_step_metadata(
            requested_action=requested_action_array,
            applied_action=base_action_array,
        )
        _, base_reward, base_terminated, base_truncated, base_info = self.base_env.step(base_action_array)
        self._update_previous_action_after_step(base_action_array)
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
            state=state,
        )
        self._step_index = next_step_index
        base_truncation_causes = _base_truncation_causes(self.base_env, state)
        strict_limit_violations = envs.termination.state_limit_violations(state, self.diagnostic_limits)
        project_truncation_causes = envs.termination.state_limit_violations(state, self.termination_limits)
        if strict_limit_violations:
            self._strict_limit_violation_count += 1
        if project_truncation_causes:
            self._termination_limit_violation_steps += 1
        else:
            self._termination_limit_violation_steps = 0
        base_truncation_effective = bool(base_truncated and self.termination_limits.terminate_on_base_truncation)
        project_truncated = bool(project_truncation_causes and self._termination_limit_violation_steps > self.termination_limits.allow_recovery_steps)
        terminated = bool(tracking_result.done or base_terminated)
        truncated = bool(base_truncation_effective or project_truncated)
        info = self._make_info(
            current_position=tracking_result.actual_position,
            reference_position=tracking_result.reference_position,
            position_error_m=tracking_result.position_error_m,
            tracking_success=tracking_result.success,
            base_info=base_info,
            state=state,
            base_terminated=bool(base_terminated),
            base_truncated=bool(base_truncated),
            base_truncation_causes=base_truncation_causes,
            base_truncation_effective=base_truncation_effective,
            project_truncated=project_truncated,
            project_truncation_causes=project_truncation_causes,
            strict_limit_violations=strict_limit_violations,
            termination_limit_violation_steps=self._termination_limit_violation_steps,
            termination_reason=_termination_reason(
                tracking_done=tracking_result.done,
                base_terminated=bool(base_terminated),
                base_truncated=bool(base_truncation_effective),
                base_truncation_causes=base_truncation_causes,
                project_truncated=project_truncated,
                project_truncation_causes=project_truncation_causes,
                step_index=tracking_result.step_index,
                max_steps=self.reward_config.max_steps,
                reference_sample_count=int(self.reference.positions.shape[0]),
            ),
            requested_action=requested_action_array,
            applied_action=base_action_array,
            reference_step_index=tracking_result.step_index,
        )
        info.update(direct_rpm_metadata)
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

    def _refresh_task_reference_for_reset(self) -> None:
        """Refresh the active reference from a task-distribution sampler when configured."""
        if self.task_distribution_sampler is None:
            return
        task = self.task_distribution_sampler.sample_task()
        self.reference = envs.task_adapter.make_task_reference(task, limits=self._task_reference_limits)
        self.task_distribution_metadata = _task_distribution_metadata(self.task_distribution_sampler)

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
        state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build the compact float32 observation vector."""
        position_error = current_position - reference_position
        progress = self._normalized_progress(step_index)
        fields = [
            current_position,
            reference_position,
            position_error,
            np.array([progress], dtype=float),
        ]
        if self.include_dynamics_observation:
            active_state = self._current_state_vector() if state is None else np.asarray(state, dtype=float)
            fields.extend(
                [
                    np.array(active_state[10:13], dtype=float, copy=True),
                    np.array(active_state[7:10], dtype=float, copy=True),
                    np.array(active_state[13:16], dtype=float, copy=True),
                ]
            )
        if self.include_previous_action:
            fields.append(np.asarray(self._previous_action, dtype=float).reshape(-1))
        observation = np.concatenate(fields)
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
        base_truncation_causes: list[str] | None = None,
        base_truncation_effective: bool | None = None,
        project_truncated: bool = False,
        project_truncation_causes: list[str] | None = None,
        strict_limit_violations: list[str] | None = None,
        termination_limit_violation_steps: int = 0,
        termination_reason: str = "reset",
        requested_action: np.ndarray | None = None,
        applied_action: np.ndarray | None = None,
        reference_step_index: int = 0,
    ) -> dict[str, Any]:
        """Package tracking diagnostics with copied simulator metadata."""
        active_state = self._current_state_vector() if state is None else np.asarray(state, dtype=float)
        attitude = np.array(active_state[7:10], dtype=float, copy=True)
        velocity = np.array(active_state[10:13], dtype=float, copy=True)
        angular_velocity = np.array(active_state[13:16], dtype=float, copy=True)
        last_action = np.array(active_state[16:20], dtype=float, copy=True)
        base_causes = _base_truncation_causes(self.base_env, active_state) if base_truncation_causes is None else list(base_truncation_causes)
        strict_causes = (
            envs.termination.state_limit_violations(active_state, self.diagnostic_limits)
            if strict_limit_violations is None
            else list(strict_limit_violations)
        )
        project_causes = (
            envs.termination.state_limit_violations(active_state, self.termination_limits)
            if project_truncation_causes is None
            else list(project_truncation_causes)
        )
        strict_limit_violation = bool(strict_causes)
        recovery_allowed = bool(project_causes and termination_limit_violation_steps <= self.termination_limits.allow_recovery_steps)
        current = np.array(current_position, dtype=float, copy=True)
        reference = np.array(reference_position, dtype=float, copy=True)
        reference_index = int(np.clip(reference_step_index, 0, self.reference.times.shape[0] - 1))
        tracking_phase_start_step = int(self.reference.tracking_phase_start_step)
        tracking_phase_start_time_sec = float(self.reference.tracking_phase_start_time_sec)
        is_start_hold = bool(self.reference.start_hold_enabled and reference_index < tracking_phase_start_step)
        task_distribution_fields = _compact_task_distribution_info_fields(self.task_distribution_metadata)
        return {
            "action_interface": self.action_interface,
            "real_action_type": self._real_action_type(),
            "ppo_action_dim": self.ppo_action_dim,
            "include_dynamics_observation": self.include_dynamics_observation,
            "include_previous_action": self.include_previous_action,
            "observation_dim": self.observation_dim,
            "observation_components": _copy_observation_components(self.observation_components),
            "previous_action": np.array(self._previous_action, dtype=float, copy=True),
            "direct_control_limitations": envs.actions.direct_control_limitations(self.action_config.parsed_action_interface),
            "termination_limits_mode": self.termination_limits.mode,
            "termination_limits": self.termination_limits.to_dict(),
            "diagnostic_limits": self.diagnostic_limits.to_dict(),
            "base_truncation_policy": self.termination_limits.base_truncation_policy,
            "terminate_on_base_truncation": self.termination_limits.terminate_on_base_truncation,
            "reference_position": reference,
            "reference_xyz": reference,
            "current_position": current,
            "current_xyz": current,
            "position_error_m": float(position_error_m),
            "task_shape": self.reference.shape,
            "task_distribution": dict(self.task_distribution_metadata),
            **task_distribution_fields,
            "reference_step_index": reference_index,
            "reference_time_sec": float(self.reference.times[reference_index]),
            "start_hold_enabled": bool(self.reference.start_hold_enabled),
            "start_hold_sec": float(self.reference.start_hold_sec),
            "exclude_start_hold_from_tracking_metrics": bool(self.reference.exclude_start_hold_from_tracking_metrics),
            "tracking_phase_start_step": tracking_phase_start_step,
            "tracking_phase_start_time_sec": tracking_phase_start_time_sec,
            "is_start_hold": is_start_hold,
            "is_tracking_phase": not is_start_hold,
            "tracking_success": bool(tracking_success),
            "roll_pitch_yaw": attitude,
            "velocity": velocity,
            "angular_velocity": angular_velocity,
            "last_action": last_action,
            "requested_action": None if requested_action is None else np.array(requested_action, dtype=float, copy=True),
            "applied_action": None if applied_action is None else np.array(applied_action, dtype=float, copy=True),
            "base_terminated": bool(base_terminated),
            "base_truncated": bool(base_truncated),
            "base_truncation_effective": bool(base_truncated if base_truncation_effective is None else base_truncation_effective),
            "base_truncation_ignored": bool(
                base_truncated and not (base_truncated if base_truncation_effective is None else base_truncation_effective)
            ),
            "base_truncation_causes": base_causes,
            "project_truncated": bool(project_truncated),
            "project_truncation_causes": project_causes,
            "strict_limit_violation": strict_limit_violation,
            "strict_limit_violations": strict_causes,
            "strict_limit_violation_count": int(self._strict_limit_violation_count),
            "recovery_allowed_after_limit_violation": recovery_allowed,
            "recovery_steps_after_limit_violation": int(termination_limit_violation_steps),
            "recovery_steps_limit": int(self.termination_limits.allow_recovery_steps),
            "base_info": dict(base_info),
            "base_info_keys": sorted(str(key) for key in base_info),
            "base_reason_fields": _base_reason_fields(base_info),
            "termination_reason": termination_reason,
        }

    def _direct_rpm_step_metadata(self, requested_action: np.ndarray, applied_action: np.ndarray) -> dict[str, Any]:
        """Return per-step direct-RPM metadata or an empty mapping for PID control."""
        if self.action_config.parsed_action_interface != envs.actions.ActionInterface.DIRECT_RPM:
            return {}
        hover_rpm = _hover_rpm(self.base_env)
        rpm_min = _rpm_min(self.base_env)
        rpm_max = _rpm_max(self.base_env)
        motor_rpms = normalized_direct_rpm_to_motor_rpms(
            applied_action,
            hover_rpm=hover_rpm,
            rpm_delta_scale=self.rpm_delta_scale,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
        )
        unclipped_motor_rpms = hover_rpm * (1.0 + self.rpm_delta_scale * np.asarray(applied_action, dtype=float))
        command_low = normalized_direct_rpm_to_motor_rpms(
            np.full(self.action_space.shape, NORMALIZED_ACTION_LOW, dtype=np.float32),
            hover_rpm=hover_rpm,
            rpm_delta_scale=self.rpm_delta_scale,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
        )
        command_high = normalized_direct_rpm_to_motor_rpms(
            np.full(self.action_space.shape, NORMALIZED_ACTION_HIGH, dtype=np.float32),
            hover_rpm=hover_rpm,
            rpm_delta_scale=self.rpm_delta_scale,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
        )
        rpm_saturation_mask = np.logical_or(
            np.isclose(motor_rpms, rpm_min, atol=1.0e-3, rtol=0.0),
            np.isclose(motor_rpms, rpm_max, atol=1.0e-3, rtol=0.0),
        )
        return {
            "normalized_action": np.array(applied_action, dtype=float, copy=True),
            "real_action": np.array(motor_rpms, dtype=float, copy=True),
            "real_motor_rpms": np.array(motor_rpms, dtype=float, copy=True),
            "action_normalized": True,
            "action_clipped": _action_was_clipped(requested_action, applied_action),
            "hover_rpm": hover_rpm,
            "rpm_delta_scale": self.rpm_delta_scale,
            "rpm_min": rpm_min,
            "rpm_max": rpm_max,
            "rpm_command_space_low": np.array(command_low, dtype=float, copy=True),
            "rpm_command_space_high": np.array(command_high, dtype=float, copy=True),
            "rpm_clipped": bool(np.any(~np.isclose(unclipped_motor_rpms, motor_rpms, atol=1.0e-3, rtol=0.0))),
            "rpm_saturation_mask": rpm_saturation_mask,
            "real_action_space_low": np.full(self.action_space.shape, rpm_min, dtype=float),
            "real_action_space_high": np.full(self.action_space.shape, rpm_max, dtype=float),
        }

    def _real_action_type(self) -> str:
        """Return the real action command type used by the base simulator."""
        if self.action_config.parsed_action_interface == envs.actions.ActionInterface.DIRECT_RPM:
            return "motor_rpm"
        return "pid_target_position"

    def _set_previous_ppo_action_for_next_observation(self, action: Any) -> None:
        """Set a one-step PPO-facing previous-action override for action wrappers."""
        self._previous_action_override = np.array(np.asarray(action, dtype=np.float32), dtype=np.float32, copy=True)

    def _reset_previous_action(self) -> None:
        """Reset the previous-action observation component to zeros."""
        action_space = cast("spaces.Box", self.action_space)
        self._previous_action = np.zeros(action_space.shape, dtype=np.float32)
        self._previous_action_override = None

    def _update_previous_action_after_step(self, action: Any) -> None:
        """Update the previous-action observation component after applying an action."""
        if self._previous_action_override is not None:
            self._previous_action = self._previous_action_override
            self._previous_action_override = None
            return
        self._previous_action = self._coerce_previous_action(action)

    def _coerce_previous_action(self, action: Any) -> np.ndarray:
        """Return a clipped action array shaped like the PPO-facing action space."""
        action_space = cast("spaces.Box", self.action_space)
        return np.array(_clip_action_to_space(np.asarray(action), action_space), dtype=np.float32, copy=True)


def make_trajectory_tracking_env(
    task: Mapping[str, Any]
    | envs.task_adapter.EnvironmentTaskReference
    | envs.task_distribution.TaskDistributionSettings
    | envs.task_distribution.TaskDistributionSampler,
    gui: bool = False,
    record: bool = False,
    limits: validation.tasks.ValidationLimits | None = None,
    max_steps: int | None = None,
    episode_len_sec: float | None = None,
    action_interface: envs.actions.ActionInterface | str = envs.actions.DEFAULT_ACTION_INTERFACE,
    rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE,
    include_dynamics_observation: bool = envs.actions.DEFAULT_INCLUDE_DYNAMICS_OBSERVATION,
    include_previous_action: bool = envs.actions.DEFAULT_INCLUDE_PREVIOUS_ACTION,
    termination_limits: envs.termination.TerminationLimitConfig | Mapping[str, Any] | str | None = None,
    diagnostic_limits: envs.termination.DiagnosticLimitConfig | Mapping[str, Any] | str | None = None,
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
    action_interface
        Explicit action interface, either ``pid_position`` or ``direct_rpm``.
    rpm_delta_scale
        Fractional RPM delta around hover used only by ``direct_rpm``.
    include_dynamics_observation
        Whether observations append velocity, attitude, and angular velocity.
    include_previous_action
        Whether observations append the previous PPO-facing action.
    termination_limits
        Optional hard episode-control safety limits.
    diagnostic_limits
        Optional strict diagnostic thresholds reported independently of episode control.

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
        action_interface=action_interface,
        rpm_delta_scale=rpm_delta_scale,
        include_dynamics_observation=include_dynamics_observation,
        include_previous_action=include_previous_action,
        termination_limits=termination_limits,
        diagnostic_limits=diagnostic_limits,
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
    action_interface: envs.actions.ActionInterface,
    use_base_bounds: bool = False,
) -> spaces.Box:
    """Build action bounds for the selected tracking interface."""
    base_low = np.asarray(base_action_space.low, dtype=np.float32)
    base_high = np.asarray(base_action_space.high, dtype=np.float32)
    if action_interface == envs.actions.ActionInterface.DIRECT_RPM or use_base_bounds:
        return spaces.Box(low=base_low, high=base_high, dtype=np.float32)
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
    """Clip an incoming action to the wrapper's configured action bounds."""
    action_array = np.asarray(action, dtype=action_space.dtype)
    return np.clip(action_array, action_space.low, action_space.high).astype(action_space.dtype, copy=False)


def _coerce_task_distribution_sampler(task: Any) -> envs.task_distribution.TaskDistributionSampler | None:
    """Return a task-distribution sampler when task carries distribution settings."""
    if isinstance(task, envs.task_distribution.TaskDistributionSampler):
        return task
    if isinstance(task, envs.task_distribution.TaskDistributionSettings):
        return envs.task_distribution.TaskDistributionSampler(task)
    return None


def _initial_task_reference(
    task: Mapping[str, Any]
    | envs.task_adapter.EnvironmentTaskReference
    | envs.task_distribution.TaskDistributionSettings
    | envs.task_distribution.TaskDistributionSampler,
    sampler: envs.task_distribution.TaskDistributionSampler | None,
    limits: validation.tasks.ValidationLimits | None,
) -> envs.task_adapter.EnvironmentTaskReference:
    """Build the initial task reference from a fixed task or sampler."""
    if sampler is None:
        return _coerce_task_reference(task=task, limits=limits)
    return envs.task_adapter.make_task_reference(sampler.sample_task(), limits=limits)


def _task_reference_limits(
    limits: validation.tasks.ValidationLimits | None,
    sampler: envs.task_distribution.TaskDistributionSampler | None,
) -> validation.tasks.ValidationLimits | None:
    """Return explicit limits or distribution-specific validation limits."""
    if limits is not None:
        return limits
    if sampler is None:
        return None
    return sampler.settings.validation_limits


def _task_distribution_metadata(sampler: envs.task_distribution.TaskDistributionSampler | None) -> dict[str, Any]:
    """Return current task-distribution metadata or disabled metadata."""
    if sampler is None:
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
    return sampler.sample_metadata()


def _compact_task_distribution_info_fields(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact task-distribution fields suitable for every info payload."""
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
        "task_distribution_env_rank",
        "task_distribution_effective_seed",
        "task_distribution_sample_index",
        "task_distribution_sampled_family",
        "task_distribution_sampled_task_shape",
        "task_distribution_sampled_task_name",
    )
    return {key: copy_value for key in keys if (copy_value := metadata.get(key)) is not None}


def _use_distribution_action_space(sampler: envs.task_distribution.TaskDistributionSampler | None) -> bool:
    """Return whether randomized reset sampling needs stable base action bounds."""
    if sampler is None:
        return False
    settings = sampler.settings
    return bool(settings.enabled and settings.mode == envs.task_distribution.MODE_RANDOMIZED and settings.sample_on_reset and settings.strength > 0.0)


def _action_was_clipped(requested_action: np.ndarray, applied_action: np.ndarray) -> bool:
    """Return whether an incoming action changed during action-space clipping."""
    requested = np.asarray(requested_action, dtype=float)
    applied = np.asarray(applied_action, dtype=float)
    return requested.shape != applied.shape or bool(np.any(~np.isclose(requested, applied, atol=1.0e-3, rtol=0.0)))


def _action_dimension(action_space: spaces.Box) -> int:
    """Return the flattened PPO action dimension for a Box action space."""
    return int(np.prod(tuple(action_space.shape)))


def _observation_components(
    include_dynamics_observation: bool,
    include_previous_action: bool,
    previous_action_dim: int,
) -> list[dict[str, int | str]]:
    """Return named observation components for the selected observation contract."""
    components: list[dict[str, int | str]] = [
        {"name": "current_position", "dim": XYZ_DIMENSIONS},
        {"name": "reference_position", "dim": XYZ_DIMENSIONS},
        {"name": "position_error", "dim": XYZ_DIMENSIONS},
        {"name": "trajectory_progress", "dim": PROGRESS_DIMENSIONS},
    ]
    if include_dynamics_observation:
        components.extend(
            [
                {"name": "linear_velocity", "dim": XYZ_DIMENSIONS},
                {"name": "attitude_rpy", "dim": XYZ_DIMENSIONS},
                {"name": "angular_velocity", "dim": XYZ_DIMENSIONS},
            ]
        )
    if include_previous_action:
        components.append({"name": "previous_action", "dim": int(previous_action_dim)})
    return components


def _observation_dimensions(components: list[dict[str, int | str]]) -> int:
    """Return the tracking observation dimension for named observation components."""
    return int(sum(int(component["dim"]) for component in components))


def _copy_observation_components(components: list[dict[str, int | str]]) -> list[dict[str, int | str]]:
    """Return JSON-safe copies of observation-component metadata."""
    return [dict(component) for component in components]


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
    project_truncated: bool,
    project_truncation_causes: list[str],
    step_index: int,
    max_steps: int | None,
    reference_sample_count: int,
) -> str:
    """Return the most specific rollout termination reason available."""
    if project_truncated:
        if project_truncation_causes:
            return "project_truncated:" + ",".join(project_truncation_causes)
        return "project_truncated:unclassified"
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


def normalized_direct_rpm_to_motor_rpms(
    action: Any,
    hover_rpm: float,
    rpm_delta_scale: float,
    rpm_min: float,
    rpm_max: float,
) -> np.ndarray:
    """
    Map normalized direct-RPM actions to clipped physical motor RPMs.

    Parameters
    ----------
    action
        Normalized per-motor commands. Values are clipped to ``[-1, 1]``.
    hover_rpm
        Hover RPM reported by the upstream drone model.
    rpm_delta_scale
        Fractional RPM delta around hover for normalized command magnitude one.
    rpm_min
        Minimum physically valid RPM.
    rpm_max
        Maximum physically valid RPM.

    Returns
    -------
    np.ndarray
        Clipped motor RPM commands with the same shape as ``action``.

    """
    normalized = np.clip(np.asarray(action, dtype=float), NORMALIZED_ACTION_LOW, NORMALIZED_ACTION_HIGH)
    rpm = float(hover_rpm) * (1.0 + float(rpm_delta_scale) * normalized)
    return np.clip(rpm, float(rpm_min), float(rpm_max)).astype(np.float32, copy=False)


def _make_tracking_base_env(
    gui: bool,
    record: bool,
    episode_len_sec: float | None = None,
    action_interface: envs.actions.ActionInterface = envs.actions.DEFAULT_ACTION_INTERFACE,
    rpm_delta_scale: float = envs.actions.DEFAULT_RPM_DELTA_SCALE,
) -> Any:
    """Build the HoverAviary instance used by trajectory tracking."""
    from gym_pybullet_drones.envs.HoverAviary import HoverAviary  # noqa: PLC0415
    from gym_pybullet_drones.utils.enums import ActionType, ObservationType  # noqa: PLC0415

    class ConfigurableDirectRPMHoverAviary(HoverAviary):
        """HoverAviary variant with configurable normalized direct-RPM scaling."""

        def __init__(self, *args: Any, direct_rpm_delta_scale: float, **kwargs: Any) -> None:
            """Initialize the direct-RPM hover aviary with a validated scale."""
            self.DIRECT_RPM_DELTA_SCALE = envs.actions.validate_rpm_delta_scale(direct_rpm_delta_scale)
            super().__init__(*args, **kwargs)

        def _preprocessAction(self, action: Any) -> np.ndarray:  # noqa: N802 - upstream method name.
            """Convert normalized per-motor commands directly into motor RPMs."""
            self.action_buffer.append(np.clip(np.asarray(action, dtype=float), NORMALIZED_ACTION_LOW, NORMALIZED_ACTION_HIGH))
            return normalized_direct_rpm_to_motor_rpms(
                action,
                hover_rpm=_hover_rpm(self),
                rpm_delta_scale=self.DIRECT_RPM_DELTA_SCALE,
                rpm_min=_rpm_min(self),
                rpm_max=_rpm_max(self),
            )

    if action_interface == envs.actions.ActionInterface.DIRECT_RPM:
        base_env = ConfigurableDirectRPMHoverAviary(
            gui=gui,
            record=record,
            obs=ObservationType.KIN,
            act=ActionType.RPM,
            direct_rpm_delta_scale=rpm_delta_scale,
        )
    else:
        base_env = HoverAviary(gui=gui, record=record, obs=ObservationType.KIN, act=ActionType.PID)
    if episode_len_sec is not None:
        if not np.isfinite(float(episode_len_sec)) or float(episode_len_sec) <= 0.0:
            message = "episode_len_sec must be finite and positive when provided"
            raise ValueError(message)
        base_env.EPISODE_LEN_SEC = float(episode_len_sec)
    return base_env


def _hover_rpm(base_env: Any) -> float:
    """Return the finite hover RPM from the upstream drone model."""
    hover_rpm = float(getattr(base_env, "HOVER_RPM", 0.0))
    if not np.isfinite(hover_rpm) or hover_rpm <= 0.0:
        message = "base environment must expose a finite positive HOVER_RPM for direct_rpm"
        raise RuntimeError(message)
    return hover_rpm


def _rpm_min(_base_env: Any) -> float:
    """Return the minimum physically valid motor RPM."""
    return 0.0


def _rpm_max(base_env: Any) -> float:
    """Return the finite maximum motor RPM from the upstream drone model."""
    max_rpm = float(getattr(base_env, "MAX_RPM", 0.0))
    if not np.isfinite(max_rpm) or max_rpm <= 0.0:
        message = "base environment must expose a finite positive MAX_RPM for direct_rpm"
        raise RuntimeError(message)
    return max_rpm


def _coerce_task_reference(
    task: Any,
    limits: validation.tasks.ValidationLimits | None,
) -> envs.task_adapter.EnvironmentTaskReference:
    """Return an environment task reference, validating mappings as needed."""
    if isinstance(task, envs.task_adapter.EnvironmentTaskReference):
        return task
    if isinstance(task, Mapping):
        return envs.task_adapter.make_task_reference(task=task, limits=limits)
    message = "task must be a mapping, task reference, or task-distribution settings"
    raise TypeError(message)


__all__ = [
    "TrajectoryTrackingEnv",
    "make_normalized_action_env",
    "make_trajectory_tracking_env",
    "normalized_direct_rpm_to_motor_rpms",
    "normalized_to_real_action",
    "real_to_normalized_action",
]
