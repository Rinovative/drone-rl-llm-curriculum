"""Tests for the minimal Gymnasium trajectory-tracking environment."""

# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pytest

from src import envs, validation

BASE_OBSERVATION_DIM = 10
DYNAMICS_OBSERVATION_DIM = 19
PID_PREVIOUS_ACTION_OBSERVATION_DIM = 13
DIRECT_RPM_DYNAMICS_PREVIOUS_ACTION_OBSERVATION_DIM = 23
PID_ACTION_DIM = 3
DIRECT_RPM_ACTION_DIM = 4


def _hover_task() -> dict[str, object]:
    """Return a valid hover task for tracking environment tests."""
    contracts = validation.contracts
    return {
        contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
        contracts.FIELD_SHAPE: contracts.SHAPE_HOVER,
        contracts.FIELD_DURATION_SEC: 2.0,
        contracts.FIELD_SAMPLE_RATE_HZ: 5.0,
        contracts.FIELD_POSITION: [0.0, 0.0, 1.0],
    }


def _circle_task() -> dict[str, object]:
    """Return a valid circle task for tracking environment tests."""
    contracts = validation.contracts
    return {
        contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
        contracts.FIELD_SHAPE: contracts.SHAPE_CIRCLE,
        contracts.FIELD_DURATION_SEC: 8.0,
        contracts.FIELD_SAMPLE_RATE_HZ: 10.0,
        contracts.FIELD_RADIUS: 0.5,
        contracts.FIELD_HEIGHT: 1.0,
        contracts.FIELD_CENTER: [0.0, 0.0],
    }


def test_tracking_env_imports_through_package_alias() -> None:
    """Verify tracking environment helpers are exposed by the envs package."""
    assert envs.tracking_env.TrajectoryTrackingEnv is not None
    assert envs.tracking_env.make_trajectory_tracking_env is not None


def test_tracking_env_reset_returns_compact_observation_and_info() -> None:
    """Verify reset returns the requested compact observation contract."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(_hover_task(), gui=False, record=False)
    try:
        observation, info = tracking_env.reset(seed=0)

        assert observation.shape == (BASE_OBSERVATION_DIM,)
        assert observation.dtype == np.float32
        assert tracking_env.observation_space.contains(observation)
        assert isinstance(info, dict)
        assert info["action_interface"] == "pid_position"
        assert info["real_action_type"] == "pid_target_position"
        assert info["ppo_action_dim"] == PID_ACTION_DIM
        assert info["include_dynamics_observation"] is False
        assert info["include_previous_action"] is False
        assert info["observation_dim"] == BASE_OBSERVATION_DIM
        assert info["termination_limits_mode"] == "default"
        assert info["termination_limits"]["terminate_on_base_truncation"] is True
        assert info["diagnostic_limits"]["mode"] == "default"
        assert info["base_truncation_policy"] == "terminate"
        assert info["strict_limit_violation_count"] == 0
        assert [component["name"] for component in info["observation_components"]] == [
            "current_position",
            "reference_position",
            "position_error",
            "trajectory_progress",
        ]
    finally:
        tracking_env.close()


def test_tracking_env_steps_once_with_sampled_action_and_diagnostics() -> None:
    """Verify one valid base action advances the wrapper and returns diagnostics."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(_hover_task(), gui=False, record=False)
    try:
        tracking_env.reset(seed=0)
        action = tracking_env.action_space.sample()
        observation, reward, terminated, truncated, info = tracking_env.step(action)

        assert observation.shape == (BASE_OBSERVATION_DIM,)
        assert tracking_env.observation_space.contains(observation)
        assert np.isfinite(reward)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "position_error_m" in info
        assert "reference_position" in info
        assert "current_position" in info
        assert "task_shape" in info
        assert "tracking_success" in info
        assert "reference_step_index" in info
        assert "reference_time_sec" in info
        assert "start_hold_enabled" in info
        assert "tracking_phase_start_step" in info
        assert "tracking_phase_start_time_sec" in info
        assert "is_start_hold" in info
        assert "is_tracking_phase" in info
        assert "base_reward" in info
        assert info["base_action_shape"] == action.shape
        assert info["base_action_dtype"] == str(action.dtype)
        assert info["action_shape"] == action.shape
        assert info["action_dtype"] == str(action.dtype)
        assert info["base_terminated"] is terminated or info["base_terminated"] is False
        assert info["base_truncated"] is truncated
        assert info["base_info_keys"] == ["answer"]
        assert info["base_reason_fields"] == {}
        assert isinstance(info["termination_reason"], str)
        assert info["task_shape"] == validation.contracts.SHAPE_HOVER
        assert np.asarray(info["reference_position"]).shape == (3,)
        assert np.asarray(info["current_position"]).shape == (3,)
        assert np.asarray(info["roll_pitch_yaw"]).shape == (3,)
        assert np.asarray(info["velocity"]).shape == (3,)
        assert np.asarray(info["angular_velocity"]).shape == (3,)
        assert np.asarray(info["last_action"]).shape == (4,)
        assert np.asarray(info["requested_action"]).shape == action.shape
    finally:
        tracking_env.close()


def test_relaxed_termination_ignores_upstream_pitch_truncation_but_reports_it(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify relaxed training can continue through strict upstream pitch truncation."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        termination_limits={"mode": "relaxed"},
        diagnostic_limits={"mode": "default"},
    )
    try:
        tracking_env.reset(seed=0)
        state = np.zeros(20, dtype=float)
        state[:3] = [0.0, 0.0, 1.0]
        state[8] = 0.5
        monkeypatch.setattr(tracking_env.base_env, "step", lambda _action: (None, 0.0, False, True, {}))
        monkeypatch.setattr(tracking_env, "_current_state_vector", lambda: np.array(state, dtype=float, copy=True))

        _, _, terminated, truncated, info = tracking_env.step(np.zeros(tracking_env.action_space.shape, dtype=np.float32))

        assert terminated is False
        assert truncated is False
        assert info["base_truncated"] is True
        assert info["base_truncation_ignored"] is True
        assert info["base_truncation_policy"] == "diagnose_only"
        assert info["strict_limit_violation"] is True
        assert info["strict_limit_violations"] == ["pitch_above_limit"]
        assert info["strict_limit_violation_count"] == 1
        assert info["termination_reason"] == "running"
    finally:
        tracking_env.close()


def test_relaxed_project_limits_truncate_after_recovery_window(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify relaxed mode still truncates unrecovered hard-limit violations."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        termination_limits={
            "mode": "custom",
            "max_roll_pitch_rad": 0.45,
            "allow_recovery_steps": 1,
            "terminate_on_base_truncation": False,
        },
        diagnostic_limits={"mode": "default"},
    )
    try:
        tracking_env.reset(seed=0)
        state = np.zeros(20, dtype=float)
        state[:3] = [0.0, 0.0, 1.0]
        state[8] = 0.5
        monkeypatch.setattr(tracking_env.base_env, "step", lambda _action: (None, 0.0, False, False, {}))
        monkeypatch.setattr(tracking_env, "_current_state_vector", lambda: np.array(state, dtype=float, copy=True))
        action = np.zeros(tracking_env.action_space.shape, dtype=np.float32)

        _, _, _, first_truncated, first_info = tracking_env.step(action)
        _, _, _, second_truncated, second_info = tracking_env.step(action)

        assert first_truncated is False
        assert first_info["recovery_allowed_after_limit_violation"] is True
        assert second_truncated is True
        assert second_info["project_truncated"] is True
        assert second_info["project_truncation_causes"] == ["pitch_above_limit"]
        assert second_info["termination_reason"] == "project_truncated:pitch_above_limit"
    finally:
        tracking_env.close()


def test_tracking_env_dynamics_observation_extends_shape() -> None:
    """Verify dynamics observation adds velocity, attitude, and angular velocity."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        include_dynamics_observation=True,
    )
    try:
        observation, info = tracking_env.reset(seed=0)

        assert observation.shape == (DYNAMICS_OBSERVATION_DIM,)
        assert tracking_env.observation_space.contains(observation)
        assert info["include_dynamics_observation"] is True
        assert info["include_previous_action"] is False
        assert info["observation_dim"] == DYNAMICS_OBSERVATION_DIM
        assert [component["name"] for component in info["observation_components"]][-3:] == [
            "linear_velocity",
            "attitude_rpy",
            "angular_velocity",
        ]
    finally:
        tracking_env.close()


def test_tracking_env_previous_action_observation_resets_and_updates() -> None:
    """Verify previous-action observations start at zero and update after step."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        include_previous_action=True,
    )
    try:
        observation, info = tracking_env.reset(seed=0)
        action = tracking_env.action_space.sample()

        assert observation.shape == (PID_PREVIOUS_ACTION_OBSERVATION_DIM,)
        assert np.allclose(observation[-PID_ACTION_DIM:], 0.0)
        assert np.allclose(info["previous_action"], np.zeros((1, PID_ACTION_DIM)))
        assert info["include_previous_action"] is True
        assert info["observation_dim"] == PID_PREVIOUS_ACTION_OBSERVATION_DIM

        next_observation, _, _, _, step_info = tracking_env.step(action)

        assert next_observation.shape == (PID_PREVIOUS_ACTION_OBSERVATION_DIM,)
        assert np.allclose(next_observation[-PID_ACTION_DIM:], np.asarray(action).reshape(-1))
        assert np.allclose(step_info["previous_action"], action)
        assert step_info["observation_components"][-1] == {"name": "previous_action", "dim": PID_ACTION_DIM}
    finally:
        tracking_env.close()


def test_normalized_action_wrapper_previous_action_uses_ppo_facing_action() -> None:
    """Verify PID normalized wrapper appends the normalized policy action, not real PID targets."""
    real_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        include_previous_action=True,
    )
    tracking_env = envs.tracking_env.make_normalized_action_env(real_env)
    try:
        observation, _ = tracking_env.reset(seed=0)
        action = np.array([[0.25, -0.5, 0.75]], dtype=np.float32)

        assert np.allclose(observation[-PID_ACTION_DIM:], 0.0)
        next_observation, _, _, _, info = tracking_env.step(action)

        assert np.allclose(next_observation[-PID_ACTION_DIM:], action.reshape(-1))
        assert np.allclose(info["previous_action"], action)
        assert not np.allclose(info["previous_action"], info["real_action"])
    finally:
        tracking_env.close()


def test_pid_position_default_ppo_action_space_and_hover_mapping() -> None:
    """Verify default PID PPO actions are normalized and map once into hover bounds."""
    real_env = envs.tracking_env.make_trajectory_tracking_env(_hover_task(), gui=False, record=False)
    tracking_env = envs.tracking_env.make_normalized_action_env(real_env)
    try:
        tracking_env.reset(seed=0)
        hover_target = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        expected_low = np.array([[-0.2, -0.2, 0.5]], dtype=np.float32)
        expected_high = np.array([[0.2, 0.2, 1.0]], dtype=np.float32)
        normalized_hover = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)

        assert tracking_env.action_interface == "pid_position"
        assert tracking_env.action_space.shape == (1, PID_ACTION_DIM)
        assert np.allclose(tracking_env.action_space.low, -1.0)
        assert np.allclose(tracking_env.action_space.high, 1.0)
        assert np.allclose(tracking_env.real_action_space.low, expected_low)
        assert np.allclose(tracking_env.real_action_space.high, expected_high)
        assert np.allclose(tracking_env.real_to_normalized_action(hover_target), normalized_hover)
        assert np.allclose(tracking_env.normalized_to_real_action(normalized_hover), hover_target)

        _, _, _, _, info = tracking_env.step(normalized_hover)

        assert np.allclose(info["normalized_action"], normalized_hover)
        assert np.allclose(info["requested_action"], hover_target)
        assert np.allclose(info["applied_action"], hover_target)
        assert np.allclose(info["real_action"], hover_target)
    finally:
        tracking_env.close()


def test_direct_rpm_env_exposes_normalized_four_motor_action_space() -> None:
    """Verify direct RPM uses normalized per-motor actions and dynamics observations."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        action_interface="direct_rpm",
        include_dynamics_observation=True,
    )
    try:
        observation, info = tracking_env.reset(seed=0)

        assert observation.shape == (DYNAMICS_OBSERVATION_DIM,)
        assert tracking_env.observation_space.contains(observation)
        assert tracking_env.action_space.shape == (1, 4)
        assert np.allclose(tracking_env.action_space.low, -1.0)
        assert np.allclose(tracking_env.action_space.high, 1.0)
        assert info["action_interface"] == "direct_rpm"
        assert info["real_action_type"] == "motor_rpm"
        assert info["ppo_action_dim"] == DIRECT_RPM_ACTION_DIM
        assert info["include_dynamics_observation"] is True
        assert info["include_previous_action"] is False
        assert info["observation_dim"] == DYNAMICS_OBSERVATION_DIM
        assert tracking_env.base_env.ACT_TYPE.value == "rpm"
        assert not hasattr(tracking_env.base_env, "ctrl")
    finally:
        tracking_env.close()


def test_direct_rpm_dynamics_previous_action_observation_shape() -> None:
    """Verify direct RPM can combine dynamics and previous-action observations."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        action_interface="direct_rpm",
        include_dynamics_observation=True,
        include_previous_action=True,
    )
    try:
        observation, info = tracking_env.reset(seed=0)
        action = np.array([[0.1, -0.2, 0.3, -0.4]], dtype=np.float32)

        assert observation.shape == (DIRECT_RPM_DYNAMICS_PREVIOUS_ACTION_OBSERVATION_DIM,)
        assert np.allclose(observation[-DIRECT_RPM_ACTION_DIM:], 0.0)
        assert info["include_dynamics_observation"] is True
        assert info["include_previous_action"] is True
        assert info["observation_dim"] == DIRECT_RPM_DYNAMICS_PREVIOUS_ACTION_OBSERVATION_DIM
        assert info["observation_components"][-1] == {"name": "previous_action", "dim": DIRECT_RPM_ACTION_DIM}

        next_observation, _, _, _, step_info = tracking_env.step(action)

        assert np.allclose(next_observation[-DIRECT_RPM_ACTION_DIM:], action.reshape(-1))
        assert np.allclose(step_info["previous_action"], action)
        assert not np.allclose(step_info["previous_action"], step_info["real_motor_rpms"])
    finally:
        tracking_env.close()


def test_direct_rpm_step_maps_normalized_actions_to_motor_rpms() -> None:
    """Verify direct RPM records normalized commands, real RPMs, and clipping."""
    rpm_delta_scale = 0.05
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        action_interface="direct_rpm",
        rpm_delta_scale=rpm_delta_scale,
        include_dynamics_observation=True,
    )
    try:
        tracking_env.reset(seed=0)
        requested_action = np.array([[2.0, -2.0, 0.0, 0.5]], dtype=np.float32)
        _, _, _, _, info = tracking_env.step(requested_action)

        normalized_action = np.array([[1.0, -1.0, 0.0, 0.5]], dtype=np.float32)
        hover_rpm = float(info["hover_rpm"])
        expected_rpms = envs.tracking_env.normalized_direct_rpm_to_motor_rpms(
            normalized_action,
            hover_rpm=hover_rpm,
            rpm_delta_scale=rpm_delta_scale,
            rpm_min=float(info["rpm_min"]),
            rpm_max=float(info["rpm_max"]),
        )

        assert info["action_interface"] == "direct_rpm"
        assert info["action_normalized"] is True
        assert info["action_clipped"] is True
        assert np.allclose(info["normalized_action"], normalized_action)
        assert np.allclose(info["real_motor_rpms"], expected_rpms)
        assert np.allclose(info["real_action"], expected_rpms)
        assert np.asarray(info["real_action_space_low"]).shape == (1, 4)
        assert np.asarray(info["real_action_space_high"]).shape == (1, 4)
        assert info["rpm_delta_scale"] == rpm_delta_scale
        assert info["rpm_clipped"] is False
    finally:
        tracking_env.close()


def test_direct_rpm_zero_and_monotonic_actions_map_around_hover_rpm() -> None:
    """Verify direct RPM maps normalized motor commands once around hover RPM."""
    rpm_delta_scale = 0.05
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        action_interface="direct_rpm",
        rpm_delta_scale=rpm_delta_scale,
        include_dynamics_observation=True,
    )
    try:
        tracking_env.reset(seed=0)
        zero_action = np.zeros((1, DIRECT_RPM_ACTION_DIM), dtype=np.float32)
        _, _, _, _, info = tracking_env.step(zero_action)

        hover_rpm = float(info["hover_rpm"])
        rpm_min = float(info["rpm_min"])
        rpm_max = float(info["rpm_max"])
        negative_rpms = envs.tracking_env.normalized_direct_rpm_to_motor_rpms(
            -np.ones((1, DIRECT_RPM_ACTION_DIM), dtype=np.float32),
            hover_rpm=hover_rpm,
            rpm_delta_scale=rpm_delta_scale,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
        )
        zero_rpms = envs.tracking_env.normalized_direct_rpm_to_motor_rpms(
            zero_action,
            hover_rpm=hover_rpm,
            rpm_delta_scale=rpm_delta_scale,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
        )
        positive_rpms = envs.tracking_env.normalized_direct_rpm_to_motor_rpms(
            np.ones((1, DIRECT_RPM_ACTION_DIM), dtype=np.float32),
            hover_rpm=hover_rpm,
            rpm_delta_scale=rpm_delta_scale,
            rpm_min=rpm_min,
            rpm_max=rpm_max,
        )

        assert tracking_env.action_space.shape == (1, DIRECT_RPM_ACTION_DIM)
        assert np.allclose(tracking_env.action_space.low, -1.0)
        assert np.allclose(tracking_env.action_space.high, 1.0)
        assert info["action_interface"] == "direct_rpm"
        assert info["real_action_type"] == "motor_rpm"
        assert info["action_normalized"] is True
        assert np.allclose(info["normalized_action"], zero_action)
        assert np.allclose(info["applied_action"], zero_action)
        assert np.allclose(info["real_motor_rpms"], hover_rpm)
        assert np.allclose(info["real_action"], zero_rpms)
        assert np.all(negative_rpms < zero_rpms)
        assert np.all(zero_rpms < positive_rpms)
        assert np.allclose(info["rpm_command_space_low"], negative_rpms)
        assert np.allclose(info["rpm_command_space_high"], positive_rpms)
        assert info["rpm_clipped"] is False
    finally:
        tracking_env.close()


def test_tracking_env_close_works_without_error() -> None:
    """Verify close can be called more than once without surfacing errors."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(_hover_task(), gui=False, record=False)

    tracking_env.close()
    tracking_env.close()


def test_tracking_env_invalid_task_raises_value_error() -> None:
    """Verify invalid mappings are rejected before simulator construction."""
    task = _hover_task()
    task[validation.contracts.FIELD_POSITION] = [3.0, 0.0, 1.0]

    with pytest.raises(ValueError, match="invalid trajectory task"):
        envs.tracking_env.make_trajectory_tracking_env(task, gui=False, record=False)


def test_circle_task_can_reset_headlessly() -> None:
    """Verify a non-hover trajectory can create and reset the wrapper."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(_circle_task(), gui=False, record=False)
    try:
        observation, info = tracking_env.reset(seed=0)

        assert observation.shape == (BASE_OBSERVATION_DIM,)
        assert tracking_env.observation_space.contains(observation)
        assert info["task_shape"] == validation.contracts.SHAPE_CIRCLE
        assert info["start_hold_enabled"] is True
        assert info["is_start_hold"] is True
        expected_tracking_phase_start_step = 10
        assert info["tracking_phase_start_step"] == expected_tracking_phase_start_step
    finally:
        tracking_env.close()
