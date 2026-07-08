"""Tests for the minimal Gymnasium trajectory-tracking environment."""

# ruff: noqa: S101

from __future__ import annotations

import numpy as np
import pytest

from src import envs, validation


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

        assert observation.shape == (10,)
        assert observation.dtype == np.float32
        assert tracking_env.observation_space.contains(observation)
        assert isinstance(info, dict)
    finally:
        tracking_env.close()


def test_tracking_env_steps_once_with_sampled_action_and_diagnostics() -> None:
    """Verify one valid base action advances the wrapper and returns diagnostics."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(_hover_task(), gui=False, record=False)
    try:
        tracking_env.reset(seed=0)
        action = tracking_env.action_space.sample()
        observation, reward, terminated, truncated, info = tracking_env.step(action)

        assert observation.shape == (10,)
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

        assert observation.shape == (10,)
        assert tracking_env.observation_space.contains(observation)
        assert info["task_shape"] == validation.contracts.SHAPE_CIRCLE
        assert info["start_hold_enabled"] is True
        assert info["is_start_hold"] is True
        expected_tracking_phase_start_step = 10
        assert info["tracking_phase_start_step"] == expected_tracking_phase_start_step
    finally:
        tracking_env.close()
