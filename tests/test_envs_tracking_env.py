"""Tests for the minimal Gymnasium trajectory-tracking environment."""

# ruff: noqa: S101

from __future__ import annotations

import json

import numpy as np
import pytest

from src import envs, utils, validation

BASE_OBSERVATION_DIM = 10
DYNAMICS_OBSERVATION_DIM = 19
PID_PREVIOUS_ACTION_OBSERVATION_DIM = 13
DIRECT_RPM_DYNAMICS_PREVIOUS_ACTION_OBSERVATION_DIM = 23
PID_ACTION_DIM = 3
DIRECT_RPM_ACTION_DIM = 4
UPSTREAM_DEFAULT_SPAWN_REFERENCE_ERROR_MIN_M = 0.2
RANDOM_OFFSET_XY_MIN_M = 0.10
RANDOM_OFFSET_XY_MAX_M = 0.30
RANDOM_OFFSET_Z_MIN_M = -0.18
RANDOM_OFFSET_Z_MAX_M = 0.08
RANDOM_OFFSET_BELOW_PROBABILITY = 0.70


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


def _line_task(start: list[float] | None = None, end: list[float] | None = None) -> dict[str, object]:
    """Return a valid line task for initial-state tests."""
    contracts = validation.contracts
    return {
        contracts.FIELD_TASK_TYPE: contracts.TASK_TYPE_TRAJECTORY,
        contracts.FIELD_SHAPE: contracts.SHAPE_LINE,
        contracts.FIELD_DURATION_SEC: 3.0,
        contracts.FIELD_SAMPLE_RATE_HZ: 5.0,
        contracts.FIELD_START: [0.0, 0.0, 1.0] if start is None else start,
        contracts.FIELD_END: [0.4, 0.0, 1.0] if end is None else end,
    }


def _assert_xyz_close(actual: object, expected: object, *, atol: float = 1.0e-5) -> None:
    """Assert two XYZ-like values are numerically close."""
    assert np.allclose(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), atol=atol, rtol=0.0)


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


def test_default_initial_state_preserves_upstream_near_ground_spawn() -> None:
    """Verify default initial-state mode leaves HoverAviary's upstream spawn alone."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(_hover_task(), gui=False, record=False)
    try:
        _, info = tracking_env.reset(seed=0)

        actual_initial_xyz = np.asarray(info["actual_initial_xyz"], dtype=float)
        reference_xyz = np.asarray(info["initial_reference_xyz"], dtype=float)
        assert info["initial_state_mode"] == "default"
        assert info["initial_xyz_source"] == "upstream_default"
        assert info["spawned_at_reference_start"] is False
        assert info["initial_xyz_matches_reference_start"] is False
        assert actual_initial_xyz[2] < reference_xyz[2] - UPSTREAM_DEFAULT_SPAWN_REFERENCE_ERROR_MIN_M
        assert info["initial_position_error_m"] > UPSTREAM_DEFAULT_SPAWN_REFERENCE_ERROR_MIN_M
    finally:
        tracking_env.close()


def test_fixed_initial_state_sets_hover_initial_xyzs_and_reset_info() -> None:
    """Verify fixed initial-state mode passes a configured XYZ into HoverAviary."""
    fixed_xyz = [0.1, -0.2, 1.2]
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        initial_state={"mode": "fixed", "xyz": fixed_xyz},
    )
    try:
        _, info = tracking_env.reset(seed=0)

        assert info["initial_state_mode"] == "fixed"
        assert info["initial_xyz_source"] == "configured_xyz"
        _assert_xyz_close(info["requested_initial_xyz"], fixed_xyz)
        _assert_xyz_close(info["actual_initial_xyz"], fixed_xyz)
        assert np.allclose(tracking_env.base_env.INIT_XYZS, np.asarray([fixed_xyz], dtype=float))
        assert info["spawned_at_reference_start"] is False
    finally:
        tracking_env.close()


def test_reference_start_initial_state_spawns_at_first_reference_position() -> None:
    """Verify reference-start mode resets the drone at the first reference point."""
    reference_start = [0.25, -0.15, 1.0]
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _line_task(start=reference_start, end=[0.55, -0.15, 1.0]),
        gui=False,
        record=False,
        initial_state={"mode": "reference_start"},
    )
    try:
        _, info = tracking_env.reset(seed=0)

        assert info["initial_state_mode"] == "reference_start"
        assert info["initial_xyz_source"] == "reference_start"
        assert info["spawned_at_reference_start"] is True
        assert info["initial_xyz_matches_reference_start"] is True
        assert info["initial_position_error_m"] == pytest.approx(0.0, abs=1.0e-6)
        _assert_xyz_close(info["initial_reference_xyz"], reference_start)
        _assert_xyz_close(info["actual_initial_xyz"], reference_start)
        assert np.allclose(tracking_env.base_env.INIT_XYZS, np.asarray([reference_start], dtype=float))
    finally:
        tracking_env.close()


def test_reference_start_with_offset_initial_state_records_offset_error() -> None:
    """Verify reference-start-with-offset mode applies the configured offset."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        initial_state={"mode": "reference_start_with_offset", "offset_xyz": [0.0, 0.0, -0.05]},
    )
    try:
        _, info = tracking_env.reset(seed=0)

        expected_xyz = [0.0, 0.0, 0.95]
        assert info["initial_state_mode"] == "reference_start_with_offset"
        assert info["initial_xyz_source"] == "reference_start_with_offset"
        _assert_xyz_close(info["requested_initial_xyz"], expected_xyz)
        _assert_xyz_close(info["actual_initial_xyz"], expected_xyz)
        assert info["initial_z_error_signed_m"] == pytest.approx(-0.05)
        assert info["initial_z_error_m"] == pytest.approx(0.05)
        assert info["spawned_at_reference_start"] is False
    finally:
        tracking_env.close()


def test_reference_start_random_offset_spawns_near_reference_and_reports_json_safe_metadata() -> None:
    """Verify randomized reference-start mode spawns near but not exactly at the first reference."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        initial_state={
            "mode": "reference_start_random_offset",
            "xy_offset_range_m": [0.10, 0.30],
            "z_offset_range_m": [-0.18, 0.08],
            "z_offset_bias": "below",
            "below_probability": 0.70,
        },
    )
    try:
        _, info = tracking_env.reset(seed=0)

        reference_xyz = np.asarray(info["initial_reference_xyz"], dtype=float)
        actual_xyz = np.asarray(info["actual_initial_xyz"], dtype=float)
        offset_xyz = actual_xyz - reference_xyz
        xy_offset_m = float(np.linalg.norm(offset_xyz[:2]))
        offset_distance_m = float(np.linalg.norm(offset_xyz))
        assert info["initial_state_mode"] == "reference_start_random_offset"
        assert info["initial_xyz_source"] == "reference_start_random_offset"
        assert info["spawned_near_reference_start"] is True
        assert info["spawned_at_reference_start"] is False
        assert info["initial_xyz_matches_reference_start"] is False
        assert RANDOM_OFFSET_XY_MIN_M <= xy_offset_m <= RANDOM_OFFSET_XY_MAX_M
        assert RANDOM_OFFSET_Z_MIN_M <= offset_xyz[2] <= RANDOM_OFFSET_Z_MAX_M
        assert np.all(np.abs(actual_xyz[:2]) <= envs.initial_state.DEFAULT_INITIAL_STATE_MAX_ABS_XY_M)
        assert envs.initial_state.DEFAULT_INITIAL_STATE_MIN_Z_M <= actual_xyz[2] <= envs.initial_state.DEFAULT_INITIAL_STATE_MAX_Z_M
        assert info["real_pid_z_target_low"] <= actual_xyz[2] <= info["real_pid_z_target_high"]
        assert info["initial_xy_offset_m"] == pytest.approx(xy_offset_m)
        assert info["initial_z_offset_m"] == pytest.approx(offset_xyz[2])
        assert info["initial_offset_distance_m"] == pytest.approx(offset_distance_m)
        assert info["initial_position_error_m"] == pytest.approx(offset_distance_m)
        assert info["initial_z_error_m"] == pytest.approx(abs(offset_xyz[2]))
        assert info["initial_offset_seed"] == 0
        assert info["initial_offset_sample_index"] == 0
        assert info["initial_offset_policy"]["xy_offset_range_m"] == [0.10, 0.30]
        assert info["initial_offset_policy"]["z_offset_range_m"] == [-0.18, 0.08]
        assert info["initial_offset_policy"]["z_offset_bias"] == "below"
        assert info["initial_offset_policy"]["below_probability"] == pytest.approx(RANDOM_OFFSET_BELOW_PROBABILITY)
        assert info["initial_offset_policy"]["z_sampling"] == "below_biased_uniform_subranges"
        initial_state_fields = {
            key: value
            for key, value in info.items()
            if key.startswith("initial_") or key in {"spawned_at_reference_start", "spawned_near_reference_start"}
        }
        assert utils.serialization.find_non_jsonable_paths(initial_state_fields) == []
        json.dumps(initial_state_fields, allow_nan=False)
    finally:
        tracking_env.close()


def test_reference_start_random_offset_sequence_is_seed_reproducible_and_varies_by_episode() -> None:
    """Verify one seed reproduces a varied randomized-offset reset sequence."""

    def offset_sequence() -> tuple[list[list[float]], list[int]]:
        tracking_env = envs.tracking_env.make_trajectory_tracking_env(
            _hover_task(),
            gui=False,
            record=False,
            initial_state={"mode": "reference_start_random_offset"},
        )
        try:
            infos = []
            _, info = tracking_env.reset(seed=123)
            infos.append(info)
            for _ in range(2):
                _, info = tracking_env.reset()
                infos.append(info)
            return [list(info["initial_xyz_offset"]) for info in infos], [int(info["initial_offset_sample_index"]) for info in infos]
        finally:
            tracking_env.close()

    first_offsets, first_indices = offset_sequence()
    second_offsets, second_indices = offset_sequence()

    assert np.allclose(first_offsets, second_offsets)
    assert first_indices == [0, 1, 2]
    assert second_indices == [0, 1, 2]
    assert any(not np.allclose(first_offsets[0], offset) for offset in first_offsets[1:])


def test_reference_start_random_offset_sampler_varies_direction_and_biases_below() -> None:
    """Verify random offsets vary direction while biasing z starts below the reference."""
    config = envs.initial_state.InitialStateConfig(mode="reference_start_random_offset")
    offsets = [
        envs.initial_state.resolve_initial_state(
            config,
            [0.0, 0.0, 1.0],
            rng=np.random.default_rng(seed),
            offset_seed=seed,
        ).initial_xyz_offset
        for seed in range(64)
    ]
    z_offsets = [offset[2] for offset in offsets]
    angles = [float(np.arctan2(offset[1], offset[0])) for offset in offsets]

    assert any(offset > 0.0 for offset in z_offsets)
    assert any(offset < 0.0 for offset in z_offsets)
    assert sum(offset < 0.0 for offset in z_offsets) > sum(offset > 0.0 for offset in z_offsets)
    assert max(angles) - min(angles) > 1.0


def test_reference_start_random_offset_rank_seed_changes_sequence_deterministically() -> None:
    """Verify rank-derived seeds give deterministic but different offset sequences."""

    def sequence(seed: int) -> list[list[float]]:
        tracking_env = envs.tracking_env.make_trajectory_tracking_env(
            _hover_task(),
            gui=False,
            record=False,
            initial_state={"mode": "reference_start_random_offset"},
        )
        try:
            offsets = []
            _, info = tracking_env.reset(seed=seed)
            offsets.append(list(info["initial_xyz_offset"]))
            _, info = tracking_env.reset()
            offsets.append(list(info["initial_xyz_offset"]))
            return offsets
        finally:
            tracking_env.close()

    rank_zero = sequence(seed=41)
    rank_zero_repeat = sequence(seed=41)
    rank_one = sequence(seed=42)

    assert np.allclose(rank_zero, rank_zero_repeat)
    assert not np.allclose(rank_zero, rank_one)


def test_invalid_reference_start_random_offset_ranges_raise_clear_errors() -> None:
    """Verify invalid randomized-offset ranges fail during config parsing."""
    with pytest.raises(ValueError, match="xy_offset_range_m"):
        envs.initial_state.parse_initial_state_config({"mode": "reference_start_random_offset", "xy_offset_range_m": [0.30, 0.10]})
    with pytest.raises(ValueError, match="z_offset_range_m"):
        envs.initial_state.parse_initial_state_config({"mode": "reference_start_random_offset", "z_offset_range_m": [0.08, -0.18]})
    with pytest.raises(ValueError, match="z_offset_bias"):
        envs.initial_state.parse_initial_state_config({"mode": "reference_start_random_offset", "z_offset_bias": "up"})
    with pytest.raises(ValueError, match="below_probability"):
        envs.initial_state.parse_initial_state_config({"mode": "reference_start_random_offset", "below_probability": 1.1})
    with pytest.raises(ValueError, match="nonzero"):
        envs.initial_state.parse_initial_state_config(
            {"mode": "reference_start_random_offset", "xy_offset_range_m": [0.0, 0.0], "z_offset_range_m": [0.0, 0.0]}
        )


def test_invalid_initial_state_mode_and_xyz_raise_clear_errors() -> None:
    """Verify invalid modes and malformed XYZ values are rejected clearly."""
    with pytest.raises(ValueError, match=r"initial_state\.mode"):
        envs.initial_state.parse_initial_state_config({"mode": "unsupported_start"})
    with pytest.raises(ValueError, match=r"initial_state\.xyz"):
        envs.initial_state.parse_initial_state_config({"mode": "fixed", "xyz": [0.0, 0.0]})


def test_reference_start_initial_state_is_supported_for_direct_rpm() -> None:
    """Verify direct-RPM envs preserve motor semantics while using reference-start spawn."""
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        action_interface="direct_rpm",
        initial_state={"mode": "reference_start"},
    )
    try:
        _, info = tracking_env.reset(seed=0)

        assert info["action_interface"] == "direct_rpm"
        assert info["real_action_type"] == "motor_rpm"
        assert info["spawned_at_reference_start"] is True
        _assert_xyz_close(info["actual_initial_xyz"], info["initial_reference_xyz"])
        assert tracking_env.base_env.ACT_TYPE.value == "rpm"
    finally:
        tracking_env.close()


def test_reference_start_initial_state_updates_after_randomized_reset() -> None:
    """Verify randomized task reset samples the reference before updating INIT_XYZS."""
    settings = envs.task_distribution.TaskDistributionSettings(
        name="randomized_line_initial_state",
        enabled=True,
        mode=envs.task_distribution.MODE_RANDOMIZED,
        seed=7,
        strength=1.0,
        sample_on_reset=True,
        base_task=_line_task(),
        family_weights={envs.task_distribution.FAMILY_LINE: 1.0},
        variations={
            envs.task_distribution.FAMILY_LINE: {
                "start_xy_radius_m": 0.35,
                "heading_jitter_deg": 90.0,
                "length_range_m": [0.3, 0.45],
                "base_z_range_m": [0.9, 1.1],
                "duration_range_sec": [3.0, 3.0],
                "start_hold_range_sec": [1.0, 1.0],
            }
        },
    )
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        settings,
        gui=False,
        record=False,
        initial_state={"mode": "reference_start"},
    )
    try:
        _, first_info = tracking_env.reset(seed=0)
        _, second_info = tracking_env.reset(seed=1)

        first_reference = np.asarray(first_info["initial_reference_xyz"], dtype=float)
        second_reference = np.asarray(second_info["initial_reference_xyz"], dtype=float)
        assert not np.allclose(first_reference, second_reference)
        for info in (first_info, second_info):
            assert info["task_distribution_sample_on_reset"] is True
            assert info["spawned_at_reference_start"] is True
            _assert_xyz_close(info["actual_initial_xyz"], info["initial_reference_xyz"])
            _assert_xyz_close(info["requested_initial_xyz"], info["initial_reference_xyz"])
        assert np.allclose(tracking_env.base_env.INIT_XYZS, second_reference.reshape(1, 3))
    finally:
        tracking_env.close()


def test_random_offset_initial_state_updates_after_randomized_task_reset() -> None:
    """Verify sampled task references are used before randomizing initial XYZ on each reset."""
    settings = envs.task_distribution.TaskDistributionSettings(
        name="randomized_line_initial_state_offset",
        enabled=True,
        mode=envs.task_distribution.MODE_RANDOMIZED,
        seed=7,
        strength=1.0,
        sample_on_reset=True,
        base_task=_line_task(),
        family_weights={envs.task_distribution.FAMILY_LINE: 1.0},
        variations={
            envs.task_distribution.FAMILY_LINE: {
                "start_xy_radius_m": 0.35,
                "heading_jitter_deg": 90.0,
                "length_range_m": [0.3, 0.45],
                "base_z_range_m": [0.9, 1.1],
                "duration_range_sec": [3.0, 3.0],
                "start_hold_range_sec": [1.0, 1.0],
            }
        },
    )
    tracking_env = envs.tracking_env.make_trajectory_tracking_env(
        settings,
        gui=False,
        record=False,
        initial_state={
            "mode": "reference_start_random_offset",
            "xy_offset_range_m": [0.10, 0.30],
            "z_offset_range_m": [-0.18, 0.08],
            "z_offset_bias": "below",
            "below_probability": 0.70,
        },
    )
    try:
        _, first_info = tracking_env.reset(seed=0)
        _, second_info = tracking_env.reset()

        first_reference = np.asarray(first_info["initial_reference_xyz"], dtype=float)
        second_reference = np.asarray(second_info["initial_reference_xyz"], dtype=float)
        second_requested = np.asarray(second_info["requested_initial_xyz"], dtype=float)
        assert not np.allclose(first_reference, second_reference)
        assert not np.allclose(first_info["initial_xyz_offset"], second_info["initial_xyz_offset"])
        for info in (first_info, second_info):
            reference = np.asarray(info["initial_reference_xyz"], dtype=float)
            requested = np.asarray(info["requested_initial_xyz"], dtype=float)
            offset = np.asarray(info["initial_xyz_offset"], dtype=float)
            assert info["task_distribution_sample_on_reset"] is True
            assert info["spawned_near_reference_start"] is True
            assert info["spawned_at_reference_start"] is False
            assert np.allclose(requested, reference + offset)
        assert np.allclose(tracking_env.base_env.INIT_XYZS, second_requested.reshape(1, 3))
    finally:
        tracking_env.close()


def test_invalid_initial_state_config_raises_clear_error() -> None:
    """Verify invalid initial-state settings fail before simulator construction."""
    with pytest.raises(ValueError, match=r"initial_state\.xyz is required"):
        envs.tracking_env.make_trajectory_tracking_env(_hover_task(), gui=False, record=False, initial_state={"mode": "fixed"})


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
        assert "final_hold_enabled" in info
        assert "final_hold_sec" in info
        assert "tracking_phase_end_step" in info
        assert "tracking_phase_end_time_sec" in info
        assert "is_start_hold" in info
        assert "is_final_hold" in info
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
        assert info["final_hold_enabled"] is True
        assert info["final_hold_sec"] == pytest.approx(1.0)
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
    """Verify default PID PPO actions are normalized and map through expanded z bounds."""
    real_env = envs.tracking_env.make_trajectory_tracking_env(_hover_task(), gui=False, record=False)
    tracking_env = envs.tracking_env.make_normalized_action_env(real_env)
    try:
        tracking_env.reset(seed=0)
        hover_target = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        expected_low = np.array([[-0.2, -0.2, 0.2]], dtype=np.float32)
        expected_high = np.array([[0.2, 0.2, 1.5]], dtype=np.float32)
        normalized_hover = tracking_env.real_to_normalized_action(hover_target)

        assert tracking_env.action_interface == "pid_position"
        assert tracking_env.action_space.shape == (1, PID_ACTION_DIM)
        assert np.allclose(tracking_env.action_space.low, -1.0)
        assert np.allclose(tracking_env.action_space.high, 1.0)
        assert np.allclose(tracking_env.real_action_space.low, expected_low)
        assert np.allclose(tracking_env.real_action_space.high, expected_high)
        assert normalized_hover[0, 2] == pytest.approx(0.23076923)
        assert np.allclose(tracking_env.normalized_to_real_action(normalized_hover), hover_target)

        _, _, _, _, info = tracking_env.step(normalized_hover)

        assert np.allclose(info["normalized_action"], normalized_hover)
        assert np.allclose(info["requested_action"], hover_target)
        assert np.allclose(info["applied_action"], hover_target)
        assert np.allclose(info["real_action"], hover_target)
        assert info["reference_z_reachable_by_pid_position"] is True
        assert info["pid_z_action_space_expanded"] is True
    finally:
        tracking_env.close()


def test_pid_position_normalized_z_extremes_map_to_real_z_bounds() -> None:
    """Verify PID action dim 2 maps monotonically between real z target bounds."""
    real_env = envs.tracking_env.make_trajectory_tracking_env(_hover_task(), gui=False, record=False)
    tracking_env = envs.tracking_env.make_normalized_action_env(real_env)
    try:
        tracking_env.reset(seed=0)
        normalized_low_z = np.array([[0.0, 0.0, -1.0]], dtype=np.float32)
        normalized_high_z = np.array([[0.0, 0.0, 1.0]], dtype=np.float32)
        real_low_z = tracking_env.normalized_to_real_action(normalized_low_z)
        real_high_z = tracking_env.normalized_to_real_action(normalized_high_z)

        assert real_low_z[0, 2] == pytest.approx(float(tracking_env.real_action_space.low[0, 2]))
        assert real_high_z[0, 2] == pytest.approx(float(tracking_env.real_action_space.high[0, 2]))
        assert real_high_z[0, 2] > real_low_z[0, 2]
        assert np.allclose(tracking_env.real_to_normalized_action(real_low_z), normalized_low_z)
        assert np.allclose(tracking_env.real_to_normalized_action(real_high_z), normalized_high_z)
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


def test_normalized_wrapper_preserves_direct_rpm_action_metadata() -> None:
    """Verify wrapped direct-RPM traces keep motor semantics and RPM real actions."""
    real_env = envs.tracking_env.make_trajectory_tracking_env(
        _hover_task(),
        gui=False,
        record=False,
        action_interface="direct_rpm",
        include_dynamics_observation=True,
    )
    tracking_env = envs.tracking_env.make_normalized_action_env(real_env)
    try:
        tracking_env.reset(seed=0)
        action = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

        _, _, _, _, info = tracking_env.step(action)

        assert tracking_env.action_interface == "direct_rpm"
        assert info["action_interface"] == "direct_rpm"
        assert info["real_action_type"] == "motor_rpm"
        assert np.allclose(info["normalized_action"], action)
        assert np.asarray(info["real_action"]).shape == (1, DIRECT_RPM_ACTION_DIM)
        assert np.asarray(info["real_action_space_low"])[0, 0] == pytest.approx(0.0)
        assert info["real_action"][0, 2] > info["real_action"][0, 0]
        assert not np.allclose(info["real_action"], action)
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
        assert info["start_hold_reward_policy"] == "full_tracking_reward_active_during_uniform_reference_start_hold"
        assert info["tracking_reward_starts_after_start_hold"] is False
    finally:
        tracking_env.close()
