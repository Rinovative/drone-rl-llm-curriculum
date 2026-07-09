"""Tests for structured trained-policy evaluation diagnostics."""

# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import evaluation

TRACE_RECORD_COUNT = 5


class _ActionSpace:
    """Tiny action-space stand-in exposing Box-like bounds."""

    low = np.array([[-1.0, -1.0, -1.0]], dtype=float)
    high = np.array([[1.0, 1.0, 1.0]], dtype=float)


class _DirectRPMActionSpace:
    """Tiny direct-RPM action-space stand-in exposing normalized motor bounds."""

    low = np.array([[-1.0, -1.0, -1.0, -1.0]], dtype=float)
    high = np.array([[1.0, 1.0, 1.0, 1.0]], dtype=float)


def _record(
    step_index: int,
    current_x: float,
    reference_x: float,
    action: list[float],
    task_shape: str = "line",
) -> dict[str, object]:
    """Return one complete policy evaluation trace record."""
    current_position = [current_x, 0.0, 1.0]
    reference_position = [reference_x, 0.0, 1.0]
    error_x = current_x - reference_x
    return {
        "step_index": step_index,
        "episode_index": 0,
        "episode_step_index": step_index,
        "reference_step_index": step_index,
        "time_sec": float(step_index),
        "current_position": current_position,
        "reference_position": reference_position,
        "actual_position_xyz_m": current_position,
        "reference_position_xyz_m": reference_position,
        "position_error_m": abs(error_x),
        "axis_error_xyz": [error_x, 0.0, 0.0],
        "error_xyz_m": [error_x, 0.0, 0.0],
        "velocity": [0.0, 0.0, 0.0],
        "angular_velocity": [0.0, 0.0, 0.0],
        "roll_pitch_yaw": [0.0, 0.0, 0.0],
        "action": [action],
        "last_action": [0.0, 0.0, 0.0, 0.0],
        "reward": -abs(error_x),
        "terminated": False,
        "truncated": False,
        "termination_reason": "running",
        "task_shape": task_shape,
        "start_hold_enabled": False,
        "start_hold_sec": 0.0,
        "exclude_start_hold_from_tracking_metrics": False,
        "tracking_phase_start_step": 0,
        "tracking_phase_start_time_sec": 0.0,
        "final_hold_enabled": False,
        "final_hold_sec": 0.0,
        "exclude_final_hold_from_tracking_metrics": False,
        "tracking_phase_end_step": 0,
        "tracking_phase_end_time_sec": 0.0,
        "is_start_hold": False,
        "is_final_hold": False,
        "is_tracking_phase": True,
        "base_terminated": False,
        "base_truncated": False,
        "base_truncation_causes": [],
        "base_reason_fields": {},
    }


def test_diagnostics_classify_hover_lock_and_write_artifacts(tmp_path: Path) -> None:
    """Verify moving-reference hover lock produces diagnostics artifacts and curriculum feedback."""
    records = [_record(index, current_x=0.0, reference_x=float(index) / 4.0, action=[1.0, 0.0, 0.0]) for index in range(5)]

    diagnostics = evaluation.diagnostics.summarize_policy_evaluation_trace(
        trace_records=records,
        action_space=_ActionSpace(),
        training_run_name="ppo_line_smoke",
        task_shape="line",
        total_timesteps=128,
        eval_steps=5,
        seed=7,
    )
    fields = evaluation.diagnostics.write_policy_evaluation_diagnostics(diagnostics, tmp_path / "diagnostics")

    assert diagnostics.failure_report["primary_failure_mode"] == "hover_lock"
    assert "insufficient_xy_motion" in diagnostics.failure_report["failure_modes"]
    assert "action_saturation" in diagnostics.failure_report["failure_modes"]
    assert diagnostics.curriculum_feedback["readiness_level"] == "line_not_ready"
    assert diagnostics.episode_summaries[0]["reference_xy_span_m"] == pytest.approx(1.0)
    assert diagnostics.episode_summaries[0]["actual_xy_span_m"] == pytest.approx(0.0)
    assert fields["failure_primary_mode"] == "hover_lock"
    assert Path(fields["evaluation_trace_path"]).is_file()
    assert Path(fields["episode_summaries_path"]).is_file()
    assert Path(fields["failure_report_path"]).is_file()
    assert Path(fields["curriculum_feedback_path"]).is_file()

    trace_lines = Path(fields["evaluation_trace_path"]).read_text(encoding="utf-8").splitlines()
    failure_payload = json.loads(Path(fields["failure_report_path"]).read_text(encoding="utf-8"))

    assert len(trace_lines) == TRACE_RECORD_COUNT
    assert failure_payload["training_run_name"] == "ppo_line_smoke"


def test_diagnostics_keep_normalized_and_real_action_metrics_separate() -> None:
    """Verify action diagnostics state whether PPO-facing actions are normalized."""
    records = [_record(index, current_x=0.0, reference_x=0.0, action=[1.0, 0.0, -1.0], task_shape="hover") for index in range(3)]
    for record in records:
        record["normalized_action"] = [[1.0, 0.0, -1.0]]
        record["real_action"] = [[1.0, 0.0, 0.5]]
        record["actions_normalized"] = True
        record["real_action_space_low"] = [[-0.2, -0.2, 0.5]]
        record["real_action_space_high"] = [[1.0, 0.2, 1.0]]

    diagnostics = evaluation.diagnostics.summarize_policy_evaluation_trace(
        trace_records=records,
        action_space=_ActionSpace(),
        training_run_name="ppo_hover_normalized",
        task_shape="hover",
        total_timesteps=64,
        eval_steps=3,
        seed=0,
    )

    assert diagnostics.metrics["actions_normalized"] is True
    assert diagnostics.metrics["action_mean"] == [1.0, 0.0, -1.0]
    assert diagnostics.metrics["action_saturation_fraction"] == [1.0, 0.0, 1.0]
    assert diagnostics.metrics["real_action_mean"] == [1.0, 0.0, 0.5]
    assert diagnostics.metrics["real_action_saturation_fraction"] == [1.0, 0.0, 1.0]
    assert diagnostics.episode_summaries[0]["real_action_max"] == [1.0, 0.0, 0.5]


def test_diagnostics_treat_pid_hover_z_bound_as_expected_boundary_action() -> None:
    """Verify accurate hover at the PID z upper bound is not failed only for saturation."""
    records = [_record(index, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, 1.0], task_shape="hover") for index in range(3)]
    for record in records:
        record["normalized_action"] = [[0.0, 0.0, 1.0]]
        record["real_action"] = [[0.0, 0.0, 1.0]]
        record["actions_normalized"] = True
        record["action_interface"] = "pid_position"
        record["real_action_type"] = "pid_target_position"
        record["real_action_space_low"] = [[-0.2, -0.2, 0.5]]
        record["real_action_space_high"] = [[0.2, 0.2, 1.0]]

    diagnostics = evaluation.diagnostics.summarize_policy_evaluation_trace(
        trace_records=records,
        action_space=_ActionSpace(),
        training_run_name="ppo_hover_upper_z_bound",
        task_shape="hover",
        total_timesteps=64,
        eval_steps=3,
        seed=0,
    )

    saturation_diagnostic = diagnostics.metrics["action_saturation_diagnostic"]
    expected_dimensions = diagnostics.metrics["expected_target_boundary_action_dimensions"]

    assert diagnostics.metrics["action_saturation_fraction"] == [0.0, 0.0, 1.0]
    assert diagnostics.metrics["real_action_saturation_fraction"] == [0.0, 0.0, 1.0]
    assert diagnostics.metrics["expected_target_boundary_action"] is True
    assert diagnostics.metrics["problematic_action_saturation_dimensions"] == []
    assert saturation_diagnostic["expected_target_boundary_action"] is True
    assert saturation_diagnostic["problematic_action_saturation_dimensions"] == []
    assert expected_dimensions[0]["axis"] == "z"
    assert expected_dimensions[0]["bound"] == "high"
    assert expected_dimensions[0]["bound_value"] == pytest.approx(1.0)
    assert diagnostics.failure_report["overall_status"] == "successful"
    assert diagnostics.failure_report["failure_modes"] == ["no_failure_detected"]
    assert diagnostics.failure_report["evidence"]["expected_target_boundary_action"] is True


def test_diagnostics_keep_direct_rpm_saturation_as_failure() -> None:
    """Verify direct-RPM action saturation remains a failure even when tracking is close."""
    rpm_high = 12000.0
    records = [_record(index, current_x=0.0, reference_x=0.0, action=[1.0, 1.0, 1.0, 1.0], task_shape="hover") for index in range(3)]
    for record in records:
        record["normalized_action"] = [[1.0, 1.0, 1.0, 1.0]]
        record["real_action"] = [[rpm_high, rpm_high, rpm_high, rpm_high]]
        record["actions_normalized"] = True
        record["action_interface"] = "direct_rpm"
        record["real_action_type"] = "motor_rpm"
        record["ppo_action_dim"] = 4
        record["real_action_space_low"] = [[0.0, 0.0, 0.0, 0.0]]
        record["real_action_space_high"] = [[rpm_high, rpm_high, rpm_high, rpm_high]]

    diagnostics = evaluation.diagnostics.summarize_policy_evaluation_trace(
        trace_records=records,
        action_space=_DirectRPMActionSpace(),
        training_run_name="ppo_hover_direct_rpm_saturated",
        task_shape="hover",
        total_timesteps=64,
        eval_steps=3,
        seed=0,
    )

    assert diagnostics.metrics["action_saturation_fraction"] == [1.0, 1.0, 1.0, 1.0]
    assert diagnostics.metrics["real_action_saturation_fraction"] == [1.0, 1.0, 1.0, 1.0]
    assert diagnostics.metrics["expected_target_boundary_action"] is False
    assert diagnostics.metrics["problematic_action_saturation_dimensions"] == [0, 1, 2, 3]
    assert diagnostics.failure_report["primary_failure_mode"] == "action_saturation"
    assert diagnostics.failure_report["overall_status"] == "failed"


def test_strict_limit_violation_is_reported_even_without_truncation() -> None:
    """Verify strict diagnostic thresholds flag instability when relaxed termination continues."""
    records = [_record(index, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, 0.0], task_shape="hover") for index in range(3)]
    records[1]["roll_pitch_yaw"] = [0.0, 0.5, 0.0]
    records[1]["strict_limit_violation"] = True
    records[1]["strict_limit_violations"] = ["pitch_above_limit"]
    records[1]["base_truncated"] = True
    records[1]["base_truncation_ignored"] = True
    records[1]["base_truncation_causes"] = ["pitch_above_limit"]
    for record in records:
        record["termination_limits_mode"] = "relaxed"
        record["termination_limits"] = {"mode": "relaxed", "terminate_on_base_truncation": False}
        record["diagnostic_limits"] = {"mode": "default", "max_roll_pitch_rad": 0.4}
        record["base_truncation_policy"] = "diagnose_only"
        record["terminate_on_base_truncation"] = False

    diagnostics = evaluation.diagnostics.summarize_policy_evaluation_trace(
        trace_records=records,
        action_space=_ActionSpace(),
        training_run_name="relaxed_hover",
        task_shape="hover",
        total_timesteps=64,
        eval_steps=3,
        seed=0,
    )

    assert diagnostics.metrics["strict_limit_violation_count"] == 1
    assert diagnostics.metrics["strict_limit_violation_causes"] == ["pitch_above_limit"]
    assert diagnostics.metrics["base_truncation_ignored_count"] == 1
    assert diagnostics.failure_report["primary_failure_mode"] == "attitude_instability"
    assert "safety_limit_violation" in diagnostics.failure_report["failure_modes"]
    assert diagnostics.curriculum_feedback["readiness_level"] == "unstable"


def test_diagnostics_reject_task_shape_mismatch() -> None:
    """Verify traces cannot be summarized against a different task shape."""
    records = [_record(0, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, 0.0], task_shape="line")]

    with pytest.raises(ValueError, match="task_shape mismatch"):
        evaluation.diagnostics.summarize_policy_evaluation_trace(
            trace_records=records,
            action_space=_ActionSpace(),
            training_run_name="bad_task",
            task_shape="hover",
            total_timesteps=1,
            eval_steps=1,
            seed=0,
        )


def test_diagnostics_reject_mismatched_position_lengths() -> None:
    """Verify strict trace checks catch malformed actual/reference rows."""
    record = _record(0, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, 0.0])
    record["actual_position_xyz_m"] = [0.0, 0.0]

    with pytest.raises(ValueError, match="exactly 3 values"):
        evaluation.diagnostics.summarize_policy_evaluation_trace(
            trace_records=[record],
            action_space=_ActionSpace(),
            training_run_name="bad_trace",
            task_shape="line",
            total_timesteps=1,
            eval_steps=1,
            seed=0,
        )


def test_tracking_only_metrics_exclude_start_hold_rows() -> None:
    """Verify tracking-only metrics omit configured start-hold rows."""
    records = [
        _record(0, current_x=1.0, reference_x=0.0, action=[0.0, 0.0, 0.0]),
        _record(1, current_x=1.0, reference_x=0.0, action=[0.0, 0.0, 0.0]),
        _record(2, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, 0.0]),
        _record(3, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, 0.0]),
    ]
    tracking_phase_start_step = 2
    tracking_phase_start_time_sec = 2.0
    eval_steps = 4
    total_timesteps = 64

    for index, record in enumerate(records):
        record["start_hold_enabled"] = True
        record["start_hold_sec"] = tracking_phase_start_time_sec
        record["exclude_start_hold_from_tracking_metrics"] = True
        record["tracking_phase_start_step"] = tracking_phase_start_step
        record["tracking_phase_start_time_sec"] = tracking_phase_start_time_sec
        record["is_start_hold"] = index < tracking_phase_start_step
        record["is_tracking_phase"] = index >= tracking_phase_start_step

    diagnostics = evaluation.diagnostics.summarize_policy_evaluation_trace(
        trace_records=records,
        action_space=_ActionSpace(),
        training_run_name="start_hold",
        task_shape="line",
        total_timesteps=total_timesteps,
        eval_steps=eval_steps,
        seed=0,
    )

    assert diagnostics.metrics["mean_position_error_m"] == pytest.approx(0.5)
    assert diagnostics.metrics["mean_position_error_tracking_m"] == pytest.approx(0.0)
    assert diagnostics.metrics["tracking_phase_start_step"] == tracking_phase_start_step
    assert diagnostics.metrics["tracking_phase_start_time_sec"] == pytest.approx(tracking_phase_start_time_sec)


def test_tracking_only_metrics_exclude_final_hold_rows() -> None:
    """Verify tracking-only metrics omit configured final-hold rows."""
    records = [
        _record(0, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, 0.0]),
        _record(1, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, 0.0]),
        _record(2, current_x=1.0, reference_x=0.0, action=[0.0, 0.0, 0.0]),
        _record(3, current_x=1.0, reference_x=0.0, action=[0.0, 0.0, 0.0]),
    ]
    tracking_phase_end_step = 2
    tracking_phase_end_time_sec = 1.0

    for index, record in enumerate(records):
        record["final_hold_enabled"] = True
        record["final_hold_sec"] = 1.0
        record["exclude_final_hold_from_tracking_metrics"] = True
        record["tracking_phase_end_step"] = tracking_phase_end_step
        record["tracking_phase_end_time_sec"] = tracking_phase_end_time_sec
        record["is_final_hold"] = index >= tracking_phase_end_step
        record["is_tracking_phase"] = index < tracking_phase_end_step

    diagnostics = evaluation.diagnostics.summarize_policy_evaluation_trace(
        trace_records=records,
        action_space=_ActionSpace(),
        training_run_name="final_hold",
        task_shape="line",
        total_timesteps=64,
        eval_steps=4,
        seed=0,
    )

    assert diagnostics.metrics["mean_position_error_m"] == pytest.approx(0.5)
    assert diagnostics.metrics["mean_position_error_tracking_m"] == pytest.approx(0.0)
    assert diagnostics.metrics["final_hold_enabled"] is True
    assert diagnostics.metrics["tracking_phase_end_step"] == tracking_phase_end_step
    assert diagnostics.metrics["tracking_phase_end_time_sec"] == pytest.approx(tracking_phase_end_time_sec)


def test_diagnostics_report_no_failure_for_accurate_hover() -> None:
    """Verify accurate hover evaluations produce successful curriculum feedback."""
    records = [_record(index, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, 0.0], task_shape="hover") for index in range(3)]

    diagnostics = evaluation.diagnostics.summarize_policy_evaluation_trace(
        trace_records=records,
        action_space=_ActionSpace(),
        training_run_name="ppo_hover_smoke",
        task_shape="hover",
        total_timesteps=64,
        eval_steps=3,
        seed=0,
    )

    assert diagnostics.failure_report["overall_status"] == "successful"
    assert diagnostics.failure_report["failure_modes"] == ["no_failure_detected"]
    assert diagnostics.curriculum_feedback["readiness_level"] == "near_target_ready"
