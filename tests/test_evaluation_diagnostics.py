"""Tests for structured trained-policy evaluation diagnostics."""

# ruff: noqa: S101

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src import evaluation

TRACE_RECORD_COUNT = 5
EXPECTED_CURRICULUM_FEEDBACK_VERSION = 2
PHASE_START_STEP = 2
PHASE_END_STEP = 5


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


def _set_trace_xyz(record: dict[str, object], actual: list[float], reference: list[float]) -> None:
    """Update XYZ trace fields while keeping consistency checks satisfied."""
    actual_array = np.asarray(actual, dtype=float)
    reference_array = np.asarray(reference, dtype=float)
    error = actual_array - reference_array
    record["current_position"] = actual_array.tolist()
    record["reference_position"] = reference_array.tolist()
    record["actual_position_xyz_m"] = actual_array.tolist()
    record["reference_position_xyz_m"] = reference_array.tolist()
    record["axis_error_xyz"] = error.tolist()
    record["error_xyz_m"] = error.tolist()
    record["position_error_m"] = float(np.linalg.norm(error))


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
    assert diagnostics.curriculum_feedback["readiness_level"] == "partially_ready"
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


def _curriculum_feedback(
    task_shape: str,
    failure_modes: list[str],
    metrics: dict[str, object] | None = None,
) -> dict[str, object]:
    """Build curriculum feedback from compact test diagnostics."""
    payload = {
        "mean_position_error_m": 0.2,
        "mean_position_error_tracking_m": 0.2,
        "final_position_error_m": 0.2,
        "actual_z_span_m": 0.0,
        "mean_abs_z_error": 0.0,
        "reference_xy_span_m": 0.0,
        "actual_xy_span_m": 0.0,
        "xy_tracking_ratio": None,
        "action_saturation_fraction": [0.0, 0.0, 0.0],
        "eval_terminated_count": 0,
        "eval_truncated_count": 0,
        "strict_limit_violation_count": 0,
    }
    payload.update(metrics or {})
    return evaluation.diagnostics.build_curriculum_feedback(
        current_task_shape=task_shape,
        failure_report={"failure_modes": failure_modes},
        metrics=payload,
    )


def test_curriculum_feedback_schema_contains_structured_constructive_guidance() -> None:
    """Verify curriculum feedback includes the new structured schema and legacy fields."""
    feedback = _curriculum_feedback(
        "line",
        ["hover_lock", "insufficient_xy_motion"],
        {"mean_position_error_tracking_m": 0.42, "reference_xy_span_m": 1.0, "actual_xy_span_m": 0.0, "xy_tracking_ratio": 0.0},
    )

    assert feedback["feedback_version"] == EXPECTED_CURRICULUM_FEEDBACK_VERSION
    assert feedback["readiness_level"] == "partially_ready"
    assert feedback["current_task_family"] == "line"
    assert feedback["performance_summary"]["own_task_status"] == "stable_with_skill_gaps"
    assert "xy_tracking" in feedback["diagnosis"]["primary_skill_gaps"]
    assert feedback["recommended_next_tasks"]
    assert feedback["recommended_next_task_families"]
    assert all(
        "reason" in item and "targeted_skill" in item and "difficulty_hint" in item and "priority" in item
        for item in feedback["recommended_next_task_families"]
    )
    assert all("reason" in item and "condition_to_reintroduce" in item for item in feedback["avoid_next_task_families"])
    assert feedback["constraints_for_next_curriculum"]
    assert "llm_instruction_summary" in feedback


def test_curriculum_feedback_z_instability_recommends_controlled_altitude_tasks() -> None:
    """Verify z instability alone becomes altitude-control practice instead of blanket avoidance."""
    feedback = _curriculum_feedback(
        "line",
        ["z_instability"],
        {
            "mean_position_error_tracking_m": 0.31,
            "final_position_error_m": 0.28,
            "mean_abs_z_error": 0.45,
            "actual_z_span_m": 0.8,
            "position_error_trend": "improving",
        },
    )
    recommended = {item["task_family"] for item in feedback["recommended_next_task_families"]}
    avoided = {item["task_family"] for item in feedback["avoid_next_task_families"]}

    assert feedback["readiness_level"] == "improving"
    assert feedback["curriculum_strategy"]["progression_type"] == "targeted_skill_training"
    assert "altitude_control" in feedback["diagnosis"]["primary_skill_gaps"]
    assert {"takeoff_stabilization", "hover_stabilization", "multi_height_polyline"}.issubset(recommended)
    assert "takeoff_stabilization" not in avoided
    assert "multi_height_polyline" not in avoided


def test_curriculum_feedback_action_saturation_alone_is_not_unstable() -> None:
    """Verify action saturation alone remains diagnostic when tracking is stable."""
    feedback = _curriculum_feedback(
        "line",
        ["action_saturation"],
        {
            "mean_position_error_tracking_m": 0.12,
            "final_position_error_m": 0.1,
            "action_saturation_fraction": [1.0, 0.0, 0.0],
        },
    )
    action_mode = next(item for item in feedback["diagnosis"]["interpreted_failure_modes"] if item["name"] == "action_saturation")

    assert feedback["readiness_level"] == "partially_ready"
    assert action_mode["is_policy_instability"] is False
    assert feedback["curriculum_strategy"]["should_recover"] is False


def test_curriculum_feedback_reference_too_hard_recommends_easier_same_family() -> None:
    """Verify hard references produce easier same-family guidance instead of broad family avoidance."""
    feedback = _curriculum_feedback(
        "polyline",
        ["reference_too_fast_or_too_hard"],
        {"mean_position_error_tracking_m": 0.55, "final_position_error_m": 0.5, "reference_xy_span_m": 1.2, "actual_xy_span_m": 0.5},
    )
    polyline_recommendation = next(item for item in feedback["recommended_next_task_families"] if item["task_family"] == "polyline")

    assert polyline_recommendation["difficulty_hint"] == "easier_same_family"
    assert "slower" in polyline_recommendation["reason"]
    assert "speed_control" in feedback["diagnosis"]["primary_skill_gaps"]
    assert feedback["diagnosis"]["interpreted_failure_modes"][0]["is_task_difficulty"] is True


def test_curriculum_feedback_hard_scenario_failure_does_not_dominate_own_task_readiness() -> None:
    """Verify hard scenario failures remain stress-test context for own-task feedback."""
    feedback = _curriculum_feedback(
        "line",
        ["no_failure_detected"],
        {"mean_position_error_tracking_m": 0.08, "final_position_error_m": 0.08, "scenario_status": "hard_scenario_failed"},
    )

    assert feedback["readiness_level"] == "ready"
    assert feedback["performance_summary"]["own_task_status"] == "own_task_ready"
    assert feedback["performance_summary"]["scenario_status"] == "hard_scenario_failed"


def test_curriculum_feedback_turn_and_curvature_gaps_are_constructive() -> None:
    """Verify turn and curvature weaknesses recommend targeted families."""
    turn_feedback = _curriculum_feedback(
        "polyline",
        ["reference_too_fast_or_too_hard"],
        {"mean_position_error_tracking_m": 0.5, "final_position_error_m": 0.45, "reference_xy_span_m": 1.0, "actual_xy_span_m": 0.6},
    )
    curve_feedback = _curriculum_feedback(
        "figure_eight",
        ["reference_too_fast_or_too_hard"],
        {"mean_position_error_tracking_m": 0.5, "final_position_error_m": 0.45, "reference_xy_span_m": 1.0, "actual_xy_span_m": 0.6},
    )

    assert {"l_shape", "polyline"}.issubset({item["task_family"] for item in turn_feedback["recommended_next_task_families"]})
    assert "turn_following" in turn_feedback["diagnosis"]["primary_skill_gaps"]
    assert {"ellipse", "circle"}.issubset({item["task_family"] for item in curve_feedback["recommended_next_task_families"]})
    assert "figure_eight" in {item["task_family"] for item in curve_feedback["avoid_next_task_families"]}


def test_curriculum_feedback_severe_attitude_divergence_still_recovers() -> None:
    """Verify true control instability still produces recovery recommendations."""
    feedback = _curriculum_feedback(
        "circle",
        ["attitude_instability", "safety_limit_violation"],
        {"mean_position_error_tracking_m": 0.6, "final_position_error_m": 0.7, "max_abs_roll_pitch_rad": 0.6, "strict_limit_violation_count": 1},
    )

    assert feedback["readiness_level"] == "unstable"
    assert feedback["curriculum_strategy"]["progression_type"] == "recover"
    assert feedback["curriculum_strategy"]["should_recover"] is True
    assert feedback["recommended_next_task_families"][0]["task_family"] == "hover_stabilization"


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

    assert diagnostics.metrics["action_upper_saturation_fraction_by_dim"] == [1.0, 0.0, 0.0]
    assert diagnostics.metrics["action_lower_saturation_fraction_by_dim"] == [0.0, 0.0, 1.0]
    assert diagnostics.metrics["normalized_action_p95_by_dim"] == [1.0, 0.0, -1.0]
    assert diagnostics.metrics["real_action_p95_by_dim"] == [1.0, 0.0, 0.5]


def test_diagnostics_report_phase_saturation_and_pid_z_target_metrics() -> None:
    """Verify PID z saturation diagnostics are split by rollout phase."""
    actions = [1.0, 1.0, 1.0, 0.0, -1.0, 1.0]
    actual_z = [1.0, 1.0, 0.9, 1.1, 1.2, 1.0]
    real_z = [1.2, 1.2, 1.2, 1.0, 0.8, 1.2]
    velocity_z = [0.0, 0.0, 0.2, -0.1, 0.0, 0.0]
    records = [_record(index, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, action], task_shape="line") for index, action in enumerate(actions)]
    for index, record in enumerate(records):
        _set_trace_xyz(record, actual=[0.0, 0.0, actual_z[index]], reference=[0.0, 0.0, 1.0])
        record["normalized_action"] = [[0.0, 0.0, actions[index]]]
        record["real_action"] = [[0.0, 0.0, real_z[index]]]
        record["actions_normalized"] = True
        record["action_interface"] = "pid_position"
        record["real_action_type"] = "pid_target_position"
        record["real_action_space_low"] = [[-0.2, -0.2, 0.8]]
        record["real_action_space_high"] = [[0.2, 0.2, 1.2]]
        record["velocity"] = [0.0, 0.0, velocity_z[index]]
        record["start_hold_enabled"] = True
        record["final_hold_enabled"] = True
        record["tracking_phase_start_step"] = PHASE_START_STEP
        record["tracking_phase_end_step"] = PHASE_END_STEP
        record["is_start_hold"] = index < PHASE_START_STEP
        record["is_final_hold"] = index == PHASE_END_STEP
        record["is_tracking_phase"] = PHASE_START_STEP <= index < PHASE_END_STEP

    diagnostics = evaluation.diagnostics.summarize_policy_evaluation_trace(
        trace_records=records,
        action_space=_ActionSpace(),
        training_run_name="ppo_line_pid_z_audit",
        task_shape="line",
        total_timesteps=64,
        eval_steps=len(records),
        seed=0,
    )

    assert diagnostics.metrics["action_saturation_fraction_by_dim"] == [0.0, 0.0, pytest.approx(5.0 / 6.0)]
    assert diagnostics.metrics["action_upper_saturation_fraction_by_dim"] == [0.0, 0.0, pytest.approx(4.0 / 6.0)]
    assert diagnostics.metrics["action_lower_saturation_fraction_by_dim"] == [0.0, 0.0, pytest.approx(1.0 / 6.0)]
    assert diagnostics.metrics["action_saturation_fraction_start_hold_by_dim"] == [0.0, 0.0, 1.0]
    assert diagnostics.metrics["action_saturation_fraction_tracking_by_dim"] == [0.0, 0.0, pytest.approx(2.0 / 3.0)]
    assert diagnostics.metrics["action_saturation_fraction_final_hold_by_dim"] == [0.0, 0.0, 1.0]
    assert diagnostics.metrics["z_action_upper_saturation_fraction_tracking"] == pytest.approx(1.0 / 3.0)
    assert diagnostics.metrics["z_target_minus_reference_mean"] == pytest.approx(0.0)
    assert diagnostics.metrics["z_target_minus_reference_p95"] == pytest.approx(0.18)
    assert diagnostics.metrics["z_error_mean_tracking"] == pytest.approx((0.1 + 0.1 + 0.2) / 3.0)
    assert diagnostics.metrics["z_error_p95_tracking"] == pytest.approx(0.19)
    assert diagnostics.metrics["z_overshoot_fraction_tracking"] == pytest.approx(2.0 / 3.0)
    assert diagnostics.metrics["vertical_velocity_mean_tracking"] == pytest.approx(1.0 / 30.0)
    assert diagnostics.metrics["vertical_velocity_p95_abs_tracking"] == pytest.approx(0.19)


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
    assert diagnostics.metrics["rpm_saturation_fraction_by_motor"] == [1.0, 1.0, 1.0, 1.0]
    assert diagnostics.metrics["rpm_upper_saturation_fraction_by_motor"] == [1.0, 1.0, 1.0, 1.0]
    assert diagnostics.metrics["rpm_lower_saturation_fraction_by_motor"] == [0.0, 0.0, 0.0, 0.0]
    assert diagnostics.metrics["rpm_clipped_fraction"] == 0.0
    assert "z_target_minus_reference_mean" not in diagnostics.metrics
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
    assert diagnostics.curriculum_feedback["readiness_level"] == "ready"
