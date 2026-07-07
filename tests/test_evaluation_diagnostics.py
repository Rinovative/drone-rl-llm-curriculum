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


def _record(step_index: int, current_x: float, reference_x: float, action: list[float]) -> dict[str, object]:
    """Return one complete policy evaluation trace record."""
    current_position = [current_x, 0.0, 1.0]
    reference_position = [reference_x, 0.0, 1.0]
    error_x = current_x - reference_x
    return {
        "step_index": step_index,
        "episode_index": 0,
        "episode_step_index": step_index,
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
    records = [_record(index, current_x=0.0, reference_x=0.0, action=[1.0, 0.0, -1.0]) for index in range(3)]
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


def test_diagnostics_report_no_failure_for_accurate_hover() -> None:
    """Verify accurate hover evaluations produce successful curriculum feedback."""
    records = [_record(index, current_x=0.0, reference_x=0.0, action=[0.0, 0.0, 0.0]) for index in range(3)]

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
