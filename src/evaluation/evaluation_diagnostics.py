"""
===============================================================================
evaluation_diagnostics.py
===============================================================================
Build structured trained-policy evaluation diagnostics for PPO tracking runs.

Responsibilities:
  - Collect deterministic evaluation step traces from trained policies
  - Summarize per-episode tracking, action, and termination behavior
  - Classify common tracking failure modes for curriculum feedback
  - Write compact diagnostic artifacts under a run-scoped diagnostics directory

Design principles:
  - Keep detailed step data in one JSONL trace artifact
  - Store summaries and curriculum feedback separately from large traces
  - Use named thresholds so failure heuristics are easy to audit

Boundaries:
  - PPO training and model persistence belong in experiments modules
  - Reward logic, action semantics, and simulator physics must not be changed here
===============================================================================

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src import evaluation, utils

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

EVALUATION_TRACE_FILENAME = "evaluation_trace.jsonl"
EPISODE_SUMMARIES_FILENAME = "episode_summaries.json"
FAILURE_REPORT_FILENAME = "failure_report.json"
CURRICULUM_FEEDBACK_FILENAME = "curriculum_feedback.json"

POSITION_DIMENSIONS = 3
ROLL_PITCH_DIMENSIONS = 2
Z_AXIS_INDEX = 2
ARRAY_BOUNDS_MAX_NDIM = 2
ACTION_SATURATION_TOLERANCE = 1.0e-6
TARGET_BOUNDARY_ACTION_TOLERANCE_M = 1.0e-6
XY_TRACKING_RATIO_MIN_REFERENCE_SPAN_M = 1.0e-9
REFERENCE_XY_SPAN_MOVING_TASK_MIN_M = 0.2
HOVER_LOCK_ACTUAL_XY_SPAN_MAX_M = 0.05
INSUFFICIENT_XY_MOTION_RATIO = 0.25
ACTION_SATURATION_FRACTION_MIN = 0.5
OVERSHOOT_XY_SPAN_RATIO = 1.5
HIGH_MEAN_POSITION_ERROR_M = 0.35
ACCEPTABLE_MEAN_POSITION_ERROR_M = 0.25
ACCEPTABLE_FINAL_POSITION_ERROR_M = 0.35
Z_INSTABILITY_SPAN_M = 0.75
Z_INSTABILITY_MEAN_ABS_ERROR_M = 0.4
ATTITUDE_INSTABILITY_MAX_ABS_ROLL_PITCH_RAD = 0.35
STRICT_LIMIT_VIOLATION_COUNT_MIN = 1
CURRICULUM_FEEDBACK_VERSION = 2
READINESS_BLOCKED = "blocked"
READINESS_UNSTABLE = "unstable"
READINESS_IMPROVING = "improving"
READINESS_PARTIALLY_READY = "partially_ready"
READINESS_READY = "ready"
READINESS_STRONG = "strong"
TREND_IMPROVING = "improving"
TREND_WORSENING = "worsening"
TREND_FLAT = "flat"
TREND_UNKNOWN = "unknown"

FAILURE_HOVER_LOCK = "hover_lock"
FAILURE_INSUFFICIENT_XY_MOTION = "insufficient_xy_motion"
FAILURE_ACTION_SATURATION = "action_saturation"
FAILURE_OVERSHOOT = "overshoot"
FAILURE_Z_INSTABILITY = "z_instability"
FAILURE_ATTITUDE_INSTABILITY = "attitude_instability"
FAILURE_EARLY_TERMINATION = "early_termination"
FAILURE_REPEATED_TRUNCATION = "repeated_truncation"
FAILURE_SAFETY_LIMIT_VIOLATION = "safety_limit_violation"
FAILURE_REFERENCE_TOO_HARD = "reference_too_fast_or_too_hard"
FAILURE_NONE = "no_failure_detected"
DIAGNOSTIC_EXPECTED_TARGET_BOUNDARY_ACTION = "expected_target_boundary_action"
ACTION_AXIS_NAMES = ("x", "y", "z")
PID_Z_REACHABILITY_KEYS = (
    "pid_target_z_min_m",
    "pid_target_z_max_m",
    "base_pid_z_target_low",
    "base_pid_z_target_high",
    "real_pid_z_target_low",
    "real_pid_z_target_high",
    "reference_z_min",
    "reference_z_max",
    "reference_z_reachable_by_pid_position",
    "z_reference_above_pid_high_margin",
    "z_reference_below_pid_low_margin",
    "pid_z_action_space_expanded",
    "pid_z_action_space_changed",
    "real_pid_z_target_for_normalized_action2_minus1",
    "real_pid_z_target_for_normalized_action2_zero",
    "real_pid_z_target_for_normalized_action2_plus1",
    "normalized_action2_real_z_targets",
)
INITIAL_STATE_KEYS = (
    "initial_state_mode",
    "initial_state",
    "initial_xyz",
    "requested_initial_xyz",
    "actual_initial_xyz",
    "initial_xyz_source",
    "initial_xyz_offset",
    "initial_reference_xyz",
    "initial_xyz_matches_reference_start",
    "initial_position_error_m",
    "initial_z_error_m",
    "initial_z_error_signed_m",
    "spawned_at_reference_start",
)
SKILL_ALTITUDE_CONTROL = "altitude_control"
SKILL_XY_TRACKING = "xy_tracking"
SKILL_SPEED_CONTROL = "speed_control"
SKILL_TURN_FOLLOWING = "turn_following"
SKILL_CURVATURE_FOLLOWING = "curvature_following"
SKILL_MULTI_SEGMENT_TRACKING = "multi_segment_tracking"
SKILL_STABILITY_RECOVERY = "stability_recovery"

FAMILY_HOVER = "hover_stabilization"
FAMILY_TAKEOFF = "takeoff_stabilization"
FAMILY_LINE = "line"
FAMILY_START_HOLD_LINE = "start_hold_then_line"
FAMILY_POLYLINE = "polyline"
FAMILY_L_SHAPE = "l_shape"
FAMILY_ZIGZAG = "zigzag"
FAMILY_MULTI_HEIGHT_POLYLINE = "multi_height_polyline"
FAMILY_CIRCLE = "circle"
FAMILY_ELLIPSE = "ellipse"
FAMILY_FIGURE_EIGHT = "figure_eight"


@dataclass(frozen=True)
class PolicyEvaluationDiagnostics:
    """
    Structured diagnostic payloads derived from a deterministic policy evaluation.

    Parameters
    ----------
    metrics
        Compact JSON-serializable summary metrics suitable for metrics JSON and W&B.
    trace_records
        Per-step diagnostic records. These are the source of truth for detailed rollout data.
    episode_summaries
        Per-episode summaries derived from ``trace_records``.
    failure_report
        Automatic failure-mode classification with evidence.
    curriculum_feedback
        Compact feedback payload for later manual or LLM-guided curriculum design.

    """

    metrics: dict[str, Any]
    trace_records: list[dict[str, Any]]
    episode_summaries: list[dict[str, Any]]
    failure_report: dict[str, Any]
    curriculum_feedback: dict[str, Any]


def collect_policy_evaluation_diagnostics(
    model: Any,
    tracking_env: Any,
    eval_steps: int,
    seed: int,
    training_run_name: str,
    task_shape: str,
    total_timesteps: int,
) -> PolicyEvaluationDiagnostics:
    """
    Run deterministic policy evaluation and return structured diagnostics.

    Parameters
    ----------
    model
        Trained policy object exposing ``predict(observation, deterministic=True)``.
    tracking_env
        Gymnasium-compatible trajectory tracking environment.
    eval_steps
        Number of deterministic evaluation steps to execute.
    seed
        Base reset seed.
    training_run_name
        Run identifier used in failure reports.
    task_shape
        Current training task shape.
    total_timesteps
        PPO training timestep budget.

    Returns
    -------
    PolicyEvaluationDiagnostics
        Metrics, trace rows, episode summaries, failure report, and curriculum feedback.

    """
    observation, _ = tracking_env.reset(seed=seed)
    trace_records: list[dict[str, Any]] = []
    episode_index = 0
    episode_step_index = 0
    reset_count = 0
    for step_index in range(eval_steps):
        action, _ = model.predict(observation, deterministic=True)
        observation, reward, terminated, truncated, info = tracking_env.step(action)
        trace_records.append(
            _make_trace_record(
                step_index=step_index,
                episode_index=episode_index,
                episode_step_index=episode_step_index,
                action=action,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
        )
        if terminated or truncated:
            reset_count += 1
            episode_index += 1
            episode_step_index = 0
            observation, _ = tracking_env.reset(seed=seed + reset_count)
        else:
            episode_step_index += 1

    return summarize_policy_evaluation_trace(
        trace_records=trace_records,
        action_space=tracking_env.action_space,
        training_run_name=training_run_name,
        task_shape=task_shape,
        total_timesteps=total_timesteps,
        eval_steps=eval_steps,
        seed=seed,
    )


def summarize_policy_evaluation_trace(
    trace_records: Sequence[Mapping[str, Any]],
    action_space: Any,
    training_run_name: str,
    task_shape: str,
    total_timesteps: int,
    eval_steps: int,
    seed: int,
) -> PolicyEvaluationDiagnostics:
    """
    Build diagnostics from an existing policy evaluation trace.

    Parameters
    ----------
    trace_records
        Per-step records containing positions, rewards, actions, and terminal flags.
    action_space
        Action space used to compute saturation fractions.
    training_run_name
        Run identifier used in reports.
    task_shape
        Current task shape.
    total_timesteps
        PPO training timestep budget.
    eval_steps
        Requested evaluation length.
    seed
        Evaluation reset seed.

    Returns
    -------
    PolicyEvaluationDiagnostics
        Metrics, normalized trace records, summaries, failure report, and curriculum feedback.

    Raises
    ------
    ValueError
        If the trace is empty.

    """
    records = [_json_ready(dict(record)) for record in trace_records]
    if not records:
        message = "trace_records must contain at least one step"
        raise ValueError(message)
    _validate_trace_consistency(records)
    _validate_trace_task_shape(records=records, task_shape=task_shape)
    episode_summaries = _episode_summaries(records, action_space)
    metrics = _overall_metrics(records, episode_summaries, action_space)
    metrics.update(
        {
            "actual_eval_steps": len(records),
            "eval_steps": eval_steps,
            "eval_resets": int(sum(1 for record in records if bool(record.get("terminated")) or bool(record.get("truncated")))),
            "eval_terminated_count": int(sum(1 for record in records if bool(record.get("terminated")))),
            "eval_truncated_count": int(sum(1 for record in records if bool(record.get("truncated")))),
        }
    )
    failure_report = build_failure_report(
        metrics=metrics,
        episode_summaries=episode_summaries,
        training_run_name=training_run_name,
        task_shape=task_shape,
        total_timesteps=total_timesteps,
        seed=seed,
        eval_steps=eval_steps,
    )
    curriculum_feedback = build_curriculum_feedback(current_task_shape=task_shape, failure_report=failure_report, metrics=metrics)
    return PolicyEvaluationDiagnostics(
        metrics=metrics,
        trace_records=records,
        episode_summaries=episode_summaries,
        failure_report=failure_report,
        curriculum_feedback=curriculum_feedback,
    )


def write_policy_evaluation_diagnostics(
    diagnostics: PolicyEvaluationDiagnostics,
    diagnostics_dir: str | Path,
) -> dict[str, Any]:
    """
    Write evaluation diagnostics artifacts and return metrics/manifest fields.

    Parameters
    ----------
    diagnostics
        Structured diagnostics created from deterministic policy evaluation.
    diagnostics_dir
        Output directory for trace, episode summaries, failure report, and feedback.

    Returns
    -------
    dict[str, Any]
        Summary fields and artifact paths for metrics JSON and manifests.

    """
    resolved_dir = Path(diagnostics_dir)
    resolved_dir.mkdir(parents=True, exist_ok=True)
    trace_path = resolved_dir / EVALUATION_TRACE_FILENAME
    episode_summaries_path = resolved_dir / EPISODE_SUMMARIES_FILENAME
    failure_report_path = resolved_dir / FAILURE_REPORT_FILENAME
    curriculum_feedback_path = resolved_dir / CURRICULUM_FEEDBACK_FILENAME

    evaluation.rollout.write_policy_rollout_trace(diagnostics.trace_records, trace_path)
    _write_json(episode_summaries_path, diagnostics.episode_summaries)
    _write_json(failure_report_path, diagnostics.failure_report)
    _write_json(curriculum_feedback_path, diagnostics.curriculum_feedback)

    failure_modes = list(diagnostics.failure_report["failure_modes"])
    feedback = diagnostics.curriculum_feedback
    diagnosis = feedback.get("diagnosis", {}) if isinstance(feedback.get("diagnosis"), dict) else {}
    return {
        "diagnostics_dir": str(resolved_dir),
        "evaluation_trace_path": str(trace_path),
        "episode_summaries_path": str(episode_summaries_path),
        "failure_report_path": str(failure_report_path),
        "curriculum_feedback_path": str(curriculum_feedback_path),
        "failure_primary_mode": diagnostics.failure_report["primary_failure_mode"],
        "failure_modes": failure_modes,
        "failure_overall_status": diagnostics.failure_report["overall_status"],
        "curriculum_feedback_version": feedback.get("feedback_version"),
        "curriculum_feedback_summary": feedback.get("llm_instruction_summary"),
        "curriculum_current_task_family": feedback.get("current_task_family"),
        "curriculum_current_difficulty_level": feedback.get("current_difficulty_level"),
        "curriculum_primary_skill_gaps": list(diagnosis.get("primary_skill_gaps", [])),
        "curriculum_diagnostic_signals": dict(diagnosis.get("diagnostic_signals", {})),
        "curriculum_strategy": dict(feedback.get("curriculum_strategy", {})),
        "curriculum_recommended_next_task_families": list(feedback.get("recommended_next_task_families", [])),
        "curriculum_avoid_next_task_families": list(feedback.get("avoid_next_task_families", [])),
        "curriculum_constraints_for_next": list(feedback.get("constraints_for_next_curriculum", [])),
        "curriculum_readiness_level": feedback["readiness_level"],
        "curriculum_recommended_next_tasks": list(feedback["recommended_next_tasks"]),
        "curriculum_avoid_next_tasks": list(feedback["avoid_next_tasks"]),
    }


def build_failure_report(
    metrics: Mapping[str, Any],
    episode_summaries: Sequence[Mapping[str, Any]],
    training_run_name: str,
    task_shape: str,
    total_timesteps: int,
    seed: int,
    eval_steps: int,
) -> dict[str, Any]:
    """
    Classify evaluation failures using named tracking and stability thresholds.

    Parameters
    ----------
    metrics
        Overall deterministic evaluation metrics.
    episode_summaries
        Per-episode summaries used as supporting evidence.
    training_run_name
        Run identifier.
    task_shape
        Current task shape.
    total_timesteps
        PPO training timestep budget.
    seed
        Evaluation seed.
    eval_steps
        Requested evaluation length.

    Returns
    -------
    dict[str, Any]
        JSON-serializable failure report.

    """
    failure_modes: list[str] = []
    reference_xy_span_m = _float(metrics.get("reference_xy_span_m"))
    actual_xy_span_m = _float(metrics.get("actual_xy_span_m"))
    mean_position_error_m = _float(metrics.get("mean_position_error_m"))
    action_saturation_fraction = _float_list(metrics.get("action_saturation_fraction"))
    max_action_saturation = max(action_saturation_fraction, default=0.0)
    moving_reference = reference_xy_span_m > REFERENCE_XY_SPAN_MOVING_TASK_MIN_M
    mean_position_error_m = _float(metrics.get("mean_position_error_tracking_m", mean_position_error_m))
    high_error = mean_position_error_m >= HIGH_MEAN_POSITION_ERROR_M
    tracking_acceptable = _tracking_acceptable(metrics)

    if moving_reference and actual_xy_span_m < HOVER_LOCK_ACTUAL_XY_SPAN_MAX_M:
        failure_modes.append(FAILURE_HOVER_LOCK)
    if moving_reference and actual_xy_span_m < INSUFFICIENT_XY_MOTION_RATIO * reference_xy_span_m:
        failure_modes.append(FAILURE_INSUFFICIENT_XY_MOTION)
    if max_action_saturation >= ACTION_SATURATION_FRACTION_MIN and _action_saturation_is_failure(metrics, tracking_acceptable):
        failure_modes.append(FAILURE_ACTION_SATURATION)
    if moving_reference and actual_xy_span_m > OVERSHOOT_XY_SPAN_RATIO * reference_xy_span_m and high_error:
        failure_modes.append(FAILURE_OVERSHOOT)
    if _float(metrics.get("actual_z_span_m")) >= Z_INSTABILITY_SPAN_M or _float(metrics.get("mean_abs_z_error")) >= Z_INSTABILITY_MEAN_ABS_ERROR_M:
        failure_modes.append(FAILURE_Z_INSTABILITY)
    if _max_episode_value(episode_summaries, "max_abs_roll_pitch_rad") >= ATTITUDE_INSTABILITY_MAX_ABS_ROLL_PITCH_RAD:
        failure_modes.append(FAILURE_ATTITUDE_INSTABILITY)
    if _int(metrics.get("eval_terminated_count")) > 0 and _int(metrics.get("actual_eval_steps")) < eval_steps:
        failure_modes.append(FAILURE_EARLY_TERMINATION)
    if _int(metrics.get("eval_truncated_count")) > 0:
        failure_modes.append(FAILURE_REPEATED_TRUNCATION)
    if _int(metrics.get("strict_limit_violation_count")) >= STRICT_LIMIT_VIOLATION_COUNT_MIN:
        failure_modes.append(FAILURE_SAFETY_LIMIT_VIOLATION)
    if high_error and (
        FAILURE_ACTION_SATURATION in failure_modes
        or FAILURE_Z_INSTABILITY in failure_modes
        or FAILURE_ATTITUDE_INSTABILITY in failure_modes
        or FAILURE_REPEATED_TRUNCATION in failure_modes
        or FAILURE_SAFETY_LIMIT_VIOLATION in failure_modes
    ):
        failure_modes.append(FAILURE_REFERENCE_TOO_HARD)

    failure_modes = _dedupe(failure_modes)
    if not failure_modes and tracking_acceptable:
        failure_modes = [FAILURE_NONE]
    primary_failure_mode = failure_modes[0] if failure_modes else FAILURE_NONE
    overall_status = _overall_status(failure_modes, tracking_acceptable)
    evidence = _failure_evidence(metrics, episode_summaries)
    return {
        "training_run_name": training_run_name,
        "task_shape": task_shape,
        "total_timesteps": int(total_timesteps),
        "seed": int(seed),
        "eval_steps": int(eval_steps),
        "overall_status": overall_status,
        "primary_failure_mode": primary_failure_mode,
        "failure_modes": failure_modes,
        "human_readable_summary": _failure_summary(primary_failure_mode, overall_status, evidence),
        "evidence": evidence,
    }


def build_curriculum_feedback(
    current_task_shape: str,
    failure_report: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Build compact curriculum guidance from classified evaluation diagnostics.

    Parameters
    ----------
    current_task_shape
        Current evaluated task shape.
    failure_report
        Failure report produced by ``build_failure_report``.
    metrics
        Overall deterministic evaluation metrics.

    Returns
    -------
    dict[str, Any]
        JSON-serializable curriculum feedback payload.

    """
    failure_modes = {str(mode) for mode in failure_report.get("failure_modes", []) if str(mode)}
    if not failure_modes:
        failure_modes = {FAILURE_NONE} if _tracking_acceptable(metrics) else set()
    current_task_family = _task_family_from_shape(current_task_shape)
    trend_status = _trend_status(metrics)
    difficulty_level = _task_difficulty_level(failure_modes, metrics)
    diagnostic_signals = _diagnostic_signal_counts(failure_modes, metrics)
    primary_skill_gaps = _primary_skill_gaps(
        current_task_family=current_task_family,
        failure_modes=failure_modes,
        metrics=metrics,
        diagnostic_signals=diagnostic_signals,
    )
    policy_instability = _has_policy_instability(failure_modes=failure_modes, metrics=metrics, trend_status=trend_status)
    interpreted_failure_modes = _interpreted_failure_modes(
        current_task_family=current_task_family,
        failure_modes=failure_modes,
        metrics=metrics,
        policy_instability=policy_instability,
        trend_status=trend_status,
    )
    readiness_level = _readiness_level(
        failure_modes=failure_modes,
        metrics=metrics,
        policy_instability=policy_instability,
        trend_status=trend_status,
        primary_skill_gaps=primary_skill_gaps,
    )
    recommended_next_task_families = _recommended_next_task_families(
        current_task_family=current_task_family,
        failure_modes=failure_modes,
        metrics=metrics,
        primary_skill_gaps=primary_skill_gaps,
        policy_instability=policy_instability,
    )
    avoid_next_task_families = _avoid_next_task_families(
        current_task_family=current_task_family,
        failure_modes=failure_modes,
        primary_skill_gaps=primary_skill_gaps,
        policy_instability=policy_instability,
        trend_status=trend_status,
        metrics=metrics,
    )
    constraints = _constraints_for_next_curriculum(
        failure_modes=failure_modes,
        primary_skill_gaps=primary_skill_gaps,
        policy_instability=policy_instability,
        metrics=metrics,
    )
    strategy = _curriculum_strategy(
        readiness_level=readiness_level,
        current_task_family=current_task_family,
        failure_modes=failure_modes,
        primary_skill_gaps=primary_skill_gaps,
        policy_instability=policy_instability,
        trend_status=trend_status,
        difficulty_level=difficulty_level,
    )
    instruction_summary = _llm_instruction_summary(
        readiness_level=readiness_level,
        current_task_family=current_task_family,
        primary_skill_gaps=primary_skill_gaps,
        recommended_next_task_families=recommended_next_task_families,
        strategy=strategy,
    )
    recommendation_tasks = [str(item["task_family"]) for item in recommended_next_task_families]
    avoid_tasks = [str(item["task_family"]) for item in avoid_next_task_families]
    constraint_texts = [str(item["constraint"]) for item in constraints]

    return {
        "feedback_version": CURRICULUM_FEEDBACK_VERSION,
        "stage_index": _optional_int(metrics.get("stage_index", metrics.get("curriculum_stage_index"))),
        "current_task_family": current_task_family,
        "current_task_shape": current_task_shape,
        "current_task_source": metrics.get("task_source", "unknown"),
        "current_difficulty_level": difficulty_level,
        "stage_budget_profile": metrics.get("stage_budget_profile", metrics.get("selected_stage_budget_profile")),
        "performance_summary": {
            "own_task_status": _own_task_status(readiness_level),
            "generalization_status": str(metrics.get("generalization_status", "unknown")),
            "scenario_status": str(metrics.get("scenario_status", "not_evaluated_stress_test")),
            "trend_status": trend_status,
            "mean_position_error_tracking_m": _float(metrics.get("mean_position_error_tracking_m", metrics.get("mean_position_error_m"))),
            "reward_trend": str(metrics.get("reward_trend", TREND_UNKNOWN)),
            "error_trend": str(metrics.get("error_trend", trend_status)),
            "stability_trend": str(metrics.get("stability_trend", TREND_WORSENING if policy_instability else TREND_UNKNOWN)),
        },
        "diagnosis": {
            "primary_skill_gaps": primary_skill_gaps,
            "diagnostic_signals": diagnostic_signals,
            "interpreted_failure_modes": interpreted_failure_modes,
        },
        "curriculum_strategy": strategy,
        "recommended_next_task_families": recommended_next_task_families,
        "avoid_next_task_families": avoid_next_task_families,
        "constraints_for_next_curriculum": constraints,
        "constraint_texts": _dedupe(constraint_texts),
        "llm_instruction_summary": instruction_summary,
        "current_task_shape_legacy": current_task_shape,
        "readiness_level": readiness_level,
        "recommended_next_tasks": _dedupe(recommendation_tasks),
        "avoid_next_tasks": _dedupe(avoid_tasks),
        "legacy_constraints_for_next_curriculum": _dedupe(constraint_texts),
        "rationale": strategy["rationale"],
    }


def _make_trace_record(
    step_index: int,
    episode_index: int,
    episode_step_index: int,
    action: Any,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one JSON-serializable policy evaluation trace row."""
    current_position = _array3(info.get("current_position"))
    reference_position = _array3(info.get("reference_position"))
    axis_error = current_position - reference_position
    position_error_m = float(np.linalg.norm(axis_error))
    reference_step_index = _int(info.get("reference_step_index", episode_step_index))
    record = {
        "step_index": int(step_index),
        "episode_index": int(episode_index),
        "episode_step_index": int(episode_step_index),
        "reference_step_index": reference_step_index,
        "time_sec": float(info.get("reference_time_sec", episode_step_index)),
        "current_position": current_position,
        "reference_position": reference_position,
        "actual_position_xyz_m": current_position,
        "reference_position_xyz_m": reference_position,
        "position_error_m": position_error_m,
        "axis_error_xyz": axis_error,
        "error_xyz_m": axis_error,
        "velocity": _array_to_jsonable(info.get("velocity", [])),
        "angular_velocity": _array_to_jsonable(info.get("angular_velocity", [])),
        "roll_pitch_yaw": _array_to_jsonable(info.get("roll_pitch_yaw", [])),
        "action": _array_to_jsonable(info.get("normalized_action", action)),
        "normalized_action": _array_to_jsonable(info.get("normalized_action", action)),
        "real_action": _array_to_jsonable(info.get("real_action", info.get("applied_action", action))),
        "actions_normalized": bool(info.get("action_normalized", False)),
        "action_interface": str(info.get("action_interface", "")),
        "real_action_type": str(info.get("real_action_type", "")),
        "ppo_action_dim": _int(info.get("ppo_action_dim")),
        "hover_rpm": _json_ready(info.get("hover_rpm")),
        "rpm_delta_scale": _json_ready(info.get("rpm_delta_scale")),
        "rpm_min": _json_ready(info.get("rpm_min")),
        "rpm_max": _json_ready(info.get("rpm_max")),
        "rpm_command_space_low": _array_to_jsonable(info.get("rpm_command_space_low", [])),
        "rpm_command_space_high": _array_to_jsonable(info.get("rpm_command_space_high", [])),
        "rpm_clipped": bool(info.get("rpm_clipped", False)),
        "rpm_saturation_mask": _array_to_jsonable(info.get("rpm_saturation_mask", [])),
        "real_motor_rpms": _array_to_jsonable(info.get("real_motor_rpms", [])),
        "action_clipped": bool(info.get("action_clipped", False)),
        "include_dynamics_observation": bool(info.get("include_dynamics_observation", False)),
        "include_previous_action": bool(info.get("include_previous_action", False)),
        "observation_dim": _int(info.get("observation_dim")),
        "observation_components": [dict(component) for component in info.get("observation_components", [])],
        "previous_action": _array_to_jsonable(info.get("previous_action", [])),
        "direct_control_limitations": list(info.get("direct_control_limitations", [])),
        "real_action_space_low": _array_to_jsonable(info.get("real_action_space_low", [])),
        "real_action_space_high": _array_to_jsonable(info.get("real_action_space_high", [])),
        **_pid_z_metadata_fields(info),
        **_initial_state_metadata_fields(info),
        "last_action": _array_to_jsonable(info.get("last_action", [])),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": str(info.get("termination_reason", "running")),
        "task_shape": str(info.get("task_shape", "unknown")),
        "start_hold_enabled": bool(info.get("start_hold_enabled", False)),
        "start_hold_sec": _float(info.get("start_hold_sec")),
        "exclude_start_hold_from_tracking_metrics": bool(info.get("exclude_start_hold_from_tracking_metrics", False)),
        "tracking_phase_start_step": _int(info.get("tracking_phase_start_step")),
        "tracking_phase_start_time_sec": _float(info.get("tracking_phase_start_time_sec")),
        "final_hold_enabled": bool(info.get("final_hold_enabled", False)),
        "final_hold_sec": _float(info.get("final_hold_sec")),
        "exclude_final_hold_from_tracking_metrics": bool(info.get("exclude_final_hold_from_tracking_metrics", False)),
        "tracking_phase_end_step": _int(info.get("tracking_phase_end_step")),
        "tracking_phase_end_time_sec": _float(info.get("tracking_phase_end_time_sec")),
        "is_start_hold": bool(info.get("is_start_hold", False)),
        "is_final_hold": bool(info.get("is_final_hold", False)),
        "is_tracking_phase": bool(
            info.get("is_tracking_phase", not bool(info.get("is_start_hold", False)) and not bool(info.get("is_final_hold", False)))
        ),
        "base_terminated": bool(info.get("base_terminated", False)),
        "base_truncated": bool(info.get("base_truncated", False)),
        "base_truncation_effective": bool(info.get("base_truncation_effective", info.get("base_truncated", False))),
        "base_truncation_ignored": bool(info.get("base_truncation_ignored", False)),
        "base_truncation_causes": list(info.get("base_truncation_causes", [])),
        "project_truncated": bool(info.get("project_truncated", False)),
        "project_truncation_causes": list(info.get("project_truncation_causes", [])),
        "strict_limit_violation": bool(info.get("strict_limit_violation", False)),
        "strict_limit_violations": list(info.get("strict_limit_violations", [])),
        "strict_limit_violation_count": _int(info.get("strict_limit_violation_count")),
        "termination_limits_mode": str(info.get("termination_limits_mode", "")),
        "termination_limits": _json_ready(info.get("termination_limits", {})),
        "diagnostic_limits": _json_ready(info.get("diagnostic_limits", {})),
        "base_truncation_policy": str(info.get("base_truncation_policy", "")),
        "terminate_on_base_truncation": bool(info.get("terminate_on_base_truncation", True)),
        "recovery_allowed_after_limit_violation": bool(info.get("recovery_allowed_after_limit_violation", False)),
        "recovery_steps_after_limit_violation": _int(info.get("recovery_steps_after_limit_violation")),
        "recovery_steps_limit": _int(info.get("recovery_steps_limit")),
        "base_reason_fields": dict(info.get("base_reason_fields", {})),
    }
    if info.get("applied_action") is not None:
        record["clipped_action"] = _array_to_jsonable(info.get("applied_action"))
        record["applied_action"] = record["clipped_action"]
    if info.get("reward_components") is not None:
        record["reward_components"] = _json_ready(info["reward_components"])
    return _json_ready(record)


def _episode_summaries(records: Sequence[Mapping[str, Any]], action_space: Any) -> list[dict[str, Any]]:
    """Return one summary per episode represented in trace records."""
    summaries: list[dict[str, Any]] = []
    episode_indices = sorted({_int(record.get("episode_index")) for record in records})
    for episode_index in episode_indices:
        episode_records = [record for record in records if _int(record.get("episode_index")) == episode_index]
        if episode_records:
            summaries.append(_summarize_episode(episode_records, action_space))
    return summaries


def _summarize_episode(records: Sequence[Mapping[str, Any]], action_space: Any) -> dict[str, Any]:
    """Summarize one evaluated episode from trace records."""
    positions = _array_field(records, "current_position")
    references = _array_field(records, "reference_position")
    rewards = np.asarray([_float(record.get("reward")) for record in records], dtype=float)
    errors = np.asarray([_float(record.get("position_error_m")) for record in records], dtype=float)
    tracking_errors = np.asarray([_float(record.get("position_error_m")) for record in _tracking_metric_records(records)], dtype=float)
    axis_errors = np.abs(positions - references)
    real_action_space = _real_action_space_from_records(records, fallback_action_space=action_space)
    action_metrics = _action_distribution_metrics(_array_field(records, "action"), action_space)
    real_action_metrics = _prefixed_action_distribution_metrics(
        _array_field(records, "real_action"),
        real_action_space,
        prefix="real_action",
    )
    action_audit_metrics = _action_audit_metrics(records=records, action_space=action_space, real_action_space=real_action_space)
    position_bounds = _position_bounds(positions)
    reference_bounds = _position_bounds(references)
    actual_xy_span_m = _xy_span(position_bounds)
    reference_xy_span_m = _xy_span(reference_bounds)
    roll_pitch = _array_field(records, "roll_pitch_yaw", min_width=ROLL_PITCH_DIMENSIONS)[:, :ROLL_PITCH_DIMENSIONS]
    velocities = _array_field(records, "velocity")[:, :POSITION_DIMENSIONS]
    angular_velocities = _array_field(records, "angular_velocity")[:, :POSITION_DIMENSIONS]
    return {
        "episode_index": _int(records[0].get("episode_index")),
        "start_step_index": _int(records[0].get("step_index")),
        "end_step_index": _int(records[-1].get("step_index")),
        "steps": len(records),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "final_reward": float(rewards[-1]),
        "mean_position_error_m": float(np.mean(errors)),
        "mean_position_error_tracking_m": float(np.mean(tracking_errors)),
        "final_position_error_m": float(errors[-1]),
        "max_position_error_m": float(np.max(errors)),
        **_start_hold_summary(records),
        **_initial_state_summary(records),
        "mean_abs_x_error_m": float(np.mean(axis_errors[:, 0])),
        "mean_abs_y_error_m": float(np.mean(axis_errors[:, 1])),
        "mean_abs_z_error_m": float(np.mean(axis_errors[:, 2])),
        "final_abs_x_error_m": float(axis_errors[-1, 0]),
        "final_abs_y_error_m": float(axis_errors[-1, 1]),
        "final_abs_z_error_m": float(axis_errors[-1, 2]),
        "reference_xy_span_m": reference_xy_span_m,
        "actual_xy_span_m": actual_xy_span_m,
        "xy_tracking_ratio": _xy_tracking_ratio(actual_xy_span_m, reference_xy_span_m),
        **action_metrics,
        **real_action_metrics,
        **action_audit_metrics,
        "z_min": _axis_min(position_bounds, axis=2),
        "z_max": _axis_max(position_bounds, axis=2),
        "max_abs_roll_pitch_rad": float(np.max(np.abs(roll_pitch))) if roll_pitch.size else 0.0,
        "max_speed_mps": _max_row_norm(velocities),
        "max_angular_velocity_radps": _max_row_norm(angular_velocities),
        "strict_limit_violation_count": _strict_limit_violation_count(records),
        "strict_limit_violation_causes": _unique_list_field(records, "strict_limit_violations"),
        "base_truncation_causes": _unique_list_field(records, "base_truncation_causes"),
        "project_truncation_causes": _unique_list_field(records, "project_truncation_causes"),
        "base_truncated_count": int(sum(1 for record in records if bool(record.get("base_truncated")))),
        "project_truncated_count": int(sum(1 for record in records if bool(record.get("project_truncated")))),
        "terminated": bool(records[-1].get("terminated")),
        "truncated": bool(records[-1].get("truncated")),
        "termination_reason": str(records[-1].get("termination_reason", "running")),
        "reset_after_episode": bool(records[-1].get("terminated")) or bool(records[-1].get("truncated")),
    }


def _overall_metrics(records: Sequence[Mapping[str, Any]], episode_summaries: Sequence[Mapping[str, Any]], action_space: Any) -> dict[str, Any]:
    """Return compact overall metrics derived from all trace records."""
    positions = _array_field(records, "current_position")
    references = _array_field(records, "reference_position")
    rewards = np.asarray([_float(record.get("reward")) for record in records], dtype=float)
    errors = np.asarray([_float(record.get("position_error_m")) for record in records], dtype=float)
    tracking_errors = np.asarray([_float(record.get("position_error_m")) for record in _tracking_metric_records(records)], dtype=float)
    axis_errors = np.abs(positions - references)
    position_bounds = _position_bounds(positions)
    reference_bounds = _position_bounds(references)
    actual_xy_span_m = _xy_span(position_bounds)
    reference_xy_span_m = _xy_span(reference_bounds)
    real_action_space = _real_action_space_from_records(records, fallback_action_space=action_space)
    action_metrics = _action_distribution_metrics(_array_field(records, "action"), action_space)
    real_action_metrics = _prefixed_action_distribution_metrics(
        _array_field(records, "real_action"),
        real_action_space,
        prefix="real_action",
    )
    action_boundary_metrics = _action_saturation_diagnostics(
        records=records,
        action_space=action_space,
        real_action_space=real_action_space,
    )
    action_audit_metrics = _action_audit_metrics(records=records, action_space=action_space, real_action_space=real_action_space)
    roll_pitch = _array_field(records, "roll_pitch_yaw", min_width=ROLL_PITCH_DIMENSIONS)[:, :ROLL_PITCH_DIMENSIONS]
    velocities = _array_field(records, "velocity")[:, :POSITION_DIMENSIONS]
    angular_velocities = _array_field(records, "angular_velocity")[:, :POSITION_DIMENSIONS]
    first_record = records[0]
    return {
        "mean_eval_reward": float(np.mean(rewards)),
        "final_eval_reward": float(rewards[-1]),
        "mean_position_error_m": float(np.mean(errors)),
        "mean_position_error_tracking_m": float(np.mean(tracking_errors)),
        "final_position_error_m": float(errors[-1]),
        "max_position_error_m": float(np.max(errors)),
        **_start_hold_summary(records),
        **_initial_state_summary(records),
        "position_bounds": position_bounds,
        "reference_position_bounds": reference_bounds,
        "action_bounds": _position_bounds(_array_field(records, "action")),
        "real_action_bounds": _position_bounds(_array_field(records, "real_action")),
        "actions_normalized": any(bool(record.get("actions_normalized", False)) for record in records),
        "actual_z_span_m": _axis_span(position_bounds, axis=2),
        "actual_z_min_m": _axis_min(position_bounds, axis=2),
        "actual_z_max_m": _axis_max(position_bounds, axis=2),
        "actual_xy_span_m": actual_xy_span_m,
        "reference_z_span_m": _axis_span(reference_bounds, axis=2),
        "reference_xy_span_m": reference_xy_span_m,
        "xy_tracking_ratio": _xy_tracking_ratio(actual_xy_span_m, reference_xy_span_m),
        "mean_abs_x_error": float(np.mean(axis_errors[:, 0])),
        "mean_abs_y_error": float(np.mean(axis_errors[:, 1])),
        "mean_abs_z_error": float(np.mean(axis_errors[:, 2])),
        "final_abs_x_error": float(axis_errors[-1, 0]),
        "final_abs_y_error": float(axis_errors[-1, 1]),
        "final_abs_z_error": float(axis_errors[-1, 2]),
        "episode_count": len(episode_summaries),
        "max_abs_roll_pitch_rad": float(np.max(np.abs(roll_pitch))) if roll_pitch.size else 0.0,
        "max_speed_mps": _max_row_norm(velocities),
        "max_angular_velocity_radps": _max_row_norm(angular_velocities),
        "strict_limit_violation_count": _strict_limit_violation_count(records),
        "strict_limit_violation_causes": _unique_list_field(records, "strict_limit_violations"),
        "base_truncated_count": int(sum(1 for record in records if bool(record.get("base_truncated")))),
        "base_truncation_effective_count": int(sum(1 for record in records if bool(record.get("base_truncation_effective")))),
        "base_truncation_ignored_count": int(sum(1 for record in records if bool(record.get("base_truncation_ignored")))),
        "base_truncation_causes": _unique_list_field(records, "base_truncation_causes"),
        "project_truncated_count": int(sum(1 for record in records if bool(record.get("project_truncated")))),
        "project_truncation_causes": _unique_list_field(records, "project_truncation_causes"),
        "recovery_allowed_after_limit_violation_count": int(
            sum(1 for record in records if bool(record.get("recovery_allowed_after_limit_violation")))
        ),
        "termination_limits_mode": str(first_record.get("termination_limits_mode", "")),
        "termination_limits": _json_ready(first_record.get("termination_limits", {})),
        "diagnostic_limits": _json_ready(first_record.get("diagnostic_limits", {})),
        "base_truncation_policy": str(first_record.get("base_truncation_policy", "")),
        "terminate_on_base_truncation": bool(first_record.get("terminate_on_base_truncation", True)),
        **_trace_action_metadata(records),
        **action_metrics,
        **real_action_metrics,
        **action_boundary_metrics,
        **action_audit_metrics,
    }


def _trace_action_metadata(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return action-interface metadata and direct-RPM clipping summaries."""
    first = records[0]
    metadata: dict[str, Any] = {
        "action_interface": str(first.get("action_interface", "")),
        "real_action_type": str(first.get("real_action_type", "")),
        "ppo_action_dim": _int(first.get("ppo_action_dim")),
        "include_dynamics_observation": bool(first.get("include_dynamics_observation", False)),
        "include_previous_action": bool(first.get("include_previous_action", False)),
        "observation_dim": _int(first.get("observation_dim")),
        "observation_components": [dict(component) for component in first.get("observation_components", [])],
        "direct_control_limitations": list(first.get("direct_control_limitations", [])),
    }
    metadata.update(_pid_z_metadata_fields(first))
    metadata.update(_initial_state_metadata_fields(first))
    for key in ("hover_rpm", "rpm_delta_scale", "rpm_min", "rpm_max", "rpm_command_space_low", "rpm_command_space_high"):
        value = first.get(key)
        if value is not None and value != []:
            metadata[key] = _json_ready(value)
    clipping_values = [bool(record.get("action_clipped", False)) or bool(record.get("rpm_clipped", False)) for record in records]
    if any(clipping_values):
        metadata["direct_rpm_clipping_fraction"] = float(np.mean(np.asarray(clipping_values, dtype=float)))
    saturation_rows = []
    for record in records:
        saturation = np.asarray(record.get("rpm_saturation_mask", []), dtype=float).reshape(-1)
        if saturation.size:
            saturation_rows.append(saturation)
    if saturation_rows:
        saturation_array = np.vstack(saturation_rows)
        saturation_fraction = [float(value) for value in np.mean(saturation_array, axis=0)]
        metadata["direct_rpm_saturation_fraction"] = saturation_fraction
        metadata["rpm_saturation_fraction_by_motor"] = saturation_fraction
    return metadata


def _action_audit_metrics(records: Sequence[Mapping[str, Any]], action_space: Any, real_action_space: Any) -> dict[str, Any]:
    """Return action metrics used to audit PID z-target and direct-RPM saturation."""
    action_values = _array_field(records, "action")
    real_action_values = _array_field(records, "real_action")
    action_distribution = _action_distribution_metrics(action_values, action_space)
    real_action_distribution = _prefixed_action_distribution_metrics(real_action_values, real_action_space, prefix="real_action")
    metrics: dict[str, Any] = {
        "action_mean_by_dim": action_distribution["action_mean"],
        "action_min_by_dim": action_distribution["action_min"],
        "action_max_by_dim": action_distribution["action_max"],
        "action_p95_by_dim": action_distribution["action_p95"],
        "action_saturation_fraction_by_dim": action_distribution["action_saturation_fraction"],
        "action_upper_saturation_fraction_by_dim": action_distribution["action_upper_saturation_fraction"],
        "action_lower_saturation_fraction_by_dim": action_distribution["action_lower_saturation_fraction"],
        "normalized_action_mean_by_dim": action_distribution["action_mean"],
        "normalized_action_min_by_dim": action_distribution["action_min"],
        "normalized_action_max_by_dim": action_distribution["action_max"],
        "normalized_action_p95_by_dim": action_distribution["action_p95"],
        "normalized_action_saturation_fraction_by_dim": action_distribution["action_saturation_fraction"],
        "normalized_action_upper_saturation_fraction_by_dim": action_distribution["action_upper_saturation_fraction"],
        "normalized_action_lower_saturation_fraction_by_dim": action_distribution["action_lower_saturation_fraction"],
        "real_action_mean_by_dim": real_action_distribution["real_action_mean"],
        "real_action_min_by_dim": real_action_distribution["real_action_min"],
        "real_action_max_by_dim": real_action_distribution["real_action_max"],
        "real_action_p95_by_dim": real_action_distribution["real_action_p95"],
        "real_action_saturation_fraction_by_dim": real_action_distribution["real_action_saturation_fraction"],
        "real_action_upper_saturation_fraction_by_dim": real_action_distribution["real_action_upper_saturation_fraction"],
        "real_action_lower_saturation_fraction_by_dim": real_action_distribution["real_action_lower_saturation_fraction"],
        **_phase_action_saturation_metrics(records=records, actions=action_values, action_space=action_space),
    }
    metrics.update(
        _pid_z_action_metrics(records=records, action_values=action_values, real_action_values=real_action_values, action_space=action_space)
    )
    metrics.update(_direct_rpm_action_metrics(records=records, real_action_values=real_action_values, real_action_space=real_action_space))
    return metrics


def _phase_action_saturation_metrics(records: Sequence[Mapping[str, Any]], actions: np.ndarray, action_space: Any) -> dict[str, list[float]]:
    """Return PPO-facing saturation fractions split by rollout phase."""
    return {
        "action_saturation_fraction_start_hold_by_dim": _masked_saturation_fraction(actions, action_space, _phase_mask(records, "start_hold")),
        "action_saturation_fraction_tracking_by_dim": _masked_saturation_fraction(actions, action_space, _phase_mask(records, "tracking")),
        "action_saturation_fraction_final_hold_by_dim": _masked_saturation_fraction(actions, action_space, _phase_mask(records, "final_hold")),
    }


def _pid_z_action_metrics(
    records: Sequence[Mapping[str, Any]],
    action_values: np.ndarray,
    real_action_values: np.ndarray,
    action_space: Any,
) -> dict[str, Any]:
    """Return PID z-target diagnostics for normalized action saturation audits."""
    if not records:
        return {}
    first = records[0]
    if str(first.get("action_interface", "")) != "pid_position" or str(first.get("real_action_type", "")) != "pid_target_position":
        return {}
    reachability_metrics = _pid_z_reachability_metrics(records)
    if action_values.shape[1] <= Z_AXIS_INDEX or real_action_values.shape[1] <= Z_AXIS_INDEX:
        return reachability_metrics
    tracking_mask = _phase_mask(records, "tracking")
    if not np.any(tracking_mask):
        return reachability_metrics
    positions = _array_field(records, "current_position")
    references = _array_field(records, "reference_position")
    velocities = _array_field(records, "velocity")
    z_error = positions[tracking_mask, Z_AXIS_INDEX] - references[tracking_mask, Z_AXIS_INDEX]
    z_error_abs = np.abs(z_error)
    z_target_minus_reference = real_action_values[tracking_mask, Z_AXIS_INDEX] - references[tracking_mask, Z_AXIS_INDEX]
    _, upper_fraction = _saturation_fraction_by_side(action_values[tracking_mask], action_space)
    vertical_velocity = (
        velocities[tracking_mask, Z_AXIS_INDEX] if velocities.shape[1] > Z_AXIS_INDEX else np.zeros(int(np.sum(tracking_mask)), dtype=float)
    )
    return {
        **reachability_metrics,
        "z_action_upper_saturation_fraction_tracking": _fraction_at(upper_fraction, Z_AXIS_INDEX),
        "z_target_minus_reference_mean": float(np.mean(z_target_minus_reference)),
        "z_target_minus_reference_p95": float(np.percentile(z_target_minus_reference, 95)),
        "z_error_mean_tracking": float(np.mean(z_error_abs)),
        "z_error_p95_tracking": float(np.percentile(z_error_abs, 95)),
        "z_overshoot_fraction_tracking": float(np.mean(z_error > 0.0)),
        "vertical_velocity_mean_tracking": float(np.mean(vertical_velocity)),
        "vertical_velocity_p95_abs_tracking": float(np.percentile(np.abs(vertical_velocity), 95)),
    }


def _initial_state_metadata_fields(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return initial-state fields present in a trace or metrics mapping."""
    return {key: _json_ready(source[key]) for key in INITIAL_STATE_KEYS if key in source}


def _pid_z_metadata_fields(source: Mapping[str, Any]) -> dict[str, Any]:
    """Return PID z reachability fields present in a trace or metrics mapping."""
    return {key: _json_ready(source[key]) for key in PID_Z_REACHABILITY_KEYS if key in source}


def _pid_z_reachability_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return PID z reference reachability metrics from trace records."""
    first = records[0]
    references = _array_field(records, "reference_position")
    reference_z = references[:, Z_AXIS_INDEX] if references.shape[1] > Z_AXIS_INDEX else np.asarray([], dtype=float)
    reference_z_min = _first_numeric_field(records, "reference_z_min")
    reference_z_max = _first_numeric_field(records, "reference_z_max")
    if reference_z.size:
        reference_z_min = float(np.min(reference_z)) if reference_z_min is None else min(reference_z_min, float(np.min(reference_z)))
        reference_z_max = float(np.max(reference_z)) if reference_z_max is None else max(reference_z_max, float(np.max(reference_z)))
    real_low_z = _first_numeric_field(records, "real_pid_z_target_low")
    real_high_z = _first_numeric_field(records, "real_pid_z_target_high")
    if real_low_z is None:
        real_low_z = _record_space_axis_bound(first, "real_action_space_low", Z_AXIS_INDEX)
    if real_high_z is None:
        real_high_z = _record_space_axis_bound(first, "real_action_space_high", Z_AXIS_INDEX)
    if real_low_z is None or real_high_z is None or reference_z_min is None or reference_z_max is None:
        return _pid_z_metadata_fields(first)
    above_margin = max(0.0, reference_z_max - real_high_z)
    below_margin = max(0.0, real_low_z - reference_z_min)
    metrics = _pid_z_metadata_fields(first)
    metrics.update(
        {
            "real_pid_z_target_low": float(real_low_z),
            "real_pid_z_target_high": float(real_high_z),
            "reference_z_min": float(reference_z_min),
            "reference_z_max": float(reference_z_max),
            "reference_z_reachable_by_pid_position": bool(
                above_margin <= TARGET_BOUNDARY_ACTION_TOLERANCE_M and below_margin <= TARGET_BOUNDARY_ACTION_TOLERANCE_M
            ),
            "z_reference_above_pid_high_margin": float(above_margin),
            "z_reference_below_pid_low_margin": float(below_margin),
        }
    )
    if "pid_target_z_min_m" not in metrics:
        metrics["pid_target_z_min_m"] = float(real_low_z)
    if "pid_target_z_max_m" not in metrics:
        metrics["pid_target_z_max_m"] = float(real_high_z)
    if "real_pid_z_target_for_normalized_action2_minus1" not in metrics:
        metrics["real_pid_z_target_for_normalized_action2_minus1"] = float(real_low_z)
    if "real_pid_z_target_for_normalized_action2_zero" not in metrics:
        metrics["real_pid_z_target_for_normalized_action2_zero"] = float((real_low_z + real_high_z) / 2.0)
    if "real_pid_z_target_for_normalized_action2_plus1" not in metrics:
        metrics["real_pid_z_target_for_normalized_action2_plus1"] = float(real_high_z)
    return metrics


def _first_numeric_field(records: Sequence[Mapping[str, Any]], key: str) -> float | None:
    """Return the first finite numeric trace field value."""
    for record in records:
        value = record.get(key)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            return float(value)
    return None


def _record_space_axis_bound(record: Mapping[str, Any], key: str, axis: int) -> float | None:
    """Return an action-space bound from one JSON-ready record."""
    values = np.asarray(record.get(key, []), dtype=float).reshape(-1)
    if values.size <= axis or not np.isfinite(values[axis]):
        return None
    return float(values[axis])


def _direct_rpm_action_metrics(records: Sequence[Mapping[str, Any]], real_action_values: np.ndarray, real_action_space: Any) -> dict[str, Any]:
    """Return direct-RPM motor-specific saturation and clipping diagnostics."""
    if not records:
        return {}
    first = records[0]
    if str(first.get("action_interface", "")) != "direct_rpm" or str(first.get("real_action_type", "")) != "motor_rpm":
        return {}
    lower_fraction, upper_fraction = _saturation_fraction_by_side(real_action_values, real_action_space)
    saturation_fraction = _saturation_fraction(real_action_values, real_action_space)
    clipped = np.asarray([bool(record.get("rpm_clipped", False)) for record in records], dtype=float)
    metrics: dict[str, Any] = {
        "rpm_saturation_fraction_by_motor": [float(value) for value in saturation_fraction],
        "rpm_upper_saturation_fraction_by_motor": [float(value) for value in upper_fraction],
        "rpm_lower_saturation_fraction_by_motor": [float(value) for value in lower_fraction],
        "rpm_clipped_fraction": float(np.mean(clipped)) if clipped.size else 0.0,
    }
    saturation_rows = []
    for record in records:
        saturation = np.asarray(record.get("rpm_saturation_mask", []), dtype=float).reshape(-1)
        if saturation.size:
            saturation_rows.append(saturation)
    if saturation_rows:
        metrics["rpm_saturation_fraction_by_motor"] = [float(value) for value in np.mean(np.vstack(saturation_rows), axis=0)]
    return metrics


def _phase_mask(records: Sequence[Mapping[str, Any]], phase: str) -> np.ndarray:
    """Return a boolean mask for start-hold, tracking, or final-hold trace rows."""
    if phase == "start_hold":
        return np.asarray([bool(record.get("is_start_hold", False)) for record in records], dtype=bool)
    if phase == "final_hold":
        return np.asarray([bool(record.get("is_final_hold", False)) for record in records], dtype=bool)
    if phase == "tracking":
        return np.asarray(
            [
                bool(record.get("is_tracking_phase", not bool(record.get("is_start_hold", False)) and not bool(record.get("is_final_hold", False))))
                for record in records
            ],
            dtype=bool,
        )
    message = f"unsupported trace phase: {phase}"
    raise ValueError(message)


def _masked_saturation_fraction(actions: np.ndarray, action_space: Any, mask: np.ndarray) -> list[float]:
    """Return per-dimension saturation for masked rows, or an empty list when the phase is absent."""
    if actions.size == 0 or not np.any(mask):
        return []
    return [float(value) for value in _saturation_fraction(actions[mask], action_space)]


def _action_saturation_diagnostics(records: Sequence[Mapping[str, Any]], action_space: Any, real_action_space: Any) -> dict[str, Any]:
    """Return structured action-saturation metadata for failure classification."""
    action_values = _array_field(records, "action")
    real_action_values = _array_field(records, "real_action")
    action_saturation_fraction = _saturation_fraction(action_values, action_space)
    real_action_saturation_fraction = _saturation_fraction(real_action_values, real_action_space)
    action_dimensions = _dimension_indices_at_threshold(action_saturation_fraction)
    real_action_dimensions = _dimension_indices_at_threshold(real_action_saturation_fraction)
    expected_details = _expected_target_boundary_action_dimensions(
        records=records,
        real_action_values=real_action_values,
        real_action_space=real_action_space,
        action_dimensions=action_dimensions,
        action_saturation_fraction=action_saturation_fraction,
        real_action_saturation_fraction=real_action_saturation_fraction,
    )
    expected_dimensions = {int(detail["dimension"]) for detail in expected_details}
    problematic_dimensions = [dimension for dimension in action_dimensions if dimension not in expected_dimensions]
    diagnostic = {
        "saturated_dimensions": action_dimensions,
        "real_saturated_dimensions": real_action_dimensions,
        "expected_target_boundary_action": bool(expected_details),
        "expected_target_boundary_dimensions": expected_details,
        "problematic_action_saturation_dimensions": problematic_dimensions,
    }
    return {
        "action_saturation_diagnostic": diagnostic,
        "action_saturation_dimensions": action_dimensions,
        "real_action_saturation_dimensions": real_action_dimensions,
        "expected_target_boundary_action": bool(expected_details),
        "expected_target_boundary_action_dimensions": expected_details,
        "problematic_action_saturation_dimensions": problematic_dimensions,
    }


def _expected_target_boundary_action_dimensions(
    records: Sequence[Mapping[str, Any]],
    real_action_values: np.ndarray,
    real_action_space: Any,
    action_dimensions: Sequence[int],
    action_saturation_fraction: np.ndarray,
    real_action_saturation_fraction: np.ndarray,
) -> list[dict[str, Any]]:
    """Return PID target-position dimensions whose saturated bound equals the reference target."""
    if not records:
        return []
    first = records[0]
    if str(first.get("action_interface", "")) != "pid_position" or str(first.get("real_action_type", "")) != "pid_target_position":
        return []
    low = np.asarray(getattr(real_action_space, "low", []), dtype=float).reshape(-1)
    high = np.asarray(getattr(real_action_space, "high", []), dtype=float).reshape(-1)
    if low.size < POSITION_DIMENSIONS or high.size < POSITION_DIMENSIONS:
        return []
    real_actions = np.asarray(real_action_values, dtype=float).reshape(len(records), -1)
    references = _array_field(records, "reference_position")[:, :POSITION_DIMENSIONS]
    details: list[dict[str, Any]] = []
    for dimension in action_dimensions:
        if dimension >= POSITION_DIMENSIONS or dimension >= real_actions.shape[1]:
            continue
        low_fraction = _values_at_bound_fraction(real_actions[:, dimension], low[dimension])
        high_fraction = _values_at_bound_fraction(real_actions[:, dimension], high[dimension])
        if low_fraction >= ACTION_SATURATION_FRACTION_MIN and _reference_matches_bound(references[:, dimension], low[dimension]):
            details.append(
                _target_boundary_detail(
                    dimension=dimension,
                    bound_name="low",
                    bound_value=low[dimension],
                    reference_values=references[:, dimension],
                    action_saturation_fraction=action_saturation_fraction,
                    real_action_saturation_fraction=real_action_saturation_fraction,
                    real_action_bound_fraction=low_fraction,
                )
            )
        if high_fraction >= ACTION_SATURATION_FRACTION_MIN and _reference_matches_bound(references[:, dimension], high[dimension]):
            details.append(
                _target_boundary_detail(
                    dimension=dimension,
                    bound_name="high",
                    bound_value=high[dimension],
                    reference_values=references[:, dimension],
                    action_saturation_fraction=action_saturation_fraction,
                    real_action_saturation_fraction=real_action_saturation_fraction,
                    real_action_bound_fraction=high_fraction,
                )
            )
    return details


def _target_boundary_detail(
    dimension: int,
    bound_name: str,
    bound_value: float,
    reference_values: np.ndarray,
    action_saturation_fraction: np.ndarray,
    real_action_saturation_fraction: np.ndarray,
    real_action_bound_fraction: float,
) -> dict[str, Any]:
    """Return JSON-ready metadata for one expected target-boundary action dimension."""
    return {
        "mode": DIAGNOSTIC_EXPECTED_TARGET_BOUNDARY_ACTION,
        "dimension": int(dimension),
        "axis": _action_axis_name(dimension),
        "bound": bound_name,
        "bound_value": float(bound_value),
        "reference_min": float(np.min(reference_values)),
        "reference_max": float(np.max(reference_values)),
        "action_saturation_fraction": _fraction_at(action_saturation_fraction, dimension),
        "real_action_saturation_fraction": _fraction_at(real_action_saturation_fraction, dimension),
        "real_action_bound_fraction": float(real_action_bound_fraction),
    }


def _values_at_bound_fraction(values: np.ndarray, bound: float) -> float:
    """Return the fraction of values that sit on one numeric bound."""
    if values.size == 0:
        return 0.0
    return float(np.mean(np.isclose(values, bound, atol=TARGET_BOUNDARY_ACTION_TOLERANCE_M, rtol=0.0)))


def _reference_matches_bound(values: np.ndarray, bound: float) -> bool:
    """Return whether all reference values in one dimension equal a real-action bound."""
    return bool(values.size) and bool(np.all(np.isclose(values, bound, atol=TARGET_BOUNDARY_ACTION_TOLERANCE_M, rtol=0.0)))


def _action_axis_name(dimension: int) -> str:
    """Return a human-readable PID target-position axis name."""
    if 0 <= dimension < len(ACTION_AXIS_NAMES):
        return ACTION_AXIS_NAMES[dimension]
    return f"dim_{dimension}"


def _saturation_fraction(actions: np.ndarray, action_space: Any) -> np.ndarray:
    """Return per-dimension fraction of actions that sit on their action-space bounds."""
    near_low, near_high = _saturation_masks(actions, action_space)
    if near_low.shape[0] == 0:
        return np.zeros(near_low.shape[1], dtype=float)
    return np.mean(np.logical_or(near_low, near_high), axis=0)


def _saturation_fraction_by_side(actions: np.ndarray, action_space: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return lower-bound and upper-bound saturation fractions by action dimension."""
    near_low, near_high = _saturation_masks(actions, action_space)
    if near_low.shape[0] == 0:
        return np.zeros(near_low.shape[1], dtype=float), np.zeros(near_high.shape[1], dtype=float)
    return np.mean(near_low, axis=0), np.mean(near_high, axis=0)


def _saturation_masks(actions: np.ndarray, action_space: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return lower-bound and upper-bound saturation masks for an action array."""
    action_input = np.asarray(actions, dtype=float)
    if action_input.ndim == 1:
        action_array = action_input.reshape(1, -1)
    elif action_input.size == 0 and action_input.ndim == ARRAY_BOUNDS_MAX_NDIM:
        action_array = action_input.reshape(0, action_input.shape[1])
    elif action_input.size == 0:
        action_array = np.zeros((0, 0), dtype=float)
    else:
        action_array = action_input.reshape(action_input.shape[0], -1)
    low = np.asarray(getattr(action_space, "low", []), dtype=float).reshape(-1)
    high = np.asarray(getattr(action_space, "high", []), dtype=float).reshape(-1)
    if low.size != action_array.shape[1] or high.size != action_array.shape[1]:
        zeros = np.zeros(action_array.shape, dtype=bool)
        return zeros, zeros
    near_low = np.isclose(action_array, low, atol=ACTION_SATURATION_TOLERANCE, rtol=0.0)
    near_high = np.isclose(action_array, high, atol=ACTION_SATURATION_TOLERANCE, rtol=0.0)
    return near_low, near_high


def _dimension_indices_at_threshold(fractions: np.ndarray) -> list[int]:
    """Return dimensions whose saturation fraction reaches the failure threshold."""
    return [int(index) for index, value in enumerate(np.asarray(fractions, dtype=float).reshape(-1)) if value >= ACTION_SATURATION_FRACTION_MIN]


def _fraction_at(fractions: np.ndarray, dimension: int) -> float:
    """Return one saturation fraction or zero when the dimension is unavailable."""
    values = np.asarray(fractions, dtype=float).reshape(-1)
    if dimension >= values.size:
        return 0.0
    return float(values[dimension])


def _validate_trace_consistency(records: Sequence[Mapping[str, Any]]) -> None:
    """Validate deterministic trace alignment before metrics or plots consume it."""
    task_shapes = {str(record.get("task_shape", "unknown")) for record in records if record.get("task_shape") is not None}
    if len(task_shapes) > 1:
        message = f"trace contains multiple task shapes: {', '.join(sorted(task_shapes))}"
        raise ValueError(message)

    for row_index, record in enumerate(records):
        actual = _trace_position_row(record, "actual_position_xyz_m", "current_position")
        reference = _trace_position_row(record, "reference_position_xyz_m", "reference_position")
        expected_error = float(np.linalg.norm(actual - reference))
        reported_error = _float(record.get("position_error_m"))
        if not np.isclose(reported_error, expected_error, atol=1.0e-9, rtol=1.0e-9):
            message = f"trace row {row_index} position_error_m does not match same-row actual/reference positions"
            raise ValueError(message)

    for episode_index in sorted({_int(record.get("episode_index")) for record in records}):
        episode_records = [record for record in records if _int(record.get("episode_index")) == episode_index]
        episode_steps = np.asarray([_int(record.get("episode_step_index", record.get("step_index"))) for record in episode_records], dtype=int)
        reference_steps = np.asarray(
            [_int(record.get("reference_step_index", record.get("episode_step_index", record.get("step_index")))) for record in episode_records],
            dtype=int,
        )
        times = np.asarray([_float(record.get("time_sec")) for record in episode_records], dtype=float)
        if episode_steps.shape[0] > 1 and np.any(np.diff(episode_steps) <= 0):
            message = f"episode_step_index must increase within episode {episode_index}"
            raise ValueError(message)
        if reference_steps.shape[0] > 1 and np.any(np.diff(reference_steps) < 0):
            message = f"reference_step_index must be monotonic within episode {episode_index}"
            raise ValueError(message)
        if times.shape[0] > 1 and np.any(np.diff(times) < 0.0):
            message = f"time_sec must be monotonic within episode {episode_index}"
            raise ValueError(message)


def _validate_trace_task_shape(records: Sequence[Mapping[str, Any]], task_shape: str) -> None:
    """Raise when trace rows identify a different task than the evaluated task."""
    if str(task_shape).strip().lower() in {"", "none", "unknown"}:
        return
    trace_shapes = {str(record.get("task_shape")) for record in records if record.get("task_shape") not in (None, "unknown")}
    if not trace_shapes:
        return
    if trace_shapes == {str(task_shape)}:
        return
    message = f"trace task_shape mismatch: expected {task_shape!r}, got {', '.join(sorted(trace_shapes))}"
    raise ValueError(message)


def _trace_position_row(record: Mapping[str, Any], primary_key: str, fallback_key: str) -> np.ndarray:
    """Return one strict XYZ trace position row for consistency checks."""
    value = record.get(primary_key, record.get(fallback_key))
    row = np.asarray(value, dtype=float).reshape(-1)
    if row.shape != (POSITION_DIMENSIONS,):
        message = f"trace field {primary_key} must contain exactly {POSITION_DIMENSIONS} values"
        raise ValueError(message)
    if not np.all(np.isfinite(row)):
        message = f"trace field {primary_key} must contain only finite values"
        raise ValueError(message)
    return row


def _tracking_metric_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    """Return records used for tracking-only metrics."""
    if not any(
        bool(record.get("exclude_start_hold_from_tracking_metrics", False)) or bool(record.get("exclude_final_hold_from_tracking_metrics", False))
        for record in records
    ):
        return list(records)
    filtered = [record for record in records if bool(record.get("is_tracking_phase", True))]
    return filtered or list(records)


def _start_hold_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return start-hold metadata shared by one trace or episode."""
    first = records[0]
    return {
        "start_hold_enabled": bool(first.get("start_hold_enabled", False)),
        "start_hold_sec": _float(first.get("start_hold_sec")),
        "exclude_start_hold_from_tracking_metrics": bool(first.get("exclude_start_hold_from_tracking_metrics", False)),
        "tracking_phase_start_step": _int(first.get("tracking_phase_start_step")),
        "tracking_phase_start_time_sec": _float(first.get("tracking_phase_start_time_sec")),
        "final_hold_enabled": bool(first.get("final_hold_enabled", False)),
        "final_hold_sec": _float(first.get("final_hold_sec")),
        "exclude_final_hold_from_tracking_metrics": bool(first.get("exclude_final_hold_from_tracking_metrics", False)),
        "tracking_phase_end_step": _int(first.get("tracking_phase_end_step")),
        "tracking_phase_end_time_sec": _float(first.get("tracking_phase_end_time_sec")),
    }


def _initial_state_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return initial-state metadata shared by one trace or episode."""
    if not records:
        return {}
    return _initial_state_metadata_fields(records[0])


def _prefixed_action_distribution_metrics(actions: np.ndarray, action_space: Any, prefix: str) -> dict[str, list[float]]:
    """Return action distribution metrics with a custom key prefix."""
    metrics = _action_distribution_metrics(actions, action_space)
    return {key.replace("action_", f"{prefix}_", 1): value for key, value in metrics.items()}


def _real_action_space_from_records(records: Sequence[Mapping[str, Any]], fallback_action_space: Any) -> Any:
    """Build a Box-like real action space from trace info when normalized wrappers record it."""
    for record in records:
        low = np.asarray(record.get("real_action_space_low", []), dtype=float)
        high = np.asarray(record.get("real_action_space_high", []), dtype=float)
        if low.size and low.shape == high.shape:
            return _ArrayActionSpace(low=low, high=high)
    return fallback_action_space


@dataclass(frozen=True)
class _ArrayActionSpace:
    """Small Box-like action-space stand-in used for diagnostic saturation checks."""

    low: np.ndarray
    high: np.ndarray


def _action_distribution_metrics(actions: np.ndarray, action_space: Any) -> dict[str, list[float]]:
    """Return per-dimension action distribution and saturation diagnostics."""
    if actions.size == 0:
        return {
            "action_mean": [],
            "action_std": [],
            "action_min": [],
            "action_max": [],
            "action_p95": [],
            "action_saturation_fraction": [],
            "action_upper_saturation_fraction": [],
            "action_lower_saturation_fraction": [],
        }
    action_array = np.asarray(actions, dtype=float).reshape(actions.shape[0], -1)
    lower_fraction, upper_fraction = _saturation_fraction_by_side(action_array, action_space)
    saturation_fraction = _saturation_fraction(action_array, action_space)
    return {
        "action_mean": [float(value) for value in np.mean(action_array, axis=0)],
        "action_std": [float(value) for value in np.std(action_array, axis=0)],
        "action_min": [float(value) for value in np.min(action_array, axis=0)],
        "action_max": [float(value) for value in np.max(action_array, axis=0)],
        "action_p95": [float(value) for value in np.percentile(action_array, 95, axis=0)],
        "action_saturation_fraction": [float(value) for value in saturation_fraction],
        "action_upper_saturation_fraction": [float(value) for value in upper_fraction],
        "action_lower_saturation_fraction": [float(value) for value in lower_fraction],
    }


def _action_saturation_is_failure(metrics: Mapping[str, Any], tracking_acceptable: bool) -> bool:
    """Return whether saturated PPO actions should count as a failure mode."""
    saturated_dimensions = _dimension_indices_at_threshold(np.asarray(_float_list(metrics.get("action_saturation_fraction")), dtype=float))
    if not saturated_dimensions:
        return False
    if not tracking_acceptable:
        return True
    diagnostic = metrics.get("action_saturation_diagnostic")
    if not isinstance(diagnostic, dict):
        return True
    raw_problematic_dimensions = diagnostic.get("problematic_action_saturation_dimensions")
    if isinstance(raw_problematic_dimensions, list):
        return bool(raw_problematic_dimensions)
    expected_dimensions = {
        _int(detail.get("dimension")) for detail in diagnostic.get("expected_target_boundary_dimensions", []) if isinstance(detail, dict)
    }
    return any(dimension not in expected_dimensions for dimension in saturated_dimensions)


def _tracking_acceptable(metrics: Mapping[str, Any]) -> bool:
    """Return whether tracking metrics look acceptable when no major failure is detected."""
    mean_error = _float(metrics.get("mean_position_error_tracking_m", metrics.get("mean_position_error_m")))
    return mean_error <= ACCEPTABLE_MEAN_POSITION_ERROR_M and _float(metrics.get("final_position_error_m")) <= ACCEPTABLE_FINAL_POSITION_ERROR_M


def _overall_status(failure_modes: Sequence[str], tracking_acceptable: bool) -> str:
    """Return successful, partial, or failed from detected failure modes."""
    if failure_modes == [FAILURE_NONE]:
        return "successful"
    if failure_modes:
        return "failed"
    if tracking_acceptable:
        return "successful"
    return "partial"


def _failure_evidence(metrics: Mapping[str, Any], episode_summaries: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return compact evidence values supporting failure classification."""
    return {
        "mean_position_error_m": _float(metrics.get("mean_position_error_m")),
        "mean_position_error_tracking_m": _float(metrics.get("mean_position_error_tracking_m", metrics.get("mean_position_error_m"))),
        "final_position_error_m": _float(metrics.get("final_position_error_m")),
        "max_position_error_m": _float(metrics.get("max_position_error_m")),
        "reference_xy_span_m": _float(metrics.get("reference_xy_span_m")),
        "actual_xy_span_m": _float(metrics.get("actual_xy_span_m")),
        "xy_tracking_ratio": metrics.get("xy_tracking_ratio"),
        "action_saturation_fraction": _float_list(metrics.get("action_saturation_fraction")),
        "action_saturation_diagnostic": _json_ready(metrics.get("action_saturation_diagnostic", {})),
        "action_saturation_dimensions": _json_ready(metrics.get("action_saturation_dimensions", [])),
        "actions_normalized": bool(metrics.get("actions_normalized", False)),
        "real_action_saturation_fraction": _float_list(metrics.get("real_action_saturation_fraction")),
        "real_action_saturation_dimensions": _json_ready(metrics.get("real_action_saturation_dimensions", [])),
        "expected_target_boundary_action": bool(metrics.get("expected_target_boundary_action", False)),
        "expected_target_boundary_action_dimensions": _json_ready(metrics.get("expected_target_boundary_action_dimensions", [])),
        "problematic_action_saturation_dimensions": _json_ready(metrics.get("problematic_action_saturation_dimensions", [])),
        "eval_terminated_count": _int(metrics.get("eval_terminated_count")),
        "eval_truncated_count": _int(metrics.get("eval_truncated_count")),
        "strict_limit_violation_count": _int(metrics.get("strict_limit_violation_count")),
        "strict_limit_violation_causes": _json_ready(metrics.get("strict_limit_violation_causes", [])),
        "base_truncation_causes": _json_ready(metrics.get("base_truncation_causes", [])),
        "project_truncation_causes": _json_ready(metrics.get("project_truncation_causes", [])),
        "base_truncated_count": _int(metrics.get("base_truncated_count")),
        "project_truncated_count": _int(metrics.get("project_truncated_count")),
        "termination_limits_mode": str(metrics.get("termination_limits_mode", "")),
        "base_truncation_policy": str(metrics.get("base_truncation_policy", "")),
        "actual_z_span_m": _float(metrics.get("actual_z_span_m")),
        "mean_abs_z_error": _float(metrics.get("mean_abs_z_error")),
        "max_abs_roll_pitch_rad": _max_episode_value(episode_summaries, "max_abs_roll_pitch_rad"),
        "max_speed_mps": _float(metrics.get("max_speed_mps")),
        "max_angular_velocity_radps": _float(metrics.get("max_angular_velocity_radps")),
    }


def _failure_summary(primary_failure_mode: str, overall_status: str, evidence: Mapping[str, Any]) -> str:
    """Return a short human-readable failure summary."""
    if primary_failure_mode == FAILURE_NONE:
        return "Evaluation tracking is acceptable and no major failure mode was detected."
    return (
        f"Evaluation classified as {overall_status}; primary failure mode is {primary_failure_mode}. "
        f"Mean error={_float(evidence.get('mean_position_error_m')):.3f} m, "
        f"reference XY span={_float(evidence.get('reference_xy_span_m')):.3f} m, "
        f"actual XY span={_float(evidence.get('actual_xy_span_m')):.3f} m."
    )


def _curriculum_rationale(readiness_level: str, failure_modes: set[str], metrics: Mapping[str, Any]) -> str:
    """Return a concise curriculum-feedback rationale."""
    modes = ", ".join(sorted(failure_modes)) if failure_modes else "tracking_error_without_specific_failure"
    return (
        f"Readiness is {readiness_level} because evaluation modes were {modes}; "
        f"mean tracking error was {_float(metrics.get('mean_position_error_tracking_m', metrics.get('mean_position_error_m'))):.3f} m."
    )


def _task_family_from_shape(task_shape: str) -> str:
    """Return the closest supported task-family name for a task shape."""
    normalized = str(task_shape).strip().lower()
    mapping = {
        "": "unknown",
        "hover": FAMILY_HOVER,
        "hover_stabilization": FAMILY_HOVER,
        "takeoff": FAMILY_TAKEOFF,
        "vertical": FAMILY_TAKEOFF,
        "takeoff_stabilization": FAMILY_TAKEOFF,
        "line": FAMILY_LINE,
        "short_slow_line": FAMILY_LINE,
        "long_line": FAMILY_LINE,
        "start_hold_then_line": FAMILY_START_HOLD_LINE,
        "start_hold_then_short_line": FAMILY_START_HOLD_LINE,
        "polyline": FAMILY_POLYLINE,
        "fast_polyline": FAMILY_POLYLINE,
        "l_shape": FAMILY_L_SHAPE,
        "zigzag": FAMILY_ZIGZAG,
        "multi_height_polyline": FAMILY_MULTI_HEIGHT_POLYLINE,
        "circle": FAMILY_CIRCLE,
        "slow_circle": FAMILY_CIRCLE,
        "ellipse": FAMILY_ELLIPSE,
        "figure_eight": FAMILY_FIGURE_EIGHT,
    }
    return mapping.get(normalized, normalized or "unknown")


def _trend_status(metrics: Mapping[str, Any]) -> str:
    """Return the compact trend status available to curriculum feedback."""
    trend = str(metrics.get("trend_status") or metrics.get("position_error_trend") or metrics.get("error_trend") or TREND_UNKNOWN)
    return trend if trend in {TREND_IMPROVING, TREND_WORSENING, TREND_FLAT, TREND_UNKNOWN} else TREND_UNKNOWN


def _task_difficulty_level(failure_modes: set[str], metrics: Mapping[str, Any]) -> str:
    """Return a coarse stage-relative difficulty label from diagnostics."""
    mean_error = _float(metrics.get("mean_position_error_tracking_m", metrics.get("mean_position_error_m")))
    if FAILURE_REFERENCE_TOO_HARD in failure_modes or mean_error >= HIGH_MEAN_POSITION_ERROR_M:
        return "high"
    if failure_modes and FAILURE_NONE not in failure_modes:
        return "medium"
    if mean_error <= ACCEPTABLE_MEAN_POSITION_ERROR_M:
        return "low"
    return "medium"


def _diagnostic_signal_counts(failure_modes: set[str], metrics: Mapping[str, Any]) -> dict[str, int]:
    """Return compact counts for core diagnostic signal families."""
    action_saturations = _float_list(metrics.get("action_saturation_fraction")) + _float_list(metrics.get("real_action_saturation_fraction"))
    max_action_saturation = max(action_saturations, default=0.0)
    attitude_value = _float(metrics.get("max_abs_roll_pitch_rad"))
    return {
        "pid_z_unreachable_reference_count": int(_pid_reference_unreachable(metrics)),
        "z_instability_count": int(
            FAILURE_Z_INSTABILITY in failure_modes
            or _float(metrics.get("actual_z_span_m")) >= Z_INSTABILITY_SPAN_M
            or _float(metrics.get("mean_abs_z_error")) >= Z_INSTABILITY_MEAN_ABS_ERROR_M
        ),
        "action_saturation_count": int(FAILURE_ACTION_SATURATION in failure_modes or max_action_saturation >= ACTION_SATURATION_FRACTION_MIN),
        "attitude_instability_count": int(
            FAILURE_ATTITUDE_INSTABILITY in failure_modes
            or attitude_value >= ATTITUDE_INSTABILITY_MAX_ABS_ROLL_PITCH_RAD
            or _int(metrics.get("strict_limit_violation_count")) >= STRICT_LIMIT_VIOLATION_COUNT_MIN
        ),
        "reference_too_fast_or_too_hard_count": int(FAILURE_REFERENCE_TOO_HARD in failure_modes),
    }


def _primary_skill_gaps(
    *,
    current_task_family: str,
    failure_modes: set[str],
    metrics: Mapping[str, Any],
    diagnostic_signals: Mapping[str, int],
) -> list[str]:
    """Map diagnostic signals to constructive curriculum skill gaps."""
    gaps: list[str] = []
    if diagnostic_signals.get("z_instability_count", 0) > 0 and diagnostic_signals.get("pid_z_unreachable_reference_count", 0) == 0:
        gaps.append(SKILL_ALTITUDE_CONTROL)
    if failure_modes & {FAILURE_HOVER_LOCK, FAILURE_INSUFFICIENT_XY_MOTION, FAILURE_OVERSHOOT} or _xy_tracking_weak(metrics):
        gaps.append(SKILL_XY_TRACKING)
    if FAILURE_REFERENCE_TOO_HARD in failure_modes or (
        diagnostic_signals.get("action_saturation_count", 0) > 0 and not _tracking_acceptable(metrics)
    ):
        gaps.append(SKILL_SPEED_CONTROL)
    if current_task_family in {FAMILY_POLYLINE, FAMILY_L_SHAPE, FAMILY_ZIGZAG} and _tracking_error_high(metrics):
        gaps.extend([SKILL_TURN_FOLLOWING, SKILL_MULTI_SEGMENT_TRACKING])
    if current_task_family in {FAMILY_CIRCLE, FAMILY_ELLIPSE, FAMILY_FIGURE_EIGHT} and _tracking_error_high(metrics):
        gaps.append(SKILL_CURVATURE_FOLLOWING)
    if diagnostic_signals.get("attitude_instability_count", 0) > 0 or failure_modes & {
        FAILURE_EARLY_TERMINATION,
        FAILURE_REPEATED_TRUNCATION,
        FAILURE_SAFETY_LIMIT_VIOLATION,
    }:
        gaps.append(SKILL_STABILITY_RECOVERY)
    return _dedupe(gaps)


def _pid_reference_unreachable(metrics: Mapping[str, Any]) -> bool:
    """Return whether diagnostics show a reference outside PID z target bounds."""
    reachable = metrics.get("reference_z_reachable_by_pid_position")
    if isinstance(reachable, bool):
        return not reachable
    return (
        _float(metrics.get("z_reference_above_pid_high_margin")) > TARGET_BOUNDARY_ACTION_TOLERANCE_M
        or _float(metrics.get("z_reference_below_pid_low_margin")) > TARGET_BOUNDARY_ACTION_TOLERANCE_M
    )


def _xy_tracking_weak(metrics: Mapping[str, Any]) -> bool:
    """Return whether XY tracking diagnostics suggest controlled XY practice."""
    xy_ratio = metrics.get("xy_tracking_ratio")
    if isinstance(xy_ratio, (int, float)) and float(xy_ratio) < INSUFFICIENT_XY_MOTION_RATIO:
        return True
    return _tracking_error_high(metrics) and _float(metrics.get("reference_xy_span_m")) > REFERENCE_XY_SPAN_MOVING_TASK_MIN_M


def _tracking_error_high(metrics: Mapping[str, Any]) -> bool:
    """Return whether tracking error is above the high-error threshold."""
    return _float(metrics.get("mean_position_error_tracking_m", metrics.get("mean_position_error_m"))) >= HIGH_MEAN_POSITION_ERROR_M


def _has_policy_instability(*, failure_modes: set[str], metrics: Mapping[str, Any], trend_status: str) -> bool:
    """Return whether diagnostics show real control instability rather than task difficulty."""
    attitude_or_limits = bool(
        failure_modes & {FAILURE_ATTITUDE_INSTABILITY, FAILURE_SAFETY_LIMIT_VIOLATION}
        or _int(metrics.get("strict_limit_violation_count")) >= STRICT_LIMIT_VIOLATION_COUNT_MIN
    )
    repeated_crash_or_divergence = bool(
        failure_modes & {FAILURE_EARLY_TERMINATION, FAILURE_REPEATED_TRUNCATION}
        or _int(metrics.get("eval_terminated_count")) > 0
        or _int(metrics.get("eval_truncated_count")) > 1
    )
    if attitude_or_limits or repeated_crash_or_divergence:
        return True
    if FAILURE_Z_INSTABILITY not in failure_modes:
        return False
    if _pid_reference_unreachable(metrics):
        return False
    return _z_instability_severe(metrics) or trend_status == TREND_WORSENING


def _z_instability_severe(metrics: Mapping[str, Any]) -> bool:
    """Return whether altitude diagnostics are severe enough to gate progression."""
    return (
        _float(metrics.get("actual_z_span_m")) >= Z_INSTABILITY_SPAN_M * 1.5
        or _float(metrics.get("mean_abs_z_error")) >= Z_INSTABILITY_MEAN_ABS_ERROR_M * 1.5
    )


def _interpreted_failure_modes(
    *,
    current_task_family: str,
    failure_modes: set[str],
    metrics: Mapping[str, Any],
    policy_instability: bool,
    trend_status: str,
) -> list[dict[str, Any]]:
    """Return curriculum-oriented interpretations for each failure mode."""
    modes = sorted(failure_modes or {"tracking_error_without_specific_failure"})
    return [
        _interpreted_failure_mode(
            mode=mode,
            current_task_family=current_task_family,
            metrics=metrics,
            policy_instability=policy_instability,
            trend_status=trend_status,
        )
        for mode in modes
    ]


def _interpreted_failure_mode(
    *,
    mode: str,
    current_task_family: str,
    metrics: Mapping[str, Any],
    policy_instability: bool,
    trend_status: str,
) -> dict[str, Any]:
    """Return one interpreted failure-mode record."""
    evidence = _feedback_evidence(metrics)
    if mode == FAILURE_Z_INSTABILITY:
        if _pid_reference_unreachable(metrics):
            return _failure_mode_record(
                name=mode,
                severity="configuration",
                interpretation=(
                    "Reference altitude exceeds pid_position z target reachability; review action-space bounds "
                    "before using another pure vertical recovery task."
                ),
                evidence=evidence,
                is_policy_instability=False,
                is_task_difficulty=False,
                is_training_signal=False,
            )
        is_instability = policy_instability and (_z_instability_severe(metrics) or trend_status == TREND_WORSENING)
        return _failure_mode_record(
            name=mode,
            severity="severe" if is_instability else "moderate",
            interpretation="Altitude control is weak; train controlled vertical or altitude-hold tasks instead of avoiding z motion by default.",
            evidence=evidence,
            is_policy_instability=is_instability,
            is_task_difficulty=not is_instability,
        )
    if mode == FAILURE_ACTION_SATURATION:
        if _pid_reference_unreachable(metrics):
            return _failure_mode_record(
                name=mode,
                severity="configuration",
                interpretation=(
                    "PID z-target saturation is expected when the reference is outside configured pid_position z bounds; "
                    "fix reachability before assigning policy blame."
                ),
                evidence=evidence,
                is_policy_instability=False,
                is_task_difficulty=False,
                is_training_signal=False,
            )
        tracking_ok = _tracking_acceptable(metrics)
        return _failure_mode_record(
            name=mode,
            severity="low" if tracking_ok else "moderate",
            interpretation=(
                "Action saturation is a diagnostic signal; use slower or easier related tasks when tracking is poor, "
                "but do not call it instability by itself."
            ),
            evidence=evidence,
            is_policy_instability=False,
            is_task_difficulty=not tracking_ok,
        )
    if mode == FAILURE_REFERENCE_TOO_HARD:
        return _failure_mode_record(
            name=mode,
            severity="moderate",
            interpretation=(
                f"The {current_task_family} reference appears too hard; choose a slower or smaller same-family version before abandoning the family."
            ),
            evidence=evidence,
            is_policy_instability=False,
            is_task_difficulty=True,
        )
    if mode in {FAILURE_HOVER_LOCK, FAILURE_INSUFFICIENT_XY_MOTION, FAILURE_OVERSHOOT}:
        return _failure_mode_record(
            name=mode,
            severity="moderate",
            interpretation="XY tracking needs a smaller controlled movement progression, such as start-hold line or short slow line.",
            evidence=evidence,
            is_policy_instability=False,
            is_task_difficulty=True,
        )
    if mode in {FAILURE_ATTITUDE_INSTABILITY, FAILURE_EARLY_TERMINATION, FAILURE_REPEATED_TRUNCATION, FAILURE_SAFETY_LIMIT_VIOLATION}:
        return _failure_mode_record(
            name=mode,
            severity="severe",
            interpretation=(
                "Control stability or safety limits failed; recover with stabilization and very simple slow references before adding path complexity."
            ),
            evidence=evidence,
            is_policy_instability=True,
            is_task_difficulty=False,
        )
    if mode == FAILURE_NONE:
        return _failure_mode_record(
            name=mode,
            severity="none",
            interpretation="Own-task tracking is acceptable; progress with a bounded next-family task while avoiding immediate duplicates.",
            evidence=evidence,
            is_policy_instability=False,
            is_task_difficulty=False,
        )
    return _failure_mode_record(
        name=mode,
        severity="moderate",
        interpretation="Tracking diagnostics suggest targeted skill practice before a large difficulty jump.",
        evidence=evidence,
        is_policy_instability=policy_instability,
        is_task_difficulty=not policy_instability,
    )


def _failure_mode_record(
    *,
    name: str,
    severity: str,
    interpretation: str,
    evidence: Mapping[str, Any],
    is_policy_instability: bool,
    is_task_difficulty: bool,
    is_training_signal: bool = True,
) -> dict[str, Any]:
    """Return one JSON-serializable interpreted failure-mode record."""
    return {
        "name": name,
        "severity": severity,
        "interpretation": interpretation,
        "evidence": dict(evidence),
        "is_policy_instability": bool(is_policy_instability),
        "is_task_difficulty": bool(is_task_difficulty),
        "is_training_signal": bool(is_training_signal),
    }


def _feedback_evidence(metrics: Mapping[str, Any]) -> dict[str, Any]:
    """Return compact evidence embedded in interpreted feedback records."""
    return {
        "mean_position_error_tracking_m": _float(metrics.get("mean_position_error_tracking_m", metrics.get("mean_position_error_m"))),
        "mean_abs_z_error": _float(metrics.get("mean_abs_z_error")),
        "actual_z_span_m": _float(metrics.get("actual_z_span_m")),
        "xy_tracking_ratio": metrics.get("xy_tracking_ratio"),
        "action_saturation_fraction": _float_list(metrics.get("action_saturation_fraction")),
        **_pid_z_metadata_fields(metrics),
        "eval_terminated_count": _int(metrics.get("eval_terminated_count")),
        "eval_truncated_count": _int(metrics.get("eval_truncated_count")),
        "strict_limit_violation_count": _int(metrics.get("strict_limit_violation_count")),
    }


def _readiness_level(
    *,
    failure_modes: set[str],
    metrics: Mapping[str, Any],
    policy_instability: bool,
    trend_status: str,
    primary_skill_gaps: Sequence[str],
) -> str:
    """Return an allowed, trend-aware readiness label."""
    if bool(metrics.get("policy_artifact_missing", False)):
        return READINESS_BLOCKED
    if policy_instability:
        return READINESS_UNSTABLE
    if trend_status == TREND_IMPROVING:
        return READINESS_IMPROVING
    if FAILURE_NONE in failure_modes and _tracking_acceptable(metrics):
        if str(metrics.get("generalization_status", "")).lower() == "successful" and str(metrics.get("scenario_status", "")).lower() in {
            "successful",
            "passed",
        }:
            return READINESS_STRONG
        return READINESS_READY
    if primary_skill_gaps:
        return READINESS_PARTIALLY_READY
    return READINESS_PARTIALLY_READY


def _own_task_status(readiness_level: str) -> str:
    """Return a compact own-task status for performance_summary."""
    mapping = {
        READINESS_BLOCKED: "blocked",
        READINESS_UNSTABLE: "control_instability",
        READINESS_IMPROVING: "improving_with_skill_gaps",
        READINESS_PARTIALLY_READY: "stable_with_skill_gaps",
        READINESS_READY: "own_task_ready",
        READINESS_STRONG: "own_task_strong",
    }
    return mapping.get(readiness_level, "unknown")


def _recommended_next_task_families(
    *,
    current_task_family: str,
    failure_modes: set[str],
    metrics: Mapping[str, Any],
    primary_skill_gaps: Sequence[str],
    policy_instability: bool,
) -> list[dict[str, Any]]:
    """Return constructive next-task family recommendations."""
    recommendations: list[dict[str, Any]] = []
    if _pid_reference_unreachable(metrics):
        recommendations.extend(
            [
                _recommendation(
                    FAMILY_START_HOLD_LINE,
                    "Use reachable short XY tracking while PID z bounds are reviewed.",
                    SKILL_XY_TRACKING,
                    "low",
                    1,
                ),
                _recommendation(
                    FAMILY_LINE,
                    "Continue bounded XY progression instead of repeating unreachable vertical references.",
                    SKILL_XY_TRACKING,
                    "low",
                    2,
                ),
            ]
        )
        return _dedupe_recommendations(recommendations)
    if policy_instability:
        recommendations.extend(
            [
                _recommendation(
                    FAMILY_HOVER, "Recover stable attitude and altitude before path complexity.", SKILL_STABILITY_RECOVERY, "recovery", 1
                ),
                _recommendation(FAMILY_TAKEOFF, "Practice slow vertical control after stabilization is bounded.", SKILL_ALTITUDE_CONTROL, "low", 2),
                _recommendation(FAMILY_START_HOLD_LINE, "Reintroduce tiny XY motion only after a start hold.", SKILL_XY_TRACKING, "low", 3),
            ]
        )
    elif SKILL_ALTITUDE_CONTROL in primary_skill_gaps:
        recommendations.extend(
            [
                _recommendation(
                    FAMILY_TAKEOFF,
                    "Train altitude control directly with a slow bounded vertical target.",
                    SKILL_ALTITUDE_CONTROL,
                    "low",
                    1,
                    bounds_hint="short z range and conservative duration",
                ),
                _recommendation(
                    FAMILY_HOVER,
                    "Consolidate altitude hold with mild z variation rather than freezing all vertical demand.",
                    SKILL_ALTITUDE_CONTROL,
                    "low",
                    2,
                    bounds_hint="small altitude band around the current target",
                ),
                _recommendation(
                    FAMILY_MULTI_HEIGHT_POLYLINE,
                    "Add multi-height path practice only at low speed if XY tracking is otherwise stable.",
                    SKILL_ALTITUDE_CONTROL,
                    "low",
                    3,
                    bounds_hint="few segments with small z changes",
                ),
            ]
        )
    if FAILURE_REFERENCE_TOO_HARD in failure_modes and current_task_family not in {"unknown", FAMILY_HOVER}:
        recommendations.append(
            _recommendation(
                current_task_family,
                "Use a slower, shorter, or lower-variation version of the same family before avoiding it.",
                SKILL_SPEED_CONTROL,
                "easier_same_family",
                1,
                variation_strength_hint="lower",
                bounds_hint="reduce speed, displacement, radius, or segment count",
            )
        )
    if SKILL_XY_TRACKING in primary_skill_gaps:
        recommendations.extend(
            [
                _recommendation(
                    FAMILY_START_HOLD_LINE,
                    "Start hold isolates lift and settling before a short XY move.",
                    SKILL_XY_TRACKING,
                    "low",
                    2,
                    bounds_hint="short displacement and slow move duration",
                ),
                _recommendation(
                    FAMILY_LINE,
                    "Practice direct XY tracking with a short slow line or diagonal line.",
                    SKILL_XY_TRACKING,
                    "low",
                    3,
                    bounds_hint="short segment with conservative speed",
                ),
            ]
        )
    if SKILL_TURN_FOLLOWING in primary_skill_gaps or SKILL_MULTI_SEGMENT_TRACKING in primary_skill_gaps:
        recommendations.extend(
            [
                _recommendation(FAMILY_L_SHAPE, "Train one deliberate turn at low speed before harder polylines.", SKILL_TURN_FOLLOWING, "low", 2),
                _recommendation(
                    FAMILY_POLYLINE, "Use a slow two- or three-segment polyline to target turn following.", SKILL_MULTI_SEGMENT_TRACKING, "low", 3
                ),
                _recommendation(FAMILY_ZIGZAG, "Use only if the policy is stable enough for repeated gentle turns.", SKILL_TURN_FOLLOWING, "low", 4),
            ]
        )
    if SKILL_CURVATURE_FOLLOWING in primary_skill_gaps:
        recommendations.extend(
            [
                _recommendation(
                    FAMILY_ELLIPSE, "Train gentle continuous curvature before figure-eight crossings.", SKILL_CURVATURE_FOLLOWING, "low", 2
                ),
                _recommendation(
                    FAMILY_CIRCLE, "Use a slow small-radius circle to consolidate curvature tracking.", SKILL_CURVATURE_FOLLOWING, "low", 3
                ),
            ]
        )
    if not recommendations:
        recommendations.extend(_default_progression_recommendations(current_task_family, metrics))
    return _dedupe_recommendations(recommendations)


def _default_progression_recommendations(current_task_family: str, metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return conservative progression recommendations when no major gap is detected."""
    if current_task_family == FAMILY_HOVER:
        return [
            _recommendation(FAMILY_START_HOLD_LINE, "Progress from hover to a bounded short XY move.", SKILL_XY_TRACKING, "low", 1),
            _recommendation(FAMILY_LINE, "Use a short slow line as the next tracking primitive.", SKILL_XY_TRACKING, "low", 2),
        ]
    if current_task_family in {FAMILY_LINE, FAMILY_START_HOLD_LINE}:
        return [
            _recommendation(FAMILY_L_SHAPE, "Add one low-speed turn after line tracking is stable.", SKILL_TURN_FOLLOWING, "low", 1),
            _recommendation(FAMILY_POLYLINE, "Introduce a gentle bounded multi-segment line.", SKILL_MULTI_SEGMENT_TRACKING, "low", 2),
            _recommendation(FAMILY_ELLIPSE, "Add gentle curvature only if recent tracking remains stable.", SKILL_CURVATURE_FOLLOWING, "low", 3),
        ]
    if _tracking_acceptable(metrics):
        return [
            _recommendation(
                current_task_family, "Repeat the family non-consecutively with slightly broader bounded variation.", SKILL_SPEED_CONTROL, "medium", 2
            ),
            _recommendation(FAMILY_ELLIPSE, "Use gentle curvature as complementary tracking practice.", SKILL_CURVATURE_FOLLOWING, "low", 3),
        ]
    return [_recommendation(FAMILY_LINE, "Consolidate stable short XY tracking before increasing complexity.", SKILL_XY_TRACKING, "low", 1)]


def _recommendation(
    task_family: str,
    reason: str,
    targeted_skill: str,
    difficulty_hint: str,
    priority: int,
    *,
    variation_strength_hint: str = "low",
    bounds_hint: str | None = None,
) -> dict[str, Any]:
    """Return one structured task-family recommendation."""
    record: dict[str, Any] = {
        "task_family": task_family,
        "reason": reason,
        "targeted_skill": targeted_skill,
        "difficulty_hint": difficulty_hint,
        "variation_strength_hint": variation_strength_hint,
        "priority": int(priority),
    }
    if bounds_hint is not None:
        record["bounds_hint"] = bounds_hint
    return record


def _dedupe_recommendations(recommendations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate recommendation records by task family while preserving priority."""
    unique: dict[str, dict[str, Any]] = {}
    for item in recommendations:
        family = str(item.get("task_family", ""))
        if not family:
            continue
        candidate = dict(item)
        existing = unique.get(family)
        if existing is None or int(candidate.get("priority", 999)) < int(existing.get("priority", 999)):
            unique[family] = candidate
    return sorted(unique.values(), key=lambda item: (int(item.get("priority", 999)), str(item.get("task_family", ""))))


def _avoid_next_task_families(
    *,
    current_task_family: str,
    failure_modes: set[str],
    primary_skill_gaps: Sequence[str],
    policy_instability: bool,
    trend_status: str,
    metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return temporary avoid guidance without banning useful families forever."""
    avoids: list[dict[str, Any]] = []
    if policy_instability:
        avoids.extend(
            [
                _avoidance(
                    FAMILY_FIGURE_EIGHT, "Fast crossing curves compound recovery instability.", "attitude, z, and termination diagnostics are stable"
                ),
                _avoidance(
                    FAMILY_CIRCLE, "Continuous curvature should wait until stabilization recovers.", "simple hover/takeoff and line tasks are stable"
                ),
                _avoidance(
                    FAMILY_POLYLINE, "Multi-segment turns should wait until recovery is stable.", "short line tracking succeeds without safety events"
                ),
            ]
        )
        if _z_instability_severe(metrics) or trend_status == TREND_WORSENING:
            avoids.append(
                _avoidance(
                    FAMILY_MULTI_HEIGHT_POLYLINE,
                    "Severe or worsening altitude instability makes multi-height paths too broad right now.",
                    "controlled takeoff or altitude-hold tasks reduce z error",
                )
            )
    if SKILL_CURVATURE_FOLLOWING in primary_skill_gaps:
        avoids.append(
            _avoidance(
                FAMILY_FIGURE_EIGHT,
                "Figure-eight combines curvature reversal and crossing before gentle curves are ready.",
                "ellipse or slow circle tracks acceptably",
            )
        )
    if FAILURE_REFERENCE_TOO_HARD in failure_modes and current_task_family not in {"unknown", FAMILY_HOVER}:
        avoids.append(
            _avoidance(
                current_task_family,
                "Avoid only high-speed or high-variation versions of this family, not the family itself.",
                "a slower or smaller same-family variant tracks acceptably",
            )
        )
    return _dedupe_avoidances(avoids)


def _avoidance(task_family: str, reason: str, condition_to_reintroduce: str) -> dict[str, Any]:
    """Return one structured temporary avoid record."""
    return {
        "task_family": task_family,
        "reason": reason,
        "temporary": True,
        "condition_to_reintroduce": condition_to_reintroduce,
    }


def _dedupe_avoidances(avoidances: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate avoid records by family while preserving first reason."""
    unique: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in avoidances:
        family = str(item.get("task_family", ""))
        if not family or family in seen:
            continue
        unique.append(dict(item))
        seen.add(family)
    return unique


def _constraints_for_next_curriculum(
    *,
    failure_modes: set[str],
    primary_skill_gaps: Sequence[str],
    policy_instability: bool,
    metrics: Mapping[str, Any],
) -> list[dict[str, str]]:
    """Return structured next-curriculum constraints with reasons."""
    constraints: list[dict[str, str]] = []
    if _pid_reference_unreachable(metrics):
        constraints.append(
            _constraint(
                "review pid_position z target bounds before repeating vertical tasks",
                "the reference z range is outside the configured PID target-position reachability",
            )
        )
    if policy_instability:
        constraints.append(
            _constraint("start with stabilization or very slow references", "control instability requires recovery before path complexity")
        )
    if SKILL_ALTITUDE_CONTROL in primary_skill_gaps:
        constraints.append(
            _constraint(
                "include controlled altitude practice",
                "z diagnostics should become an explicit training target rather than a reason to avoid all vertical motion",
            )
        )
    if SKILL_XY_TRACKING in primary_skill_gaps:
        constraints.append(
            _constraint("use short displacement and slower reference velocity", "XY tracking is weak but stable enough to train directly")
        )
    if SKILL_TURN_FOLLOWING in primary_skill_gaps or SKILL_MULTI_SEGMENT_TRACKING in primary_skill_gaps:
        constraints.append(_constraint("limit segment count and turn speed", "turn following should be trained with one or two gentle turns first"))
    if SKILL_CURVATURE_FOLLOWING in primary_skill_gaps:
        constraints.append(
            _constraint("prefer gentle ellipse or slow circle before figure-eight", "curvature should be introduced without crossing complexity")
        )
    if FAILURE_REFERENCE_TOO_HARD in failure_modes:
        constraints.append(
            _constraint("make the related family slower or smaller", "reference-too-hard indicates task difficulty, not automatic policy instability")
        )
    if _diagnostic_signal_counts(failure_modes, metrics).get("action_saturation_count", 0) > 0:
        constraints.append(
            _constraint("monitor action saturation without treating it as a standalone recovery trigger", "saturation alone is diagnostic context")
        )
    if not constraints:
        constraints.append(
            _constraint("avoid the immediately previous accepted family", "prevents curriculum loops while preserving bounded progression")
        )
    return _dedupe_constraints(constraints)


def _constraint(constraint: str, reason: str) -> dict[str, str]:
    """Return one structured constraint record."""
    return {"constraint": constraint, "reason": reason}


def _dedupe_constraints(constraints: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    """Deduplicate constraint records by text."""
    unique: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in constraints:
        text = str(item.get("constraint", ""))
        if not text or text in seen:
            continue
        unique.append({"constraint": text, "reason": str(item.get("reason", ""))})
        seen.add(text)
    return unique


def _curriculum_strategy(
    *,
    readiness_level: str,
    current_task_family: str,
    failure_modes: set[str],
    primary_skill_gaps: Sequence[str],
    policy_instability: bool,
    trend_status: str,
    difficulty_level: str,
) -> dict[str, Any]:
    """Return the high-level curriculum strategy for the next stage."""
    if policy_instability:
        progression_type = "recover"
    elif trend_status == TREND_IMPROVING and primary_skill_gaps:
        progression_type = "targeted_skill_training"
    elif FAILURE_REFERENCE_TOO_HARD in failure_modes or difficulty_level == "high":
        progression_type = "consolidate"
    elif primary_skill_gaps:
        progression_type = "controlled_progression"
    elif readiness_level in {READINESS_READY, READINESS_STRONG}:
        progression_type = "advance_difficulty"
    else:
        progression_type = "complementary_training"
    rationale = _strategy_rationale(
        progression_type=progression_type,
        current_task_family=current_task_family,
        failure_modes=failure_modes,
        primary_skill_gaps=primary_skill_gaps,
        trend_status=trend_status,
    )
    return {
        "progression_type": progression_type,
        "rationale": rationale,
        "should_progress": progression_type in {"controlled_progression", "complementary_training", "advance_difficulty"},
        "should_recover": progression_type == "recover",
        "should_repeat_family_non_consecutively": progression_type in {"consolidate", "targeted_skill_training"},
        "avoid_immediate_duplicate_family": True,
    }


def _strategy_rationale(
    *,
    progression_type: str,
    current_task_family: str,
    failure_modes: set[str],
    primary_skill_gaps: Sequence[str],
    trend_status: str,
) -> str:
    """Return a concise rationale for curriculum strategy."""
    modes = ", ".join(sorted(failure_modes)) if failure_modes else "tracking_error_without_specific_failure"
    gaps = ", ".join(primary_skill_gaps) if primary_skill_gaps else "no major skill gap"
    return (
        f"Use {progression_type} after {current_task_family}: diagnostics={modes}, gaps={gaps}, trend={trend_status}. "
        "Treat diagnostics as curriculum signals and prefer targeted skill training over broad avoidance."
    )


def _llm_instruction_summary(
    *,
    readiness_level: str,
    current_task_family: str,
    primary_skill_gaps: Sequence[str],
    recommended_next_task_families: Sequence[Mapping[str, Any]],
    strategy: Mapping[str, Any],
) -> str:
    """Return a short prompt-ready curriculum feedback summary."""
    recommended = ", ".join(str(item.get("task_family")) for item in recommended_next_task_families[:3]) or "bounded easy progression"
    gaps = ", ".join(primary_skill_gaps) if primary_skill_gaps else "none"
    return (
        f"Readiness {readiness_level}; current family {current_task_family}; primary gaps: {gaps}. "
        f"Next strategy: {strategy.get('progression_type')}; prefer {recommended}. "
        "Use this as guidance, not an absolute command."
    )


def _optional_int(value: Any) -> int | None:
    """Return an int for present values, otherwise ``None``."""
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _array_field(records: Sequence[Mapping[str, Any]], field: str, min_width: int = POSITION_DIMENSIONS) -> np.ndarray:
    """Return a finite two-dimensional array for a trace field."""
    rows = [np.asarray(record.get(field, []), dtype=float).reshape(-1) for record in records]
    width = max([row.size for row in rows] + [min_width])
    array = np.zeros((len(rows), width), dtype=float)
    for index, row in enumerate(rows):
        if row.size:
            array[index, : row.size] = row
    return array


def _array3(value: Any) -> np.ndarray:
    """Return a shape-(3,) float array for trace position fields."""
    array = np.asarray(value, dtype=float).reshape(-1)
    if array.size < POSITION_DIMENSIONS:
        padded = np.zeros(POSITION_DIMENSIONS, dtype=float)
        padded[: array.size] = array
        return padded
    return array[:POSITION_DIMENSIONS]


def _position_bounds(values: np.ndarray) -> dict[str, list[float]]:
    """Return min/max bounds for rows of numeric values."""
    if values.size == 0:
        return {"min": [], "max": []}
    array = np.asarray(values, dtype=float)
    if array.ndim > ARRAY_BOUNDS_MAX_NDIM:
        array = array.reshape(array.shape[0], -1)
    return {
        "min": [float(value) for value in np.min(array, axis=0)],
        "max": [float(value) for value in np.max(array, axis=0)],
    }


def _xy_span(bounds: Mapping[str, Sequence[float]]) -> float:
    """Return Euclidean XY span from position bounds."""
    return float(np.linalg.norm([_axis_span(bounds, axis=0), _axis_span(bounds, axis=1)]))


def _axis_span(bounds: Mapping[str, Sequence[float]], axis: int) -> float:
    """Return one axis span from position bounds."""
    return float(_axis_max(bounds, axis=axis) - _axis_min(bounds, axis=axis))


def _axis_min(bounds: Mapping[str, Sequence[float]], axis: int) -> float:
    """Return one min-bound axis value or 0.0 when unavailable."""
    values = list(bounds.get("min", []))
    if len(values) <= axis:
        return 0.0
    return float(values[axis])


def _axis_max(bounds: Mapping[str, Sequence[float]], axis: int) -> float:
    """Return one max-bound axis value or 0.0 when unavailable."""
    values = list(bounds.get("max", []))
    if len(values) <= axis:
        return 0.0
    return float(values[axis])


def _xy_tracking_ratio(actual_xy_span_m: float, reference_xy_span_m: float) -> float | None:
    """Return actual/reference XY span ratio when the reference span is nonzero."""
    if reference_xy_span_m <= XY_TRACKING_RATIO_MIN_REFERENCE_SPAN_M:
        return None
    return float(actual_xy_span_m / reference_xy_span_m)


def _max_row_norm(values: np.ndarray) -> float:
    """Return the maximum Euclidean row norm for a numeric 2D array."""
    if values.size == 0:
        return 0.0
    rows = np.asarray(values, dtype=float).reshape(values.shape[0], -1)
    if rows.shape[1] == 0:
        return 0.0
    return float(np.max(np.linalg.norm(rows, axis=1)))


def _strict_limit_violation_count(records: Sequence[Mapping[str, Any]]) -> int:
    """Return the number of trace rows with strict diagnostic violations."""
    return int(sum(1 for record in records if bool(record.get("strict_limit_violation"))))


def _unique_list_field(records: Sequence[Mapping[str, Any]], field: str) -> list[str]:
    """Return unique string values found in list-like trace fields."""
    values: list[str] = []
    for record in records:
        raw_values = record.get(field, [])
        candidates = [raw_values] if isinstance(raw_values, str) else list(raw_values) if isinstance(raw_values, (list, tuple)) else []
        for value in candidates:
            text = str(value)
            if text and text not in values:
                values.append(text)
    return values


def _max_episode_value(episode_summaries: Sequence[Mapping[str, Any]], key: str) -> float:
    """Return the maximum numeric value for an episode summary key."""
    return max((_float(summary.get(key)) for summary in episode_summaries), default=0.0)


def _float(value: Any) -> float:
    """Return a float value, using 0.0 for missing values."""
    if value is None:
        return 0.0
    return float(value)


def _int(value: Any) -> int:
    """Return an integer value, using 0 for missing values."""
    if value is None:
        return 0
    return int(value)


def _float_list(value: Any) -> list[float]:
    """Return a flat list of floats from a scalar or sequence value."""
    if value is None:
        return []
    array = np.asarray(value, dtype=float).reshape(-1)
    return [float(item) for item in array]


def _array_to_jsonable(value: Any) -> list[Any]:
    """Convert an array-like value to nested JSON-compatible lists."""
    return np.asarray(value).tolist()


def _json_ready(value: Any) -> Any:
    """Return a JSON-compatible copy of a nested value."""
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def _dedupe(values: Sequence[str]) -> list[str]:
    """Deduplicate strings while preserving first occurrence."""
    unique: list[str] = []
    seen: set[str] = set()
    for value in values:
        if value in seen:
            continue
        unique.append(value)
        seen.add(value)
    return unique


def _write_json(path: Path, payload: Any) -> None:
    """Write a JSON artifact with stable strict formatting."""
    safe_payload = utils.serialization.to_jsonable(payload)
    utils.serialization.assert_json_serializable(safe_payload, str(path))
    path.write_text(json.dumps(safe_payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


__all__ = [
    "CURRICULUM_FEEDBACK_FILENAME",
    "EPISODE_SUMMARIES_FILENAME",
    "EVALUATION_TRACE_FILENAME",
    "FAILURE_REPORT_FILENAME",
    "PolicyEvaluationDiagnostics",
    "build_curriculum_feedback",
    "build_failure_report",
    "collect_policy_evaluation_diagnostics",
    "summarize_policy_evaluation_trace",
    "write_policy_evaluation_diagnostics",
]
