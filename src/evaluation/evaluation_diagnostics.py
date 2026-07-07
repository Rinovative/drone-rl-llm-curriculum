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

from src import evaluation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

EVALUATION_TRACE_FILENAME = "evaluation_trace.jsonl"
EPISODE_SUMMARIES_FILENAME = "episode_summaries.json"
FAILURE_REPORT_FILENAME = "failure_report.json"
CURRICULUM_FEEDBACK_FILENAME = "curriculum_feedback.json"

POSITION_DIMENSIONS = 3
ROLL_PITCH_DIMENSIONS = 2
ARRAY_BOUNDS_MAX_NDIM = 2
ACTION_SATURATION_TOLERANCE = 1.0e-6
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

FAILURE_HOVER_LOCK = "hover_lock"
FAILURE_INSUFFICIENT_XY_MOTION = "insufficient_xy_motion"
FAILURE_ACTION_SATURATION = "action_saturation"
FAILURE_OVERSHOOT = "overshoot"
FAILURE_Z_INSTABILITY = "z_instability"
FAILURE_ATTITUDE_INSTABILITY = "attitude_instability"
FAILURE_EARLY_TERMINATION = "early_termination"
FAILURE_REPEATED_TRUNCATION = "repeated_truncation"
FAILURE_REFERENCE_TOO_HARD = "reference_too_fast_or_too_hard"
FAILURE_NONE = "no_failure_detected"


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
    return {
        "diagnostics_dir": str(resolved_dir),
        "evaluation_trace_path": str(trace_path),
        "episode_summaries_path": str(episode_summaries_path),
        "failure_report_path": str(failure_report_path),
        "curriculum_feedback_path": str(curriculum_feedback_path),
        "failure_primary_mode": diagnostics.failure_report["primary_failure_mode"],
        "failure_modes": failure_modes,
        "failure_overall_status": diagnostics.failure_report["overall_status"],
        "curriculum_readiness_level": diagnostics.curriculum_feedback["readiness_level"],
        "curriculum_recommended_next_tasks": list(diagnostics.curriculum_feedback["recommended_next_tasks"]),
        "curriculum_avoid_next_tasks": list(diagnostics.curriculum_feedback["avoid_next_tasks"]),
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
    high_error = mean_position_error_m >= HIGH_MEAN_POSITION_ERROR_M

    if moving_reference and actual_xy_span_m < HOVER_LOCK_ACTUAL_XY_SPAN_MAX_M:
        failure_modes.append(FAILURE_HOVER_LOCK)
    if moving_reference and actual_xy_span_m < INSUFFICIENT_XY_MOTION_RATIO * reference_xy_span_m:
        failure_modes.append(FAILURE_INSUFFICIENT_XY_MOTION)
    if max_action_saturation >= ACTION_SATURATION_FRACTION_MIN:
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
    if high_error and (
        FAILURE_ACTION_SATURATION in failure_modes
        or FAILURE_Z_INSTABILITY in failure_modes
        or FAILURE_ATTITUDE_INSTABILITY in failure_modes
        or FAILURE_REPEATED_TRUNCATION in failure_modes
    ):
        failure_modes.append(FAILURE_REFERENCE_TOO_HARD)

    failure_modes = _dedupe(failure_modes)
    tracking_acceptable = _tracking_acceptable(metrics)
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
    failure_modes = {str(mode) for mode in failure_report.get("failure_modes", [])}
    recommended_next_tasks: list[str]
    avoid_next_tasks: list[str]
    constraints: list[str] = []

    if failure_modes & {FAILURE_Z_INSTABILITY, FAILURE_ATTITUDE_INSTABILITY, FAILURE_EARLY_TERMINATION, FAILURE_REPEATED_TRUNCATION}:
        readiness_level = "unstable"
        recommended_next_tasks = ["hover_stabilization", "takeoff_stabilization", "start_hold_then_line"]
        avoid_next_tasks = ["long_line", "fast_polyline", "circle", "figure_eight"]
        constraints.extend(["keep altitude target simple", "add a start hold before XY motion"])
    elif failure_modes & {FAILURE_HOVER_LOCK, FAILURE_INSUFFICIENT_XY_MOTION, FAILURE_ACTION_SATURATION}:
        readiness_level = "line_not_ready"
        recommended_next_tasks = ["nearby_target_hover", "short_slow_line", "start_hold_then_line"]
        avoid_next_tasks = ["long_line", "fast_polyline", "circle"]
        constraints.extend(["shorter displacement", "slower reference velocity"])
    elif FAILURE_NONE in failure_modes and str(current_task_shape).lower() == "hover":
        readiness_level = "near_target_ready"
        recommended_next_tasks = ["nearby_target_hover", "short_slow_line"]
        avoid_next_tasks = ["fast_polyline", "circle"]
        constraints.append("increase XY demand gradually")
    elif FAILURE_NONE in failure_modes:
        readiness_level = "slow_line_ready"
        recommended_next_tasks = ["longer_slow_line", "gentle_polyline", "slow_circle"]
        avoid_next_tasks = ["fast_polyline"]
        constraints.append("preserve current altitude and speed limits")
    else:
        readiness_level = "hover_ready"
        recommended_next_tasks = ["nearby_target_hover"]
        avoid_next_tasks = ["long_line", "fast_polyline", "circle"]
        constraints.append("require acceptable tracking before increasing path complexity")

    if FAILURE_ACTION_SATURATION in failure_modes:
        constraints.extend(["possible action smoothness penalty candidate", "avoid actions near bounds for more than half the rollout"])
    if _float(metrics.get("mean_abs_z_error")) >= Z_INSTABILITY_MEAN_ABS_ERROR_M:
        constraints.append("defer XY tracking until altitude error is lower")

    return {
        "current_task_shape": current_task_shape,
        "readiness_level": readiness_level,
        "recommended_next_tasks": _dedupe(recommended_next_tasks),
        "avoid_next_tasks": _dedupe(avoid_next_tasks),
        "constraints_for_next_curriculum": _dedupe(constraints),
        "rationale": _curriculum_rationale(readiness_level, failure_modes, metrics),
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
    record = {
        "step_index": int(step_index),
        "episode_index": int(episode_index),
        "episode_step_index": int(episode_step_index),
        "time_sec": float(step_index),
        "current_position": current_position,
        "reference_position": reference_position,
        "actual_position_xyz_m": current_position,
        "reference_position_xyz_m": reference_position,
        "position_error_m": float(info.get("position_error_m", np.linalg.norm(axis_error))),
        "axis_error_xyz": axis_error,
        "error_xyz_m": axis_error,
        "velocity": _array_to_jsonable(info.get("velocity", [])),
        "angular_velocity": _array_to_jsonable(info.get("angular_velocity", [])),
        "roll_pitch_yaw": _array_to_jsonable(info.get("roll_pitch_yaw", [])),
        "action": _array_to_jsonable(info.get("normalized_action", action)),
        "normalized_action": _array_to_jsonable(info.get("normalized_action", action)),
        "real_action": _array_to_jsonable(info.get("real_action", info.get("applied_action", action))),
        "actions_normalized": bool(info.get("action_normalized", False)),
        "real_action_space_low": _array_to_jsonable(info.get("real_action_space_low", [])),
        "real_action_space_high": _array_to_jsonable(info.get("real_action_space_high", [])),
        "last_action": _array_to_jsonable(info.get("last_action", [])),
        "reward": float(reward),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "termination_reason": str(info.get("termination_reason", "running")),
        "base_terminated": bool(info.get("base_terminated", False)),
        "base_truncated": bool(info.get("base_truncated", False)),
        "base_truncation_causes": list(info.get("base_truncation_causes", [])),
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
    axis_errors = np.abs(positions - references)
    action_metrics = _action_distribution_metrics(_array_field(records, "action"), action_space)
    real_action_metrics = _prefixed_action_distribution_metrics(
        _array_field(records, "real_action"),
        _real_action_space_from_records(records, fallback_action_space=action_space),
        prefix="real_action",
    )
    position_bounds = _position_bounds(positions)
    reference_bounds = _position_bounds(references)
    actual_xy_span_m = _xy_span(position_bounds)
    reference_xy_span_m = _xy_span(reference_bounds)
    roll_pitch = _array_field(records, "roll_pitch_yaw", min_width=ROLL_PITCH_DIMENSIONS)[:, :ROLL_PITCH_DIMENSIONS]
    return {
        "episode_index": _int(records[0].get("episode_index")),
        "start_step_index": _int(records[0].get("step_index")),
        "end_step_index": _int(records[-1].get("step_index")),
        "steps": len(records),
        "total_reward": float(np.sum(rewards)),
        "mean_reward": float(np.mean(rewards)),
        "final_reward": float(rewards[-1]),
        "mean_position_error_m": float(np.mean(errors)),
        "final_position_error_m": float(errors[-1]),
        "max_position_error_m": float(np.max(errors)),
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
        "z_min": _axis_min(position_bounds, axis=2),
        "z_max": _axis_max(position_bounds, axis=2),
        "max_abs_roll_pitch_rad": float(np.max(np.abs(roll_pitch))) if roll_pitch.size else 0.0,
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
    axis_errors = np.abs(positions - references)
    position_bounds = _position_bounds(positions)
    reference_bounds = _position_bounds(references)
    actual_xy_span_m = _xy_span(position_bounds)
    reference_xy_span_m = _xy_span(reference_bounds)
    return {
        "mean_eval_reward": float(np.mean(rewards)),
        "final_eval_reward": float(rewards[-1]),
        "mean_position_error_m": float(np.mean(errors)),
        "final_position_error_m": float(errors[-1]),
        "max_position_error_m": float(np.max(errors)),
        "position_bounds": position_bounds,
        "reference_position_bounds": reference_bounds,
        "action_bounds": _position_bounds(_array_field(records, "action")),
        "real_action_bounds": _position_bounds(_array_field(records, "real_action")),
        "actions_normalized": any(bool(record.get("actions_normalized", False)) for record in records),
        "actual_z_span_m": _axis_span(position_bounds, axis=2),
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
        **_action_distribution_metrics(_array_field(records, "action"), action_space),
        **_prefixed_action_distribution_metrics(
            _array_field(records, "real_action"),
            _real_action_space_from_records(records, fallback_action_space=action_space),
            prefix="real_action",
        ),
    }


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
            "action_saturation_fraction": [],
        }
    action_array = np.asarray(actions, dtype=float).reshape(actions.shape[0], -1)
    low = np.asarray(getattr(action_space, "low", []), dtype=float).reshape(-1)
    high = np.asarray(getattr(action_space, "high", []), dtype=float).reshape(-1)
    if low.size != action_array.shape[1] or high.size != action_array.shape[1]:
        saturation_fraction = np.zeros(action_array.shape[1], dtype=float)
    else:
        near_low = np.isclose(action_array, low, atol=ACTION_SATURATION_TOLERANCE, rtol=0.0)
        near_high = np.isclose(action_array, high, atol=ACTION_SATURATION_TOLERANCE, rtol=0.0)
        saturation_fraction = np.mean(np.logical_or(near_low, near_high), axis=0)
    return {
        "action_mean": [float(value) for value in np.mean(action_array, axis=0)],
        "action_std": [float(value) for value in np.std(action_array, axis=0)],
        "action_min": [float(value) for value in np.min(action_array, axis=0)],
        "action_max": [float(value) for value in np.max(action_array, axis=0)],
        "action_saturation_fraction": [float(value) for value in saturation_fraction],
    }


def _tracking_acceptable(metrics: Mapping[str, Any]) -> bool:
    """Return whether tracking metrics look acceptable when no major failure is detected."""
    return (
        _float(metrics.get("mean_position_error_m")) <= ACCEPTABLE_MEAN_POSITION_ERROR_M
        and _float(metrics.get("final_position_error_m")) <= ACCEPTABLE_FINAL_POSITION_ERROR_M
    )


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
        "final_position_error_m": _float(metrics.get("final_position_error_m")),
        "max_position_error_m": _float(metrics.get("max_position_error_m")),
        "reference_xy_span_m": _float(metrics.get("reference_xy_span_m")),
        "actual_xy_span_m": _float(metrics.get("actual_xy_span_m")),
        "xy_tracking_ratio": metrics.get("xy_tracking_ratio"),
        "action_saturation_fraction": _float_list(metrics.get("action_saturation_fraction")),
        "actions_normalized": bool(metrics.get("actions_normalized", False)),
        "real_action_saturation_fraction": _float_list(metrics.get("real_action_saturation_fraction")),
        "eval_terminated_count": _int(metrics.get("eval_terminated_count")),
        "eval_truncated_count": _int(metrics.get("eval_truncated_count")),
        "actual_z_span_m": _float(metrics.get("actual_z_span_m")),
        "mean_abs_z_error": _float(metrics.get("mean_abs_z_error")),
        "max_abs_roll_pitch_rad": _max_episode_value(episode_summaries, "max_abs_roll_pitch_rad"),
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
        f"Readiness is {readiness_level} because evaluation modes were {modes}; mean error was {_float(metrics.get('mean_position_error_m')):.3f} m."
    )


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
    """Write a JSON artifact with stable formatting."""
    path.write_text(json.dumps(_json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


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
