"""
===============================================================================
evaluation_rollout.py
===============================================================================
Run deterministic MVP rollout evaluation for validated trajectory tasks.

Responsibilities:
  - Convert task mappings or task references into comparable trajectory samples
  - Generate deterministic baseline actual trajectories for smoke evaluation
  - Write small JSON rollout metric artifacts under approved results paths
  - Serialize trained-policy step traces for notebook and report review

Design principles:
  - Reuse trajectory metric helpers for all tracking-error calculations
  - Keep rollout evaluation independent of PyBullet, SB3, Torch, and plotting

Boundaries:
  - Training-smoke loops belong in experiments modules
  - Visualization and figure writing belong in evaluation_plots.py
===============================================================================

"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src import envs, evaluation, trajectories, utils

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

XYZ_DIMENSIONS = 3
DEFAULT_OFFSET = (0.05, 0.0, 0.0)
DEFAULT_OUTPUT_FILENAME = "rollout_metrics.json"
TRACE_REQUIRED_FIELDS = (
    "step_index",
    "time_sec",
    "reward",
    "position_error_m",
    "actual_position_xyz_m",
    "reference_position_xyz_m",
    "error_xyz_m",
    "velocity",
    "roll_pitch_yaw",
    "angular_velocity",
    "action",
    "terminated",
    "truncated",
    "termination_reason",
)


@dataclass(frozen=True)
class RolloutEvaluationResult:
    """
    Result from deterministic rollout evaluation.

    Parameters
    ----------
    reference
        Reference trajectory sampled from a validated task.
    actual
        Deterministic baseline actual trajectory sampled at matching times.
    metrics
        JSON-serializable tracking and reward metrics.

    """

    reference: trajectories.primitives.Trajectory
    actual: trajectories.primitives.Trajectory
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RolloutWriteResult:
    """
    Summary returned after writing rollout metrics.

    Parameters
    ----------
    output_path
        Path to the written JSON artifact.
    metrics
        JSON-serializable rollout metrics.

    """

    output_path: str
    metrics: dict[str, Any]


@dataclass(frozen=True)
class RolloutTraceWriteResult:
    """
    Summary returned after writing a trained-policy rollout trace.

    Parameters
    ----------
    output_path
        Path to the written JSONL trace artifact.
    step_count
        Number of rollout step records written.
    columns
        Sorted union of trace record keys.

    """

    output_path: str
    step_count: int
    columns: tuple[str, ...]


def evaluate_task_rollout(
    task_or_reference: Mapping[str, Any] | envs.task_adapter.EnvironmentTaskReference,
    offset: object = DEFAULT_OFFSET,
    lag_steps: int = 0,
) -> RolloutEvaluationResult:
    """
    Evaluate a deterministic baseline rollout for a trajectory task.

    Parameters
    ----------
    task_or_reference
        Task mapping to validate or an already validated environment task reference.
    offset
        Deterministic XYZ offset added to the baseline actual trajectory.
    lag_steps
        Optional nonnegative integer lag applied to reference positions before adding ``offset``.

    Returns
    -------
    RolloutEvaluationResult
        Reference trajectory, actual trajectory, and JSON-serializable metrics.

    """
    if lag_steps < 0:
        message = "lag_steps must be nonnegative"
        raise ValueError(message)
    reference_data = _as_task_reference(task_or_reference)
    offset_array = _as_xyz_offset(offset)
    reference = trajectories.primitives.Trajectory(
        times=np.array(reference_data.times, dtype=float, copy=True),
        positions=np.array(reference_data.positions, dtype=float, copy=True),
    )
    actual_positions = _make_actual_positions(reference.positions, offset_array=offset_array, lag_steps=lag_steps)
    actual = trajectories.primitives.Trajectory(
        times=np.array(reference.times, dtype=float, copy=True),
        positions=actual_positions,
    )
    summary = evaluation.trajectory_metrics.summarize_tracking_error(reference=reference, actual=actual)
    metrics = _metrics_to_dict(summary, reference_data.shape)
    metrics.update(_start_hold_metrics(reference_data=reference_data, reference=reference, actual=actual))
    metrics["offset_xyz_m"] = [float(value) for value in offset_array]
    metrics["lag_steps"] = lag_steps
    metrics.update(_reward_metrics(reference_data=reference_data, actual_positions=actual_positions))
    return RolloutEvaluationResult(reference=reference, actual=actual, metrics=metrics)


def write_task_rollout_evaluation(
    task_or_reference: Mapping[str, Any] | envs.task_adapter.EnvironmentTaskReference,
    output_path: str | Path | None = None,
    offset: object = DEFAULT_OFFSET,
    lag_steps: int = 0,
) -> RolloutWriteResult:
    """
    Evaluate a deterministic rollout and write metrics JSON.

    Parameters
    ----------
    task_or_reference
        Task mapping to validate or an already validated environment task reference.
    output_path
        Optional JSON output path. Defaults to ``storage/results/mvp_smoke/rollout_metrics.json``.
    offset
        Deterministic XYZ offset added to actual positions.
    lag_steps
        Optional nonnegative reference lag.

    Returns
    -------
    RolloutWriteResult
        Written output path and metrics.

    """
    result = evaluate_task_rollout(task_or_reference=task_or_reference, offset=offset, lag_steps=lag_steps)
    resolved_path = Path(output_path) if output_path is not None else utils.paths.get_results_root() / "mvp_smoke" / DEFAULT_OUTPUT_FILENAME
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(result.metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return RolloutWriteResult(output_path=str(resolved_path), metrics=result.metrics)


def write_policy_rollout_trace(
    trace_records: Sequence[Mapping[str, Any]],
    output_path: str | Path,
) -> RolloutTraceWriteResult:
    """
    Write a trained-policy rollout trace as newline-delimited JSON.

    Parameters
    ----------
    trace_records
        Per-step rollout dictionaries. Each record must include the required
        review fields listed in ``TRACE_REQUIRED_FIELDS``.
    output_path
        JSONL output path.

    Returns
    -------
    RolloutTraceWriteResult
        Written output path, number of records, and trace columns.

    Raises
    ------
    ValueError
        If the trace is empty or a required field is absent.

    """
    if not trace_records:
        message = "trace_records must contain at least one step"
        raise ValueError(message)

    json_records = [_trace_record_to_json(record) for record in trace_records]
    for index, record in enumerate(json_records):
        missing = [field for field in TRACE_REQUIRED_FIELDS if field not in record]
        if missing:
            message = f"trace record {index} is missing required fields: {', '.join(missing)}"
            raise ValueError(message)

    resolved_path = Path(output_path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record, sort_keys=True) for record in json_records]
    resolved_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    columns = tuple(sorted({key for record in json_records for key in record}))
    return RolloutTraceWriteResult(output_path=str(resolved_path), step_count=len(json_records), columns=columns)


def load_policy_rollout_trace(trace_path: str | Path) -> list[dict[str, Any]]:
    """
    Load a trained-policy rollout trace from newline-delimited JSON.

    Parameters
    ----------
    trace_path
        JSONL trace path created by ``write_policy_rollout_trace``.

    Returns
    -------
    list[dict[str, Any]]
        Per-step rollout records.

    Raises
    ------
    ValueError
        If the trace file contains no records.

    """
    resolved_path = Path(trace_path)
    records = [json.loads(line) for line in resolved_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        message = "rollout trace contains no records"
        raise ValueError(message)
    return records


def _as_task_reference(
    task_or_reference: Mapping[str, Any] | envs.task_adapter.EnvironmentTaskReference,
) -> envs.task_adapter.EnvironmentTaskReference:
    """Return an environment task reference from a mapping or existing reference."""
    if isinstance(task_or_reference, envs.task_adapter.EnvironmentTaskReference):
        return task_or_reference
    return envs.task_adapter.make_task_reference(task_or_reference)


def _as_xyz_offset(offset: object) -> np.ndarray:
    """Return a finite XYZ offset array."""
    offset_array = np.asarray(offset, dtype=float)
    if offset_array.shape != (XYZ_DIMENSIONS,):
        message = "offset must have shape (3,)"
        raise ValueError(message)
    if not np.all(np.isfinite(offset_array)):
        message = "offset must contain only finite values"
        raise ValueError(message)
    return offset_array


def _make_actual_positions(reference_positions: np.ndarray, offset_array: np.ndarray, lag_steps: int) -> np.ndarray:
    """Create deterministic actual positions from reference positions."""
    if lag_steps == 0:
        return np.array(reference_positions + offset_array, dtype=float, copy=True)
    lagged = np.empty_like(reference_positions, dtype=float)
    for index in range(reference_positions.shape[0]):
        source_index = max(0, index - lag_steps)
        lagged[index] = reference_positions[source_index] + offset_array
    return lagged


def _metrics_to_dict(summary: evaluation.trajectory_metrics.TrajectoryMetricSummary, shape: str) -> dict[str, Any]:
    """Convert trajectory metric summary to a JSON-serializable dictionary."""
    return {
        "task_shape": shape,
        "mean_position_error_m": summary.mean_position_error_m,
        "max_position_error_m": summary.max_position_error_m,
        "rmse_position_error_m": summary.rmse_position_error_m,
        "final_position_error_m": summary.final_position_error_m,
        "duration_sec": summary.duration_sec,
        "sample_count": summary.num_samples,
        "num_samples": summary.num_samples,
    }


def _start_hold_metrics(
    reference_data: envs.task_adapter.EnvironmentTaskReference,
    reference: trajectories.primitives.Trajectory,
    actual: trajectories.primitives.Trajectory,
) -> dict[str, Any]:
    """Return hold metadata and tracking-only error metrics."""
    sample_count = int(reference.positions.shape[0])
    start_step = int(reference_data.tracking_phase_start_step) if reference_data.exclude_start_hold_from_tracking_metrics else 0
    end_step = int(reference_data.tracking_phase_end_step) if reference_data.exclude_final_hold_from_tracking_metrics else sample_count
    start_step = min(max(start_step, 0), sample_count - 1)
    end_step = min(max(end_step, start_step + 1), sample_count)
    tracking_reference = trajectories.primitives.Trajectory(
        times=np.array(reference.times[start_step:end_step], dtype=float, copy=True),
        positions=np.array(reference.positions[start_step:end_step], dtype=float, copy=True),
    )
    tracking_actual = trajectories.primitives.Trajectory(
        times=np.array(actual.times[start_step:end_step], dtype=float, copy=True),
        positions=np.array(actual.positions[start_step:end_step], dtype=float, copy=True),
    )
    tracking_errors = evaluation.trajectory_metrics.compute_position_errors(reference=tracking_reference, actual=tracking_actual)
    return {
        "start_hold_enabled": bool(reference_data.start_hold_enabled),
        "start_hold_sec": float(reference_data.start_hold_sec),
        "exclude_start_hold_from_tracking_metrics": bool(reference_data.exclude_start_hold_from_tracking_metrics),
        "tracking_phase_start_step": int(reference_data.tracking_phase_start_step),
        "tracking_phase_start_time_sec": float(reference_data.tracking_phase_start_time_sec),
        "final_hold_enabled": bool(reference_data.final_hold_enabled),
        "final_hold_sec": float(reference_data.final_hold_sec),
        "exclude_final_hold_from_tracking_metrics": bool(reference_data.exclude_final_hold_from_tracking_metrics),
        "tracking_phase_end_step": int(reference_data.tracking_phase_end_step),
        "tracking_phase_end_time_sec": float(reference_data.tracking_phase_end_time_sec),
        "mean_position_error_tracking_m": float(np.mean(tracking_errors)),
    }


def _reward_metrics(reference_data: envs.task_adapter.EnvironmentTaskReference, actual_positions: np.ndarray) -> dict[str, Any]:
    """Compute optional reward summary using the MVP reward helper."""
    rewards = [
        envs.tracking_reward.step_tracking_episode(
            reference=reference_data,
            actual_position=actual_positions[index],
            step_index=index,
        ).reward
        for index in range(actual_positions.shape[0])
    ]
    return {
        "mean_reward": float(np.mean(rewards)),
        "final_reward": float(rewards[-1]),
    }


def _trace_record_to_json(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert a trace record into a JSON-compatible dictionary."""
    return {str(key): _json_ready(value) for key, value in record.items()}


def _json_ready(value: Any) -> Any:
    """Return a JSON-compatible copy of a rollout trace value."""
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_ready(nested_value) for key, nested_value in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


__all__ = [
    "DEFAULT_OUTPUT_FILENAME",
    "TRACE_REQUIRED_FIELDS",
    "RolloutEvaluationResult",
    "RolloutTraceWriteResult",
    "RolloutWriteResult",
    "evaluate_task_rollout",
    "load_policy_rollout_trace",
    "write_policy_rollout_trace",
    "write_task_rollout_evaluation",
]
