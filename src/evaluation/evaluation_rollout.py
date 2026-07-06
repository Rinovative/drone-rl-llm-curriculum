"""
===============================================================================
evaluation_rollout.py
===============================================================================
Run deterministic MVP rollout evaluation for validated trajectory tasks.

Responsibilities:
  - Convert task mappings or task references into comparable trajectory samples
  - Generate deterministic baseline actual trajectories for smoke evaluation
  - Write small JSON rollout metric artifacts under approved results paths

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
    from collections.abc import Mapping

XYZ_DIMENSIONS = 3
DEFAULT_OFFSET = (0.05, 0.0, 0.0)
DEFAULT_OUTPUT_FILENAME = "rollout_metrics.json"


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


__all__ = [
    "DEFAULT_OUTPUT_FILENAME",
    "RolloutEvaluationResult",
    "RolloutWriteResult",
    "evaluate_task_rollout",
    "write_task_rollout_evaluation",
]
