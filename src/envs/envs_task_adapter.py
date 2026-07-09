"""
===============================================================================
envs_task_adapter.py
===============================================================================
Convert validated trajectory tasks into environment-ready reference data.

Responsibilities:
  - Validate task mappings through the deterministic validation layer
  - Copy sampled trajectory times and XYZ positions for environment consumers
  - Package task metadata and validation diagnostics for later integration

Design principles:
  - Keep the adapter lightweight and independent of simulator construction
  - Copy caller-owned task data and trajectory arrays before returning them

Boundaries:
  - PyBullet environment construction stays in envs_builders.py
  - Reward wiring, scheduling, rollout collection, and training belong elsewhere
===============================================================================

"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from src import validation

if TYPE_CHECKING:
    from collections.abc import Mapping


@dataclass(frozen=True)
class EnvironmentTaskReference:
    """
    Environment-ready reference data for a validated trajectory task.

    Parameters
    ----------
    task
        Top-level immutable copy of the original task mapping for metadata.
    shape
        Validated trajectory shape name.
    times
        Copied one-dimensional reference time samples in seconds.
    positions
        Copied XYZ reference positions with shape ``(num_samples, 3)``.
    validation_messages
        Validation diagnostics returned by deterministic validation.
    start_hold_enabled
        Whether the reference includes a prepended stationary start-hold phase.
    start_hold_sec
        Effective start-hold duration in seconds.
    exclude_start_hold_from_tracking_metrics
        Whether tracking-only metrics should omit start-hold rows.
    tracking_phase_start_step
        First reference row considered part of moving tracking.
    tracking_phase_start_time_sec
        Reference time in seconds for ``tracking_phase_start_step``.
    final_hold_enabled
        Whether the reference includes an appended stationary final-hold phase.
    final_hold_sec
        Effective final-hold duration in seconds.
    exclude_final_hold_from_tracking_metrics
        Whether tracking-only metrics should omit final-hold rows.
    tracking_phase_end_step
        Exclusive reference row where moving tracking ends.
    tracking_phase_end_time_sec
        Reference time in seconds at the end of moving tracking.

    """

    task: Mapping[str, Any]
    shape: str
    times: np.ndarray
    positions: np.ndarray
    validation_messages: tuple[str, ...]
    start_hold_enabled: bool = False
    start_hold_sec: float = 0.0
    exclude_start_hold_from_tracking_metrics: bool = False
    tracking_phase_start_step: int = 0
    tracking_phase_start_time_sec: float = 0.0
    final_hold_enabled: bool = False
    final_hold_sec: float = 0.0
    exclude_final_hold_from_tracking_metrics: bool = False
    tracking_phase_end_step: int = 0
    tracking_phase_end_time_sec: float = 0.0


def make_task_reference(
    task: Mapping[str, Any],
    limits: validation.tasks.ValidationLimits | None = None,
) -> EnvironmentTaskReference:
    """
    Validate a task and package copied trajectory reference data.

    Parameters
    ----------
    task
        Mapping describing a trajectory task. The mapping is copied with
        ``dict(task)`` before validation and metadata packaging.
    limits
        Optional deterministic validation limits.

    Returns
    -------
    EnvironmentTaskReference
        Copied task metadata, shape, time samples, positions, and diagnostics.

    Raises
    ------
    ValueError
        If deterministic validation rejects the task or returns no trajectory.

    """
    task_copy = dict(task)
    result = validation.tasks.validate_task(task_copy, limits=limits)
    if not result.is_valid:
        message = "invalid trajectory task: " + "; ".join(result.messages)
        raise ValueError(message)
    if result.trajectory is None:
        message = "valid trajectory task did not produce reference trajectory data"
        raise ValueError(message)

    shape = task_copy.get(validation.contracts.FIELD_SHAPE)
    task_copy.setdefault(validation.contracts.FIELD_START_HOLD_ENABLED, result.start_hold_enabled)
    task_copy.setdefault(validation.contracts.FIELD_START_HOLD_SEC, result.start_hold_sec)
    task_copy.setdefault(
        validation.contracts.FIELD_EXCLUDE_START_HOLD_FROM_TRACKING_METRICS,
        result.exclude_start_hold_from_tracking_metrics,
    )
    task_copy.setdefault(validation.contracts.FIELD_FINAL_HOLD_ENABLED, result.final_hold_enabled)
    task_copy.setdefault(validation.contracts.FIELD_FINAL_HOLD_SEC, result.final_hold_sec)
    task_copy.setdefault(
        validation.contracts.FIELD_EXCLUDE_FINAL_HOLD_FROM_TRACKING_METRICS,
        result.exclude_final_hold_from_tracking_metrics,
    )
    return EnvironmentTaskReference(
        task=MappingProxyType(task_copy),
        shape=str(shape),
        times=np.array(result.trajectory.times, dtype=float, copy=True),
        positions=np.array(result.trajectory.positions, dtype=float, copy=True),
        validation_messages=result.messages,
        start_hold_enabled=result.start_hold_enabled,
        start_hold_sec=result.start_hold_sec,
        exclude_start_hold_from_tracking_metrics=result.exclude_start_hold_from_tracking_metrics,
        tracking_phase_start_step=result.tracking_phase_start_step,
        tracking_phase_start_time_sec=result.tracking_phase_start_time_sec,
        final_hold_enabled=result.final_hold_enabled,
        final_hold_sec=result.final_hold_sec,
        exclude_final_hold_from_tracking_metrics=result.exclude_final_hold_from_tracking_metrics,
        tracking_phase_end_step=result.tracking_phase_end_step,
        tracking_phase_end_time_sec=result.tracking_phase_end_time_sec,
    )


__all__ = [
    "EnvironmentTaskReference",
    "make_task_reference",
]
