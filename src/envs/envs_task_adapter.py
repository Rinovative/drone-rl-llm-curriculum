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

    """

    task: Mapping[str, Any]
    shape: str
    times: np.ndarray
    positions: np.ndarray
    validation_messages: tuple[str, ...]


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
    return EnvironmentTaskReference(
        task=MappingProxyType(task_copy),
        shape=str(shape),
        times=np.array(result.trajectory.times, dtype=float, copy=True),
        positions=np.array(result.trajectory.positions, dtype=float, copy=True),
        validation_messages=result.messages,
    )


__all__ = [
    "EnvironmentTaskReference",
    "make_task_reference",
]
