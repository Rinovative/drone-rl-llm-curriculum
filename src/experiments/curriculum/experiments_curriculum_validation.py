"""
===============================================================================
experiments_curriculum_validation.py
===============================================================================
Summarize curriculum trajectory tasks from experiment configuration dictionaries.

Responsibilities:
  - Extract configured task lists from already-loaded experiment configs
  - Validate each configured task through the deterministic validation layer
  - Report per-task validation outcomes and aggregate shape counts

Design principles:
  - Keep helpers side-effect free and avoid mutating caller-owned configs
  - Reuse validation.tasks.ValidationResult as the task validation contract

Boundaries:
  - Config loading mechanics stay in experiments_config.py
  - Training orchestration, scheduling, and artifact writing belong elsewhere
===============================================================================

"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src import validation
from src.experiments import experiments_config as config_loader

if TYPE_CHECKING:
    from pathlib import Path

TASKS_KEY = "tasks"
VALIDATION_LIMITS_KEY = "validation_limits"


@dataclass(frozen=True)
class CurriculumTaskSummary:
    """
    Validation summary for one configured curriculum task.

    Parameters
    ----------
    index
        Zero-based task index in the loaded task list.
    shape
        Shape value from the task mapping when present.
    validation_result
        Existing deterministic validation result for this task.

    """

    index: int
    shape: str | None
    validation_result: validation.tasks.ValidationResult

    @property
    def is_valid(self) -> bool:
        """Return whether deterministic validation accepted this task."""
        return self.validation_result.is_valid

    @property
    def messages(self) -> tuple[str, ...]:
        """Return deterministic validation diagnostics for this task."""
        return self.validation_result.messages


@dataclass(frozen=True)
class CurriculumValidationSummary:
    """
    Aggregate validation summary for configured curriculum tasks.

    Parameters
    ----------
    task_summaries
        Per-task validation summaries in config order.
    valid_count
        Number of tasks accepted by deterministic validation.
    invalid_count
        Number of tasks rejected by deterministic validation.
    shape_counts
        Counts of task shapes observed in the task list.

    """

    task_summaries: tuple[CurriculumTaskSummary, ...]
    valid_count: int
    invalid_count: int
    shape_counts: dict[str, int]

    @property
    def total_count(self) -> int:
        """Return the total number of summarized tasks."""
        return len(self.task_summaries)


def summarize_config_tasks(config: Mapping[str, Any]) -> CurriculumValidationSummary:
    """
    Validate and summarize the task list in an experiment configuration.

    The helper currently supports the top-level ``tasks`` list used by
    ``tests/fixtures/configs/smoke/trajectory_validation.yaml``. If a top-level
    ``validation_limits`` mapping is present, it is used to construct
    ``validation.tasks.ValidationLimits`` for all task checks.

    Parameters
    ----------
    config
        Already-loaded experiment configuration mapping.

    Returns
    -------
    CurriculumValidationSummary
        Per-task deterministic validation results and aggregate counts.

    Raises
    ------
    ValueError
        If the config has no task list, a non-list task container, invalid
        validation limits, or non-mapping task entries.

    """
    tasks = _extract_task_list(config)
    limits = _validation_limits_from_config(config)
    summaries: list[CurriculumTaskSummary] = []
    for index, task in enumerate(tasks):
        if not isinstance(task, Mapping):
            message = f"task at index {index} must be a mapping"
            raise ValueError(message)  # noqa: TRY004 - task spec requires ValueError.
        task_copy = dict(task)
        result = validation.tasks.validate_task(task_copy, limits=limits)
        shape = task_copy.get(validation.contracts.FIELD_SHAPE)
        summaries.append(
            CurriculumTaskSummary(
                index=index,
                shape=str(shape) if shape is not None else None,
                validation_result=result,
            )
        )

    valid_count = sum(summary.is_valid for summary in summaries)
    invalid_count = len(summaries) - valid_count
    shape_counts = dict(Counter(summary.shape for summary in summaries if summary.shape is not None))
    return CurriculumValidationSummary(
        task_summaries=tuple(summaries),
        valid_count=valid_count,
        invalid_count=invalid_count,
        shape_counts=shape_counts,
    )


def summarize_config_path(path: str | Path) -> CurriculumValidationSummary:
    """
    Load an experiment config path and summarize its curriculum tasks.

    Parameters
    ----------
    path
        YAML configuration path accepted by ``config_loader.load_experiment_config``.

    Returns
    -------
    CurriculumValidationSummary
        Per-task deterministic validation results and aggregate counts.

    """
    loaded_config = config_loader.load_experiment_config(path)
    return summarize_config_tasks(loaded_config)


def summarize_task_shapes(config: Mapping[str, Any]) -> dict[str, int]:
    """
    Count task shapes in an experiment configuration without mutating it.

    Parameters
    ----------
    config
        Already-loaded experiment configuration mapping containing a top-level
        ``tasks`` list.

    Returns
    -------
    dict[str, int]
        Mapping from shape name to number of configured tasks with that shape.

    Raises
    ------
    ValueError
        If task validation or task-list extraction fails.

    """
    return dict(summarize_config_tasks(config).shape_counts)


def _extract_task_list(config: Mapping[str, Any]) -> Sequence[Any]:
    """Return the top-level task list from a config mapping."""
    if TASKS_KEY not in config:
        message = f"experiment config must contain a top-level {TASKS_KEY!r} list"
        raise ValueError(message)
    tasks = config[TASKS_KEY]
    if not isinstance(tasks, list):
        message = f"experiment config {TASKS_KEY!r} must be a list"
        raise ValueError(message)  # noqa: TRY004 - task spec requires ValueError.
    return tasks


def _validation_limits_from_config(config: Mapping[str, Any]) -> validation.tasks.ValidationLimits | None:
    """Build validation limits from config metadata when present."""
    if VALIDATION_LIMITS_KEY not in config:
        return None
    raw_limits = config[VALIDATION_LIMITS_KEY]
    if not isinstance(raw_limits, Mapping):
        message = f"experiment config {VALIDATION_LIMITS_KEY!r} must be a mapping when present"
        raise ValueError(message)  # noqa: TRY004 - public config contract requires ValueError.
    try:
        return validation.tasks.ValidationLimits(**dict(raw_limits))
    except TypeError as exc:
        message = f"invalid validation limits: {exc}"
        raise ValueError(message) from exc


__all__ = [
    "CurriculumTaskSummary",
    "CurriculumValidationSummary",
    "summarize_config_path",
    "summarize_config_tasks",
    "summarize_task_shapes",
]
