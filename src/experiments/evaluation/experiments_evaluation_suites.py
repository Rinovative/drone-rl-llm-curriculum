"""
===============================================================================
experiments_evaluation_suites.py
===============================================================================
Load canonical policy evaluation suite configurations.

Responsibilities:
  - Parse named evaluation suites from YAML configuration files
  - Validate every suite task through deterministic task validation
  - Expose small immutable contracts for curriculum and direct policy evaluation

Design principles:
  - Keep suite parsing independent from PPO rollout execution
  - Preserve task ordering so evaluation artifacts are deterministic
  - Fail clearly on malformed schema, duplicate names, or infeasible tasks

Boundaries:
  - Policy rollout, metrics, plots, and renders belong in policy evaluation
  - Curriculum model selection belongs in curriculum evaluation
===============================================================================

"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src import utils, validation
from src.experiments import experiments_config as config

DEFAULT_RENDER_FPS = 20
DEFAULT_EVALUATION_SUITE_PATH = Path("configs/evaluation/final_benchmark_eval_suite.yaml")


@dataclass(frozen=True)
class EvaluationRenderOptions:
    """
    Render artifact options loaded from an evaluation suite.

    Parameters
    ----------
    enabled
        Whether simulator GIF rendering is enabled for suite evaluations.
    fps
        Requested GIF playback frame rate.
    max_steps
        Optional render rollout length. Uses suite ``eval_steps`` when omitted.

    """

    enabled: bool = True
    fps: int = DEFAULT_RENDER_FPS
    max_steps: int | None = None

    def __post_init__(self) -> None:
        """Validate render options."""
        if self.fps <= 0:
            message = "render.fps must be positive"
            raise ValueError(message)
        if self.max_steps is not None and self.max_steps <= 0:
            message = "render.max_steps must be positive when provided"
            raise ValueError(message)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {
            "enabled": self.enabled,
            "fps": self.fps,
            "max_steps": self.max_steps,
        }


@dataclass(frozen=True)
class EvaluationPlotOptions:
    """
    Plot artifact options loaded from an evaluation suite.

    Parameters
    ----------
    enabled
        Whether canonical trajectory plots are enabled for suite evaluations.

    """

    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {"enabled": self.enabled}


@dataclass(frozen=True)
class EvaluationTraceOptions:
    """
    Trace artifact options loaded from an evaluation suite.

    Parameters
    ----------
    enabled
        Whether JSONL rollout traces are copied for suite evaluations.

    """

    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {"enabled": self.enabled}


@dataclass(frozen=True)
class EvaluationSuiteTask:
    """
    One validated task inside a canonical evaluation suite.

    Parameters
    ----------
    task_name
        Stable task identifier used in artifact paths and result summaries.
    task_shape
        Expected trajectory shape for the task.
    task
        Validated trajectory task mapping consumed by environment builders.

    """

    task_name: str
    task_shape: str
    task: dict[str, Any]

    def __post_init__(self) -> None:
        """Validate task metadata and deterministic feasibility."""
        if not self.task_name.strip():
            message = "task_name must be non-empty"
            raise ValueError(message)
        if not self.task_shape.strip():
            message = f"suite task {self.task_name!r} task_shape must be non-empty"
            raise ValueError(message)
        if str(self.task.get("shape", "")) != self.task_shape:
            message = f"suite task {self.task_name!r} task shape must match task_shape {self.task_shape!r}"
            raise ValueError(message)
        validation_result = validation.tasks.validate_task(self.task)
        if not validation_result.is_valid:
            details = "; ".join(validation_result.messages)
            message = f"invalid suite task {self.task_name!r}: {details}"
            raise ValueError(message)

    def to_task_config_dict(self) -> dict[str, Any]:
        """Return a one-task policy evaluation config payload."""
        return {
            "name": self.task_name,
            "tasks": [copy.deepcopy(self.task)],
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {
            "task_name": self.task_name,
            "task_shape": self.task_shape,
            "task": copy.deepcopy(self.task),
        }


@dataclass(frozen=True)
class EvaluationSuite:
    """
    Canonical config-driven evaluation suite.

    Parameters
    ----------
    evaluation_name
        Stable suite identifier used in evaluation artifact paths.
    seed
        Deterministic seed used by suite evaluations.
    eval_steps
        Deterministic rollout length used by suite evaluations.
    render
        Render artifact options.
    plots
        Plot artifact options.
    traces
        Trace artifact options.
    tasks
        Ordered validated task entries.

    """

    evaluation_name: str
    seed: int
    eval_steps: int
    render: EvaluationRenderOptions
    plots: EvaluationPlotOptions
    traces: EvaluationTraceOptions
    tasks: tuple[EvaluationSuiteTask, ...]

    def __post_init__(self) -> None:
        """Validate suite-level metadata."""
        if not self.evaluation_name.strip():
            message = "evaluation_name must be non-empty"
            raise ValueError(message)
        if self.seed < 0:
            message = "seed must be nonnegative"
            raise ValueError(message)
        if self.eval_steps <= 0:
            message = "eval_steps must be positive"
            raise ValueError(message)
        if not self.tasks:
            message = "evaluation suite must contain at least one task"
            raise ValueError(message)
        task_names = [task.task_name for task in self.tasks]
        if len(task_names) != len(set(task_names)):
            message = "evaluation suite task_name values must be unique"
            raise ValueError(message)

    @property
    def task_names(self) -> list[str]:
        """Return suite task names in deterministic config order."""
        return [task.task_name for task in self.tasks]

    def get_task(self, task_name: str) -> EvaluationSuiteTask:
        """
        Return one suite task by name.

        Parameters
        ----------
        task_name
            Stable task identifier to look up.

        Returns
        -------
        EvaluationSuiteTask
            Matching suite task.

        Raises
        ------
        ValueError
            If the task name is absent from the suite.

        """
        for task in self.tasks:
            if task.task_name == task_name:
                return task
        available = ", ".join(self.task_names)
        message = f"evaluation suite task {task_name!r} not found; available: {available}"
        raise ValueError(message)

    def to_dict(self) -> dict[str, Any]:
        """Return a stable JSON-serializable representation."""
        return {
            "evaluation_name": self.evaluation_name,
            "seed": self.seed,
            "eval_steps": self.eval_steps,
            "render": self.render.to_dict(),
            "plots": self.plots.to_dict(),
            "traces": self.traces.to_dict(),
            "tasks": [task.to_dict() for task in self.tasks],
        }


@dataclass(frozen=True)
class EvaluationSuiteSnapshot:
    """Paths and identity metadata for a copied evaluation suite snapshot."""

    suite_config_path: Path
    suite_config_path_relative: str
    suite_config_sha256: str
    task_config_paths: dict[str, Path]
    task_config_paths_relative: dict[str, str]


def write_evaluation_suite_snapshot(
    run_name: str,
    suite: EvaluationSuite,
    suite_path: str | Path | None = None,
    storage_root: str | Path | None = None,
) -> EvaluationSuiteSnapshot:
    """Copy a suite config and materialize one-task configs under a run."""
    config_dir = utils.artifacts.get_run_config_evaluation_suites_dir(run_name, storage_root)
    config_dir.mkdir(parents=True, exist_ok=True)
    suite_stem = _safe_name(suite.evaluation_name)
    suite_copy_path = config_dir / f"{suite_stem}_eval_suite.yaml"
    suite_text = _suite_snapshot_text(suite=suite, suite_path=suite_path)
    suite_copy_path.write_text(suite_text, encoding="utf-8")

    task_config_dir = config_dir / suite_stem
    task_config_dir.mkdir(parents=True, exist_ok=True)
    task_config_paths: dict[str, Path] = {}
    task_config_paths_relative: dict[str, str] = {}
    for suite_task in suite.tasks:
        task_config_path = task_config_dir / f"{_safe_name(suite_task.task_name)}_task.yaml"
        payload = {
            "name": suite_task.task_name,
            "evaluation_name": suite.evaluation_name,
            "suite_task_name": suite_task.task_name,
            "tasks": [copy.deepcopy(suite_task.task)],
        }
        task_config_path.write_text(_to_yaml(payload), encoding="utf-8")
        task_config_paths[suite_task.task_name] = task_config_path
        task_config_paths_relative[suite_task.task_name] = str(utils.artifacts.path_relative_to_run(task_config_path, run_name, storage_root))

    return EvaluationSuiteSnapshot(
        suite_config_path=suite_copy_path,
        suite_config_path_relative=str(utils.artifacts.path_relative_to_run(suite_copy_path, run_name, storage_root)),
        suite_config_sha256=hashlib.sha256(suite_text.encode("utf-8")).hexdigest(),
        task_config_paths=task_config_paths,
        task_config_paths_relative=task_config_paths_relative,
    )


def _suite_snapshot_text(suite: EvaluationSuite, suite_path: str | Path | None) -> str:
    """Return exact suite source text when available, otherwise a normalized YAML snapshot."""
    if suite_path is not None:
        source_path = Path(suite_path)
        if source_path.is_file():
            return source_path.read_text(encoding="utf-8")
    return _to_yaml(suite.to_dict())


def _to_yaml(payload: Mapping[str, Any]) -> str:
    """Serialize a compact evaluation-suite payload to YAML."""
    import yaml  # noqa: PLC0415

    return yaml.safe_dump(dict(payload), sort_keys=False)


def _safe_name(value: str) -> str:
    """Return a filesystem-safe suite/task path component."""
    text = value.strip().replace(" ", "_")
    return "".join(character if character.isalnum() or character in {"_", "-"} else "_" for character in text)


def load_evaluation_suite(path: str | Path) -> EvaluationSuite:
    """
    Load and validate a canonical evaluation suite YAML file.

    Parameters
    ----------
    path
        Evaluation suite YAML path.

    Returns
    -------
    EvaluationSuite
        Parsed suite with all tasks validated.

    """
    payload = config.load_experiment_config(Path(path))
    return _suite_from_mapping(payload)


def _suite_from_mapping(payload: Mapping[str, Any]) -> EvaluationSuite:
    """Build an evaluation suite from a raw YAML mapping."""
    raw_tasks = payload.get("tasks")
    if not isinstance(raw_tasks, list):
        message = "evaluation suite must contain a top-level 'tasks' list"
        raise TypeError(message)

    tasks: list[EvaluationSuiteTask] = []
    seen_names: set[str] = set()
    for raw_task in raw_tasks:
        task = _suite_task_from_mapping(raw_task)
        if task.task_name in seen_names:
            message = f"duplicate evaluation suite task_name: {task.task_name}"
            raise ValueError(message)
        seen_names.add(task.task_name)
        tasks.append(task)

    return EvaluationSuite(
        evaluation_name=_require_non_empty_string(payload.get("evaluation_name"), "evaluation_name"),
        seed=_require_nonnegative_int(payload.get("seed"), "seed"),
        eval_steps=_require_positive_int(payload.get("eval_steps"), "eval_steps"),
        render=_render_options_from_mapping(_optional_mapping(payload.get("render"), "render")),
        plots=_plot_options_from_mapping(_optional_mapping(payload.get("plots"), "plots")),
        traces=_trace_options_from_mapping(_optional_mapping(payload.get("traces"), "traces")),
        tasks=tuple(tasks),
    )


def _suite_task_from_mapping(raw: Any) -> EvaluationSuiteTask:
    """Build one suite task from a raw YAML value."""
    if not isinstance(raw, Mapping):
        message = "evaluation suite task entry must be a mapping"
        raise TypeError(message)
    task = raw.get("task")
    if not isinstance(task, Mapping):
        message = "evaluation suite task entry must contain a task mapping"
        raise TypeError(message)
    return EvaluationSuiteTask(
        task_name=_require_non_empty_string(raw.get("task_name"), "task_name"),
        task_shape=_require_non_empty_string(raw.get("task_shape"), "task_shape"),
        task=dict(task),
    )


def _render_options_from_mapping(raw: Mapping[str, Any]) -> EvaluationRenderOptions:
    """Build render options from a raw mapping."""
    return EvaluationRenderOptions(
        enabled=_require_bool(raw.get("enabled", True), "render.enabled"),
        fps=_require_positive_int(raw.get("fps", DEFAULT_RENDER_FPS), "render.fps"),
        max_steps=_require_optional_positive_int(raw.get("max_steps"), "render.max_steps"),
    )


def _plot_options_from_mapping(raw: Mapping[str, Any]) -> EvaluationPlotOptions:
    """Build plot options from a raw mapping."""
    return EvaluationPlotOptions(enabled=_require_bool(raw.get("enabled", True), "plots.enabled"))


def _trace_options_from_mapping(raw: Mapping[str, Any]) -> EvaluationTraceOptions:
    """Build trace options from a raw mapping."""
    return EvaluationTraceOptions(enabled=_require_bool(raw.get("enabled", True), "traces.enabled"))


def _optional_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Return an option mapping, treating omitted fields as defaults."""
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        message = f"{field_name} must be a mapping"
        raise TypeError(message)
    return value


def _require_non_empty_string(value: Any, field_name: str) -> str:
    """Return a non-empty string field or raise."""
    if not isinstance(value, str) or not value.strip():
        message = f"{field_name} must be a non-empty string"
        raise ValueError(message)
    return value


def _require_bool(value: Any, field_name: str) -> bool:
    """Return a boolean field or raise."""
    if not isinstance(value, bool):
        message = f"{field_name} must be a boolean"
        raise TypeError(message)
    return value


def _require_positive_int(value: Any, field_name: str) -> int:
    """Return a positive integer field or raise."""
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"{field_name} must be an integer"
        raise TypeError(message)
    if value <= 0:
        message = f"{field_name} must be positive"
        raise ValueError(message)
    return value


def _require_nonnegative_int(value: Any, field_name: str) -> int:
    """Return a nonnegative integer field or raise."""
    if isinstance(value, bool) or not isinstance(value, int):
        message = f"{field_name} must be an integer"
        raise TypeError(message)
    if value < 0:
        message = f"{field_name} must be nonnegative"
        raise ValueError(message)
    return value


def _require_optional_positive_int(value: Any, field_name: str) -> int | None:
    """Return an optional positive integer field or raise."""
    if value is None:
        return None
    return _require_positive_int(value, field_name)


__all__ = [
    "DEFAULT_EVALUATION_SUITE_PATH",
    "DEFAULT_RENDER_FPS",
    "EvaluationPlotOptions",
    "EvaluationRenderOptions",
    "EvaluationSuite",
    "EvaluationSuiteSnapshot",
    "EvaluationSuiteTask",
    "EvaluationTraceOptions",
    "load_evaluation_suite",
    "write_evaluation_suite_snapshot",
]
