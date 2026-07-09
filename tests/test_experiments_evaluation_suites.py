"""Tests for canonical evaluation suite loading."""

# ruff: noqa: S101, PLR2004, TC003

from __future__ import annotations

from pathlib import Path

import pytest

from src.experiments.evaluation import experiments_evaluation_suites as evaluation_suites


def _suite_yaml(evaluation_name: str = "mini_suite", eval_steps: int = 120) -> str:
    return f"""evaluation_name: {evaluation_name}
seed: 3
eval_steps: {eval_steps}

render:
  enabled: true
  fps: 12
  max_steps: 40

plots:
  enabled: false

traces:
  enabled: true

tasks:
  - task_name: line_basic
    task_shape: line
    task:
      task_type: trajectory
      shape: line
      duration_sec: 3.0
      sample_rate_hz: 10.0
      start: [0.0, 0.0, 1.0]
      end: [1.0, 0.0, 1.0]
  - task_name: hover_basic
    task_shape: hover_stabilization
    task:
      task_type: trajectory
      shape: hover_stabilization
      duration_sec: 2.0
      sample_rate_hz: 10.0
      position: [0.0, 0.0, 1.0]
"""


def _write_suite(path: Path, text: str | None = None) -> Path:
    path.write_text(_suite_yaml() if text is None else text, encoding="utf-8")
    return path


def test_valid_evaluation_suite_loads_successfully(tmp_path: Path) -> None:
    """Verify a canonical suite loads with validated options and tasks."""
    suite = evaluation_suites.load_evaluation_suite(_write_suite(tmp_path / "suite.yaml"))

    assert suite.evaluation_name == "mini_suite"
    assert suite.seed == 3
    assert suite.eval_steps == 120
    assert suite.render.enabled is True
    assert suite.render.fps == 12
    assert suite.render.max_steps == 40
    assert suite.plots.enabled is False
    assert suite.traces.enabled is True
    assert suite.task_names == ["line_basic", "hover_basic"]
    assert suite.get_task("line_basic").task_shape == "line"


def test_missing_evaluation_name_fails(tmp_path: Path) -> None:
    """Verify suite schema requires an evaluation_name."""
    suite_path = _write_suite(tmp_path / "suite.yaml", _suite_yaml().replace("evaluation_name: mini_suite\n", ""))

    with pytest.raises(ValueError, match="evaluation_name"):
        evaluation_suites.load_evaluation_suite(suite_path)


def test_invalid_eval_steps_fails(tmp_path: Path) -> None:
    """Verify suite schema rejects nonpositive eval_steps."""
    suite_path = _write_suite(tmp_path / "suite.yaml", _suite_yaml(eval_steps=0))

    with pytest.raises(ValueError, match="eval_steps must be positive"):
        evaluation_suites.load_evaluation_suite(suite_path)


def test_invalid_task_fails_through_deterministic_validation(tmp_path: Path) -> None:
    """Verify every suite task passes deterministic task validation."""
    suite_path = _write_suite(tmp_path / "suite.yaml", _suite_yaml().replace("duration_sec: 3.0", "duration_sec: 0.1", 1))

    with pytest.raises(ValueError, match="invalid suite task 'line_basic'"):
        evaluation_suites.load_evaluation_suite(suite_path)


def test_duplicate_task_name_fails(tmp_path: Path) -> None:
    """Verify duplicate task names are rejected before evaluation planning."""
    suite_path = _write_suite(tmp_path / "suite.yaml", _suite_yaml().replace("task_name: hover_basic", "task_name: line_basic"))

    with pytest.raises(ValueError, match="duplicate evaluation suite task_name: line_basic"):
        evaluation_suites.load_evaluation_suite(suite_path)


def test_suite_task_lookup_is_deterministic(tmp_path: Path) -> None:
    """Verify task lookup preserves config order and reports available names."""
    suite = evaluation_suites.load_evaluation_suite(_write_suite(tmp_path / "suite.yaml"))

    assert suite.task_names == ["line_basic", "hover_basic"]
    assert suite.get_task("hover_basic").task["shape"] == "hover_stabilization"
    with pytest.raises(ValueError, match="available: line_basic, hover_basic"):
        suite.get_task("missing")


def test_suite_to_dict_output_is_stable(tmp_path: Path) -> None:
    """Verify suite serialization emits the canonical schema shape."""
    suite = evaluation_suites.load_evaluation_suite(_write_suite(tmp_path / "suite.yaml"))

    assert suite.to_dict() == {
        "evaluation_name": "mini_suite",
        "seed": 3,
        "eval_steps": 120,
        "render": {"enabled": True, "fps": 12, "max_steps": 40},
        "plots": {"enabled": False},
        "traces": {"enabled": True},
        "tasks": [
            {
                "task_name": "line_basic",
                "task_shape": "line",
                "task": {
                    "task_type": "trajectory",
                    "shape": "line",
                    "duration_sec": 3.0,
                    "sample_rate_hz": 10.0,
                    "start": [0.0, 0.0, 1.0],
                    "end": [1.0, 0.0, 1.0],
                },
            },
            {
                "task_name": "hover_basic",
                "task_shape": "hover_stabilization",
                "task": {
                    "task_type": "trajectory",
                    "shape": "hover_stabilization",
                    "duration_sec": 2.0,
                    "sample_rate_hz": 10.0,
                    "position": [0.0, 0.0, 1.0],
                },
            },
        ],
    }


def test_real_evaluation_suites_load_through_suite_loader() -> None:
    """Verify every report-facing evaluation suite loads through the canonical loader."""
    expected = {
        "configs/evaluation/line_eval_suite.yaml": ["line_basic"],
        "configs/evaluation/final_benchmark_eval_suite.yaml": [
            "line_basic",
            "line_long_final",
            "line_diagonal",
            "line_reverse",
        ],
        "configs/evaluation/generalization_eval_suite.yaml": [
            "hover_center",
            "vertical_basic",
            "line_basic",
            "diagonal_line_basic",
            "short_line_start_hold",
            "polyline_l_basic",
            "circle_basic",
            "ellipse_basic",
            "figure_eight_basic",
        ],
    }

    for suite_path, task_names in expected.items():
        suite = evaluation_suites.load_evaluation_suite(suite_path)
        assert suite.task_names == task_names


def test_final_benchmark_suite_is_line_focused() -> None:
    """Verify the primary final benchmark is fair for line-trained policies."""
    suite = evaluation_suites.load_evaluation_suite("configs/evaluation/final_benchmark_eval_suite.yaml")

    assert suite.evaluation_name == "final_benchmark"
    assert suite.eval_steps == 360
    assert len(suite.tasks) >= 3
    assert {task.task_shape for task in suite.tasks} == {"line"}
    assert "line_long_final" in suite.task_names


def test_generalization_suite_contains_optional_non_line_tasks() -> None:
    """Verify optional OOD/generalization tasks are separated from the primary benchmark."""
    suite = evaluation_suites.load_evaluation_suite("configs/evaluation/generalization_eval_suite.yaml")

    assert suite.evaluation_name == "generalization"
    assert suite.eval_steps == 420
    assert {task.task_shape for task in suite.tasks} == {
        "circle",
        "ellipse",
        "figure_eight",
        "hover_stabilization",
        "line",
        "polyline",
        "start_hold_then_short_line",
        "vertical",
    }
    assert all(task.task["final_hold_enabled"] is True for task in suite.tasks)
    assert all(task.task["final_hold_sec"] == 1.0 for task in suite.tasks)
