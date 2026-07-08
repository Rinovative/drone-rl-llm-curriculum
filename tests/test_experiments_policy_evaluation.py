"""Tests for the shared policy evaluation helper."""

# ruff: noqa: S101, TC002, PT018, ARG005

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from src.experiments.evaluation import experiments_evaluation_policy as policy_evaluation

FULL_EVALUATION_EPISODE_COUNT = 2


@dataclass
class _FakeDiagnostics:
    metrics: dict[str, Any]
    trace_records: list[dict[str, Any]]


@dataclass
class _FakePlotResult:
    plot_paths: dict[str, str]


def _write_task_config(path: Path, shape: str = "line") -> None:
    if shape == "line":
        payload = """name: eval
tasks:
  - task_type: trajectory
    shape: line
    duration_sec: 3.0
    sample_rate_hz: 10.0
    start: [0.0, 0.0, 1.0]
    end: [1.0, 0.0, 1.0]
"""
    else:
        payload = """name: eval
tasks:
  - task_type: trajectory
    shape: hover_stabilization
    duration_sec: 2.0
    sample_rate_hz: 10.0
    position: [0.0, 0.0, 1.0]
"""
    path.write_text(payload, encoding="utf-8")


def test_shared_policy_evaluation_writes_deterministic_artifact_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify helper writes deterministic metrics/manifest and review artifact paths."""
    model_path = tmp_path / "model.zip"
    task_config = tmp_path / "task.yaml"
    output_dir = tmp_path / "evaluation"
    model_path.write_bytes(b"model")
    _write_task_config(task_config, shape="line")

    def fake_collect_diagnostics(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        diagnostics_dir: Path,
    ) -> tuple[_FakeDiagnostics, dict[str, Any]]:
        del spec, task
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        trace_path = diagnostics_dir / "evaluation_trace.jsonl"
        trace_path.write_text("{}\n", encoding="utf-8")
        return (
            _FakeDiagnostics(
                metrics={
                    "episode_count": FULL_EVALUATION_EPISODE_COUNT,
                    "eval_resets": 1,
                    "eval_truncated_count": 1,
                    "start_hold_enabled": True,
                    "start_hold_sec": 1.0,
                    "exclude_start_hold_from_tracking_metrics": True,
                    "tracking_phase_start_step": 10,
                    "tracking_phase_start_time_sec": 1.0,
                    "mean_position_error_m": 0.1,
                    "mean_position_error_tracking_m": 0.08,
                    "final_position_error_m": 0.2,
                    "max_position_error_m": 0.3,
                    "actual_xy_span_m": 0.4,
                    "reference_xy_span_m": 0.5,
                    "xy_tracking_ratio": 0.8,
                    "action_saturation_fraction": [0.0, 0.0, 0.0],
                    "real_action_saturation_fraction": [0.0, 0.0, 0.0],
                    "failure_overall_status": "passed",
                    "failure_primary_mode": "none",
                    "failure_modes": [],
                },
                trace_records=[
                    {
                        "source": "evaluation",
                        "step_index": 0,
                        "episode_index": 0,
                        "time_sec": 0.0,
                        "actual_position_xyz_m": [0.0, 0.0, 1.0],
                        "reference_position_xyz_m": [0.0, 0.0, 1.0],
                        "position_error_m": 0.0,
                        "terminated": False,
                        "truncated": True,
                    },
                    {
                        "source": "evaluation",
                        "step_index": 1,
                        "episode_index": 1,
                        "time_sec": 0.0,
                        "actual_position_xyz_m": [0.0, 0.0, 1.0],
                        "reference_position_xyz_m": [0.0, 0.0, 1.0],
                        "position_error_m": 0.0,
                        "terminated": False,
                        "truncated": False,
                    },
                ],
            ),
            {
                "evaluation_trace_path": str(trace_path),
                "failure_report_path": str(diagnostics_dir / "failure_report.json"),
                "episode_summaries_path": str(diagnostics_dir / "episode_summaries.json"),
                "curriculum_feedback_path": str(diagnostics_dir / "curriculum_feedback.json"),
            },
        )

    plotted_records: list[dict[str, Any]] = []

    def fake_write_plots(records: list[dict[str, Any]], plots_dir: Path) -> _FakePlotResult:
        plotted_records.extend(records)
        plots_dir.mkdir(parents=True, exist_ok=True)
        plot_paths: dict[str, str] = {}
        for name in policy_evaluation.evaluation.plots.CANONICAL_POLICY_PLOT_FILENAMES.values():
            plot_path = plots_dir / name
            plot_path.write_bytes(b"plot")
            plot_paths[plot_path.stem] = str(plot_path)
        return _FakePlotResult(plot_paths=plot_paths)

    def fake_render(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        renders_dir: Path,
        render_steps: int,
        render_fps: int,
    ) -> policy_evaluation._RenderArtifactResult:
        del spec, task, render_steps, render_fps
        renders_dir.mkdir(parents=True, exist_ok=True)
        path = renders_dir / "scenario_rollout.gif"
        path.write_bytes(b"GIF89a")
        return policy_evaluation._RenderArtifactResult(  # noqa: SLF001
            gif_path=path,
            warnings=[],
            trace_records=[
                {
                    "source": "render",
                    "step_index": 0,
                    "episode_index": 0,
                    "time_sec": 0.0,
                    "actual_position_xyz_m": [0.0, 0.0, 1.0],
                    "reference_position_xyz_m": [0.0, 0.0, 1.0],
                    "position_error_m": 0.0,
                    "terminated": False,
                    "truncated": True,
                }
            ],
        )

    monkeypatch.setattr(policy_evaluation, "_collect_diagnostics", fake_collect_diagnostics)
    monkeypatch.setattr(policy_evaluation.evaluation.plots, "write_policy_rollout_trace_plots", fake_write_plots)
    monkeypatch.setattr(policy_evaluation, "_write_render_artifact", fake_render)

    result = policy_evaluation.run_policy_evaluation(
        policy_evaluation.PolicyEvaluationSpec(
            label="stage01_hover_stabilization",
            model_role="stage",
            model_path=model_path,
            task_config_path=task_config,
            task_shape="line",
            output_dir=output_dir,
            eval_steps=120,
            seed=0,
        )
    )

    assert result.metrics_path.endswith("metrics/stage01_hover_stabilization_metrics.json")
    assert result.manifest_path.endswith("manifests/stage01_hover_stabilization_manifest.json")
    assert result.trace_path is not None and result.trace_path.endswith("traces/evaluation_trace.jsonl")
    assert result.gif_path is not None and result.gif_path.endswith("renders/scenario_rollout.gif")
    assert Path(result.metrics_path).exists()
    assert Path(result.manifest_path).exists()
    assert Path(result.trace_path).exists()
    assert result.plot_paths["trajectory_xy"].endswith("trajectory_xy.png")
    assert result.plot_paths["trajectory_xyz"].endswith("trajectory_xyz.png")
    assert result.plot_paths["position_error"].endswith("position_error.png")
    assert result.plot_paths["action_trace"].endswith("action_trace.png")
    assert result.plot_trace_scope == "render_rollout"
    assert result.metrics["plot_trace_scope"] == "render_rollout"
    assert result.metrics["plot_trace_step_count"] == 1
    assert result.metrics["plot_trace_truncated"] is True
    assert result.metrics["episode_count"] == FULL_EVALUATION_EPISODE_COUNT
    assert result.metrics["eval_resets"] == 1
    assert [record["source"] for record in plotted_records] == ["render"]


def test_shared_policy_evaluation_no_render_records_flag_and_skips_gif(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify render disabling prevents GIF generation and records render_enabled=false."""
    model_path = tmp_path / "model.zip"
    task_config = tmp_path / "task.yaml"
    model_path.write_bytes(b"model")
    _write_task_config(task_config)

    def fake_collect(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        diagnostics_dir: Path,
    ) -> tuple[_FakeDiagnostics, dict[str, Any]]:
        del spec, task
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        trace = diagnostics_dir / "evaluation_trace.jsonl"
        trace.write_text("{}\n", encoding="utf-8")
        return _FakeDiagnostics(metrics={}, trace_records=[{"source": "evaluation", "terminated": False, "truncated": False}]), {
            "evaluation_trace_path": str(trace)
        }

    plotted_records: list[dict[str, Any]] = []

    def fake_write_plots(records: list[dict[str, Any]], output: Path) -> _FakePlotResult:
        del output
        plotted_records.extend(records)
        return _FakePlotResult(plot_paths={})

    monkeypatch.setattr(policy_evaluation, "_collect_diagnostics", fake_collect)
    monkeypatch.setattr(
        policy_evaluation.evaluation.plots,
        "write_policy_rollout_trace_plots",
        fake_write_plots,
    )

    result = policy_evaluation.run_policy_evaluation(
        policy_evaluation.PolicyEvaluationSpec(
            label="stage02_line",
            model_role="stage",
            model_path=model_path,
            task_config_path=task_config,
            task_shape="line",
            output_dir=tmp_path / "out",
            eval_steps=120,
            seed=0,
        ),
        policy_evaluation.PolicyEvaluationArtifactOptions(render_enabled=False),
    )

    assert result.render_enabled is False
    assert result.gif_path is None
    assert result.plot_trace_scope == "full_evaluation"
    assert result.metrics["plot_trace_scope"] == "full_evaluation"
    assert [record["source"] for record in plotted_records] == ["evaluation"]


def test_shared_policy_evaluation_no_plots_records_flag_and_skips_plot_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify plot disabling leaves plot_paths empty and records plots_enabled=false."""
    model_path = tmp_path / "model.zip"
    task_config = tmp_path / "task.yaml"
    model_path.write_bytes(b"model")
    _write_task_config(task_config)

    def fake_collect(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        diagnostics_dir: Path,
    ) -> tuple[_FakeDiagnostics, dict[str, Any]]:
        del spec, task
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        trace = diagnostics_dir / "evaluation_trace.jsonl"
        trace.write_text("{}\n", encoding="utf-8")
        return _FakeDiagnostics(metrics={}, trace_records=[]), {"evaluation_trace_path": str(trace)}

    monkeypatch.setattr(policy_evaluation, "_collect_diagnostics", fake_collect)
    monkeypatch.setattr(
        policy_evaluation,
        "_write_render_artifact",
        lambda *args, **kwargs: policy_evaluation._RenderArtifactResult(  # noqa: SLF001
            gif_path=tmp_path / "render.gif",
            warnings=[],
            trace_records=[{"source": "render", "terminated": False, "truncated": False}],
        ),
    )

    result = policy_evaluation.run_policy_evaluation(
        policy_evaluation.PolicyEvaluationSpec(
            label="stage02_line",
            model_role="stage",
            model_path=model_path,
            task_config_path=task_config,
            task_shape="line",
            output_dir=tmp_path / "out",
            eval_steps=120,
            seed=0,
        ),
        policy_evaluation.PolicyEvaluationArtifactOptions(plots_enabled=False),
    )

    assert result.plots_enabled is False
    assert result.plot_paths == {}
    assert result.plot_trace_scope == "disabled"
    assert result.metrics["plot_trace_scope"] == "disabled"
