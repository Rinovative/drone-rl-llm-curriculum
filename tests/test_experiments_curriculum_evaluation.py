"""Tests for curriculum evaluation orchestration through shared policy evaluation."""

# ruff: noqa: S101, PLR2004

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.experiments.cli import experiments_cli_evaluate_curriculum as cli_evaluate_curriculum
from src.experiments.curriculum import experiments_curriculum_evaluation as curriculum_evaluation
from src.experiments.evaluation import experiments_evaluation_policy as policy_evaluation


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _curriculum_summary(tmp_path: Path) -> Path:
    stage_root = tmp_path / "training"
    stage_root.mkdir(parents=True, exist_ok=True)
    stage_one_manifest = stage_root / "stage01_manifest.json"
    stage_two_manifest = stage_root / "stage02_manifest.json"
    stage_one_task = stage_root / "stage01_task.yaml"
    stage_two_task = stage_root / "stage02_task.yaml"
    stage_one_task.write_text(
        """name: stage01
tasks:
  - task_type: trajectory
    shape: hover_stabilization
    duration_sec: 2.0
    sample_rate_hz: 10.0
    position: [0.0, 0.0, 1.0]
""",
        encoding="utf-8",
    )
    stage_two_task.write_text(
        """name: stage02
tasks:
  - task_type: trajectory
    shape: line
    duration_sec: 3.0
    sample_rate_hz: 10.0
    start: [0.0, 0.0, 1.0]
    end: [1.0, 0.0, 1.0]
""",
        encoding="utf-8",
    )

    _write_json(
        stage_one_manifest,
        {
            "task_config_path": str(stage_one_task),
            "task_index": 0,
            "eval_steps": 120,
            "total_timesteps": 4096,
            "normalize_actions": True,
        },
    )
    _write_json(
        stage_two_manifest,
        {
            "task_config_path": str(stage_two_task),
            "task_index": 0,
            "eval_steps": 120,
            "total_timesteps": 4096,
            "normalize_actions": True,
            "action_interface": "direct_rpm",
            "rpm_delta_scale": 0.07,
            "include_dynamics_observation": True,
            "include_previous_action": True,
        },
    )

    summary_path = tmp_path / "storage" / "runs" / "curriculum_manual_line_smoke_seed0" / "run_manifest.json"
    _write_json(
        summary_path,
        {
            "curriculum_name": "curriculum_manual_line_smoke",
            "run_name": "curriculum_manual_line_smoke_seed0",
            "run_kind": "curriculum",
            "curriculum_kind": "manual",
            "seed": 0,
            "final_model_path": str(stage_root / "stage02_model_best.zip"),
            "final_model_source": "best",
            "stages": [
                {
                    "stage_index": 1,
                    "stage_name": "hover_stabilization",
                    "task_shape": "hover_stabilization",
                    "run_name": "curriculum_manual_line_smoke_stage01_hover_stabilization_seed0",
                    "model_path": str(stage_root / "stage01_model.zip"),
                    "last_model_path": str(stage_root / "stage01_model.zip"),
                    "last_model_path_relative": "training/stage01_model.zip",
                    "best_model_path": None,
                    "manifest_path": str(stage_one_manifest),
                    "eval_steps": 120,
                    "seed": 0,
                    "total_timesteps": 4096,
                    "normalize_actions": True,
                },
                {
                    "stage_index": 2,
                    "stage_name": "line",
                    "task_shape": "line",
                    "run_name": "curriculum_manual_line_smoke_stage02_line_seed0",
                    "model_path": str(stage_root / "stage02_model.zip"),
                    "last_model_path": str(stage_root / "stage02_model.zip"),
                    "last_model_path_relative": "training/stage02_model.zip",
                    "best_model_path": str(stage_root / "stage02_model_best.zip"),
                    "best_model_path_relative": "training/stage02_model_best.zip",
                    "best_model_metric": "mean_position_error_m",
                    "best_model_step": 4096,
                    "best_model_source": "unit_test_best",
                    "manifest_path": str(stage_two_manifest),
                    "eval_steps": 120,
                    "seed": 0,
                    "total_timesteps": 4096,
                    "normalize_actions": True,
                },
            ],
        },
    )
    return summary_path


def _suite_config(tmp_path: Path, evaluation_name: str = "final_suite") -> Path:
    config_path = tmp_path / f"{evaluation_name}.yaml"
    config_path.write_text(
        f"""evaluation_name: {evaluation_name}
seed: 7
eval_steps: 88
render:
  enabled: false
  fps: 12
  max_steps: 40
plots:
  enabled: false
traces:
  enabled: false
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
""",
        encoding="utf-8",
    )
    return config_path


def _one_task_suite(tmp_path: Path) -> Path:
    config_path = tmp_path / "line_suite.yaml"
    config_path.write_text(
        """evaluation_name: line_suite
seed: 5
eval_steps: 77
render:
  enabled: true
  fps: 20
  max_steps: null
plots:
  enabled: true
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
""",
        encoding="utf-8",
    )
    return config_path


def _fake_policy_evaluation(spec: object, artifacts: object) -> policy_evaluation.PolicyEvaluationResult:
    assert isinstance(spec, policy_evaluation.PolicyEvaluationSpec)
    assert isinstance(artifacts, policy_evaluation.PolicyEvaluationArtifactOptions)
    output_dir = spec.output_dir
    diagnostics_dir = output_dir / "diagnostics"
    traces_dir = output_dir / "traces"
    plots_dir = output_dir / "plots"
    renders_dir = output_dir / "renders"
    metrics_dir = output_dir / "metrics"
    manifests_dir = output_dir / "manifests"
    for directory in (diagnostics_dir, traces_dir, plots_dir, renders_dir, metrics_dir, manifests_dir):
        directory.mkdir(parents=True, exist_ok=True)

    trace_path = traces_dir / "evaluation_trace.jsonl" if artifacts.trace_enabled else None
    gif_path = renders_dir / "scenario_rollout.gif" if artifacts.render_enabled else None
    xy_plot = plots_dir / "trajectory_xy.png" if artifacts.plots_enabled else None
    if trace_path is not None:
        trace_path.write_text("{}\n", encoding="utf-8")
    if gif_path is not None:
        gif_path.write_bytes(b"GIF89a")
    if xy_plot is not None:
        xy_plot.write_bytes(b"plot")

    metrics_path = metrics_dir / f"{spec.label}_metrics.json"
    manifest_path = manifests_dir / f"{spec.label}_manifest.json"

    payload = {
        "label": spec.label,
        "model_role": spec.model_role,
        "evaluation_name": spec.evaluation_name,
        "evaluation_suite_name": spec.evaluation_suite_name,
        "suite_task_name": spec.suite_task_name,
        "suite_task_names": list(spec.suite_task_names),
        "suite_task_count": len(spec.suite_task_names),
        "suite_config_snapshot_path": None if spec.suite_config_snapshot_path is None else str(spec.suite_config_snapshot_path),
        "suite_config_snapshot_path_relative": spec.suite_config_snapshot_path_relative,
        "suite_config_sha256": spec.suite_config_sha256,
        "model_path": str(spec.model_path),
        "evaluated_model_path": str(spec.model_path),
        "evaluated_model_source": spec.evaluated_model_source or "specified",
        "task_config_path_used_for_evaluation": str(spec.task_config_path),
        "task_shape_used_for_evaluation": spec.task_shape,
        "source_manifest_path": None if spec.source_manifest_path is None else str(spec.source_manifest_path),
        "action_interface": spec.action_interface,
        "rpm_delta_scale": spec.rpm_delta_scale if spec.action_interface == "direct_rpm" else None,
        "normalize_actions": spec.normalize_actions,
        "include_dynamics_observation": spec.include_dynamics_observation,
        "include_previous_action": spec.include_previous_action,
        "source_run_name": spec.source_run_name,
        "source_run_kind": spec.source_run_kind,
        "source_curriculum_kind": spec.source_curriculum_kind,
        "source_stage": spec.source_stage,
        "model_scope": spec.model_scope,
        "evaluation_dir": str(output_dir),
        "diagnostics_dir": str(diagnostics_dir),
        "traces_dir": str(traces_dir),
        "plots_dir": str(plots_dir),
        "renders_dir": str(renders_dir),
        "metrics_path": str(metrics_path),
        "manifest_path": str(manifest_path),
        "trace_path": None if trace_path is None else str(trace_path),
        "gif_path": None if gif_path is None else str(gif_path),
        "plot_paths": {} if xy_plot is None else {"trajectory_xy": str(xy_plot)},
        "plot_trace_scope": "disabled" if not artifacts.plots_enabled else "render_rollout",
        "plot_trace_step_count": 0 if not artifacts.plots_enabled else 1,
        "plot_trace_terminated": False,
        "plot_trace_truncated": False,
        "failure_report_path": str(diagnostics_dir / "failure_report.json"),
        "episode_summaries_path": str(diagnostics_dir / "episode_summaries.json"),
        "curriculum_feedback_path": str(diagnostics_dir / "curriculum_feedback.json"),
        "render_enabled": artifacts.render_enabled,
        "plots_enabled": artifacts.plots_enabled,
        "trace_enabled": artifacts.trace_enabled,
        "diagnostics_enabled": artifacts.diagnostics_enabled,
        "eval_steps": spec.eval_steps,
        "seed": spec.seed,
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
    }
    _write_json(metrics_path, payload)
    _write_json(manifest_path, payload)

    return policy_evaluation.PolicyEvaluationResult(
        label=spec.label,
        model_role=spec.model_role,
        model_path=str(spec.model_path),
        task_config_path=str(spec.task_config_path),
        task_shape=spec.task_shape,
        output_dir=str(output_dir),
        diagnostics_dir=str(diagnostics_dir),
        traces_dir=str(traces_dir),
        plots_dir=str(plots_dir),
        renders_dir=str(renders_dir),
        metrics_path=str(metrics_path),
        manifest_path=str(manifest_path),
        trace_path=None if trace_path is None else str(trace_path),
        gif_path=None if gif_path is None else str(gif_path),
        plot_paths={} if xy_plot is None else {"trajectory_xy": str(xy_plot)},
        plot_trace_scope="disabled" if not artifacts.plots_enabled else "render_rollout",
        failure_report_path=str(diagnostics_dir / "failure_report.json"),
        episode_summaries_path=str(diagnostics_dir / "episode_summaries.json"),
        curriculum_feedback_path=str(diagnostics_dir / "curriculum_feedback.json"),
        render_enabled=artifacts.render_enabled,
        plots_enabled=artifacts.plots_enabled,
        trace_enabled=artifacts.trace_enabled,
        metrics=payload,
    )


def test_curriculum_evaluation_cli_parser_exposes_suite_options() -> None:
    """Verify curriculum evaluation CLI exposes suite-oriented controls."""
    parser = cli_evaluate_curriculum.build_parser()
    args = parser.parse_args(
        [
            "--summary",
            "summary.json",
            "--suite",
            "suite.yaml",
            "--no-render",
            "--render-fps",
            "12",
            "--render-max-steps",
            "40",
            "--no-plots",
            "--no-traces",
            "--model-scope",
            "final-stage",
        ]
    )

    assert args.summary == Path("summary.json")
    assert args.suite == Path("suite.yaml")
    assert args.mode == "suite"
    assert not hasattr(args, "benchmark")
    assert not hasattr(args, "benchmark_config")
    assert args.no_render is True
    assert args.render_fps == 12
    assert args.render_max_steps == 40
    assert args.no_plots is True
    assert args.no_traces is True
    assert args.model_scope == "final-stage"

    default_args = parser.parse_args(["--summary", "summary.json"])
    assert default_args.suite is None


def _fake_scenario_evaluation(**kwargs: object) -> policy_evaluation.PolicyScenarioEvaluationResult:
    """Return a lightweight scenario aggregate and update the run index like the real helper."""
    run_root = Path(kwargs["run_root"])
    run_name = str(kwargs["run_name"])
    model_scope = str(kwargs["model_scope"])
    run_manifest_path = Path(kwargs["run_manifest_path"])
    metrics_path = run_root / "evaluations" / "scenarios" / "metrics" / f"{run_name}_scenarios_metrics.json"
    manifest_path = run_root / "evaluations" / "scenarios" / "manifests" / f"{run_name}_scenarios_manifest.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = {
        "run_type": "evaluation",
        "run_kind": "curriculum",
        "mode": "standard_scenario_evaluation",
        "evaluation_name": "scenarios",
        "scenario_labels": ["easy", "medium", "hard"],
        "entry_count": 3,
        "evaluated_models": [],
        "summary_metrics_path": str(metrics_path),
        "summary_manifest_path": str(manifest_path),
    }
    metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
    manifest_path.write_text(json.dumps(metrics), encoding="utf-8")
    policy_evaluation.update_run_evaluation_index(
        run_manifest_path,
        {
            "index_key": "standard_scenario_evaluation:scenarios",
            "run_name": run_name,
            "run_kind": "curriculum",
            "mode": "standard_scenario_evaluation",
            "evaluation_name": "scenarios",
            "model_scope": model_scope,
            "aggregate_metrics_path": str(metrics_path),
            "aggregate_metrics_path_relative": "evaluations/scenarios/metrics/" + metrics_path.name,
            "evaluation_manifest_path": str(manifest_path),
            "evaluation_manifest_path_relative": "evaluations/scenarios/manifests/" + manifest_path.name,
            "task_names": ["easy", "medium", "hard"],
            "evaluated_models": [],
        },
    )
    return policy_evaluation.PolicyScenarioEvaluationResult(str(metrics_path), str(manifest_path), metrics)


def test_curriculum_evaluation_cli_standard_profile_passes_model_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the no-suite CLI path passes explicit model-scope narrowing."""
    captured: dict[str, object] = {}

    def fake_standard_evaluation(**kwargs: object) -> curriculum_evaluation.CurriculumStandardEvaluationResult:
        captured.update(kwargs)
        return curriculum_evaluation.CurriculumStandardEvaluationResult(
            metrics_path="evaluation_summary.json",
            manifest_path="evaluation_summary.json",
            metrics={"profile_name": "standard"},
        )

    monkeypatch.setattr(curriculum_evaluation, "run_curriculum_standard_evaluation", fake_standard_evaluation)

    status = cli_evaluate_curriculum.main(
        [
            "--summary",
            "storage/runs/curriculum_manual_line_smoke_seed0/run_manifest.json",
            "--model-scope",
            "final-stage",
        ]
    )

    assert status == 0
    assert captured["model_scope"] == "final-stage"


def test_curriculum_evaluation_requires_suite_for_suite_mode(tmp_path: Path) -> None:
    """Verify suite mode requires a canonical --suite config."""
    summary_path = _curriculum_summary(tmp_path)

    with pytest.raises(ValueError, match="--suite is required for suite mode"):
        curriculum_evaluation.run_curriculum_evaluation(
            summary_path=summary_path,
            mode="suite",
            suite_path=None,
        )


def test_curriculum_evaluation_own_stage_creates_stage_indexed_dirs_and_summary_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify own-stage mode produces stage-owned own_task evaluation dirs."""
    summary_path = _curriculum_summary(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", _fake_policy_evaluation)

    result = curriculum_evaluation.run_curriculum_evaluation(summary_path=summary_path, mode="own-stage")

    payload = result.metrics
    summary_payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert Path(result.metrics_path) == summary_path.parent / "evaluation_summary.json"
    assert result.manifest_path == result.metrics_path
    assert payload["evaluation_mode"] == "own-stage"
    assert payload["evaluation_name"] == "own_task"
    assert payload["run_kind"] == "curriculum"
    assert payload["curriculum_kind"] == "manual"
    assert payload["curriculum_run_name"] == "curriculum_manual_line_smoke_seed0"
    assert payload["evaluation_suite_name"] is None
    assert payload["suite_task_count"] == 0
    assert payload["model_scope"] == "all-stages"
    assert payload["summary_metrics_path_relative"] == "evaluation_summary.json"
    assert payload["summary_manifest_path_relative"] == "evaluation_summary.json"
    assert len(payload["evaluated_models"]) == 2
    directories = [entry["evaluation_dir"].replace("\\", "/") for entry in payload["evaluated_models"]]
    assert directories[0].endswith("runs/curriculum_manual_line_smoke_seed0/stages/stage01_hover_stabilization/evaluations/own_task")
    assert directories[1].endswith("runs/curriculum_manual_line_smoke_seed0/stages/stage02_line/evaluations/own_task")
    assert payload["evaluated_models"][0]["is_final_stage"] is False
    assert payload["evaluated_models"][1]["is_final_stage"] is True
    assert payload["evaluated_models"][0]["suite_task_name"] is None
    assert not (summary_path.parent / "evaluations" / "own_task").exists()
    assert summary_payload["entry_count"] == 1
    assert summary_payload["evaluations"][0]["evaluation_name"] == "own_task"


def test_curriculum_evaluation_suite_final_stage_writes_stage_owned_artifacts_and_uses_suite_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify final-stage suite evaluation writes task artifacts under the final stage."""
    summary_path = _curriculum_summary(tmp_path)
    suite_path = _suite_config(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", _fake_policy_evaluation)

    result = curriculum_evaluation.run_curriculum_evaluation(
        summary_path=summary_path,
        suite_path=suite_path,
        model_scope="final-stage",
    )

    payload = result.metrics
    summary_payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    directories = [entry["evaluation_dir"].replace("\\", "/") for entry in payload["evaluated_models"]]
    assert Path(result.metrics_path) == summary_path.parent / "evaluation_summary.json"
    assert payload["evaluation_mode"] == "suite"
    assert payload["evaluation_name"] == "final_suite"
    assert payload["evaluation_suite_name"] == "final_suite"
    assert payload["suite_task_names"] == ["line_basic", "hover_basic"]
    assert payload["suite_task_count"] == 2
    assert payload["model_scope"] == "final-stage"
    assert payload["entry_count"] == 2
    assert all(entry["stage_index"] == 2 for entry in payload["evaluated_models"])
    assert all(entry["is_final_stage"] is True for entry in payload["evaluated_models"])
    assert directories[0].endswith("runs/curriculum_manual_line_smoke_seed0/stages/stage02_line/evaluations/final_suite/line_basic")
    assert directories[1].endswith("runs/curriculum_manual_line_smoke_seed0/stages/stage02_line/evaluations/final_suite/hover_basic")
    assert all("/stages/stage02_line/" in path for path in directories)
    root_evaluation = summary_path.parent / "evaluations" / "final_suite"
    assert not root_evaluation.exists()
    assert {entry["suite_task_name"] for entry in payload["evaluated_models"]} == {"line_basic", "hover_basic"}
    assert all(entry["eval_steps"] == 88 for entry in payload["evaluated_models"])
    assert all(entry["seed"] == 7 for entry in payload["evaluated_models"])
    assert all(entry["render_enabled"] is False for entry in payload["evaluated_models"])
    assert all(entry["plots_enabled"] is False for entry in payload["evaluated_models"])
    assert all(entry["trace_enabled"] is False for entry in payload["evaluated_models"])
    assert all(entry["evaluated_model_path"].endswith("stage02_model_best.zip") for entry in payload["evaluated_models"])
    assert all(entry["evaluated_model_source"] == "best" for entry in payload["evaluated_models"])
    copied_task_config = Path(payload["evaluated_models"][0]["task_config_path_used_for_evaluation"])
    assert copied_task_config.exists()
    assert copied_task_config.as_posix().endswith("config/evaluation_suites/final_suite/line_basic_task.yaml")
    assert payload["suite_config_snapshot_path_relative"] == "config/evaluation_suites/final_suite_eval_suite.yaml"
    assert Path(payload["suite_config_snapshot_path"]).exists()
    assert payload["summary_role"] == "derived_aggregate_link_summary"
    assert payload["detailed_stage_artifacts_duplicated_at_run_root"] is False
    assert payload["summary_metrics_path_relative"] == "evaluation_summary.json"
    assert payload["summary_manifest_path_relative"] == "evaluation_summary.json"
    assert payload["canonical_stage_evaluation_manifest_paths_relative"] == [
        "stages/stage02_line/evaluations/final_suite/line_basic/manifests/stage02_line_line_basic_manifest.json",
        "stages/stage02_line/evaluations/final_suite/hover_basic/manifests/stage02_line_hover_basic_manifest.json",
    ]
    assert summary_payload["entry_count"] == 1
    assert summary_payload["evaluations"][0]["evaluation_name"] == "final_suite"

    updated_manifest = json.loads(summary_path.read_text(encoding="utf-8"))
    assert updated_manifest["evaluation_index"]["path_relative"] == "evaluation_index.json"
    assert updated_manifest["evaluation_index"]["entry_count"] == 1
    index_entry = updated_manifest["evaluation_index"]["evaluations"][0]
    assert index_entry["evaluation_name"] == "final_suite"
    assert index_entry["suite_name"] == "final_suite"
    assert index_entry["model_scope"] == "final-stage"
    assert index_entry["aggregate_metrics_path_relative"] == "evaluation_summary.json"
    assert index_entry["aggregate_metrics_relative"] == "evaluation_summary.json"
    assert index_entry["evaluation_manifest_path_relative"] == "evaluation_summary.json"
    assert index_entry["evaluation_manifest_relative"] == "evaluation_summary.json"
    assert index_entry["suite_config_snapshot_path_relative"] == "config/evaluation_suites/final_suite_eval_suite.yaml"
    assert index_entry["suite_config_snapshot_relative"] == "config/evaluation_suites/final_suite_eval_suite.yaml"
    assert index_entry["task_names"] == ["line_basic", "hover_basic"]
    assert index_entry["model_path"].endswith("stage02_model_best.zip")
    assert index_entry["model_path_relative"] == "training/stage02_model_best.zip"
    assert index_entry["evaluated_model_source"] == "best"
    assert updated_manifest["final_stage"]["model_path"].endswith("stage02_model_best.zip")
    assert updated_manifest["final_stage"]["selected_model_source"] == "best"
    assert updated_manifest["final_stage"]["last_model_path"].endswith("stage02_model.zip")
    assert updated_manifest["final_stage"]["best_model_path"].endswith("stage02_model_best.zip")
    assert index_entry["canonical_stage_evaluation_manifest_paths_relative"] == [
        "stages/stage02_line/evaluations/final_suite/line_basic/manifests/stage02_line_line_basic_manifest.json",
        "stages/stage02_line/evaluations/final_suite/hover_basic/manifests/stage02_line_hover_basic_manifest.json",
    ]
    final_stage_evaluation = updated_manifest["final_stage_evaluation"]
    assert final_stage_evaluation["final_stage_index"] == 2
    assert final_stage_evaluation["final_stage_name"] == "line"
    assert final_stage_evaluation["final_model_path"].endswith("stage02_model_best.zip")
    assert final_stage_evaluation["final_model_path_relative"] == "training/stage02_model_best.zip"
    assert final_stage_evaluation["final_model_source"] == "best"
    assert final_stage_evaluation["evaluation_manifest_path_relative"] == "evaluation_summary.json"
    assert final_stage_evaluation["canonical_stage_evaluation_manifest_paths_relative"] == [
        "stages/stage02_line/evaluations/final_suite/line_basic/manifests/stage02_line_line_basic_manifest.json",
        "stages/stage02_line/evaluations/final_suite/hover_basic/manifests/stage02_line_hover_basic_manifest.json",
    ]


def test_curriculum_evaluation_stage_suite_scope_uses_stage_dirs_without_root_details(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all-stage suite evaluation uses only owning stage dirs for details."""
    summary_path = _curriculum_summary(tmp_path)
    suite_path = _suite_config(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", _fake_policy_evaluation)

    result = curriculum_evaluation.run_curriculum_evaluation(
        summary_path=summary_path,
        suite_path=suite_path,
        render=True,
        plots=True,
        traces=True,
    )

    payload = result.metrics
    directories = [entry["evaluation_dir"].replace("\\", "/") for entry in payload["evaluated_models"]]
    assert payload["model_scope"] == "all-stages"
    assert payload["entry_count"] == 4
    assert any(
        path.endswith("runs/curriculum_manual_line_smoke_seed0/stages/stage01_hover_stabilization/evaluations/final_suite/line_basic")
        for path in directories
    )
    assert any(
        path.endswith("runs/curriculum_manual_line_smoke_seed0/stages/stage02_line/evaluations/final_suite/hover_basic") for path in directories
    )
    assert all("/stages/" in path for path in directories)
    assert not (summary_path.parent / "evaluations" / "final_suite").exists()
    assert payload["contains_convenience_baseline"] is False
    assert payload["baseline_ownership"] is None
    assert all(entry["render_enabled"] is True for entry in payload["evaluated_models"])
    assert all(entry["plots_enabled"] is True for entry in payload["evaluated_models"])
    assert all(entry["trace_enabled"] is True for entry in payload["evaluated_models"])


def test_curriculum_evaluation_rejects_convenience_baseline(tmp_path: Path) -> None:
    """Verify curriculum runs do not create root-owned baseline detail artifacts."""
    summary_path = _curriculum_summary(tmp_path)
    suite_path = _suite_config(tmp_path)
    baseline = tmp_path / "baseline_model.zip"
    baseline.write_bytes(b"model")

    with pytest.raises(ValueError, match="evaluate direct PPO separately"):
        curriculum_evaluation.run_curriculum_evaluation(
            summary_path=summary_path,
            suite_path=suite_path,
            include_baseline_model=baseline,
            baseline_label="ppo_line",
        )


def test_curriculum_evaluation_final_stage_scope_only_evaluates_final_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify --model-scope final-stage restricts evaluation to the final stage."""
    summary_path = _curriculum_summary(tmp_path)
    suite_path = _one_task_suite(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", _fake_policy_evaluation)

    result = curriculum_evaluation.run_curriculum_evaluation(
        summary_path=summary_path,
        suite_path=suite_path,
        model_scope="final-stage",
    )

    payload = result.metrics

    assert payload["model_scope"] == "final-stage"
    assert [entry["stage_index"] for entry in payload["evaluated_models"]] == [2]
    assert payload["evaluated_models"][0]["is_final_stage"] is True
    assert payload["evaluated_models"][0]["suite_task_name"] == "line_basic"
    assert payload["evaluated_models"][0]["evaluated_model_path"].endswith("stage02_model_best.zip")
    assert payload["evaluated_models"][0]["evaluated_model_source"] == "best"
    assert (
        payload["evaluated_models"][0]["evaluation_dir"]
        .replace("\\", "/")
        .endswith("runs/curriculum_manual_line_smoke_seed0/stages/stage02_line/evaluations/line_suite/line_basic")
    )


def test_curriculum_final_stage_suite_uses_final_stage_manifest_env_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify final-stage suite evaluation preserves action and observation flags from the stage manifest."""
    summary_path = _curriculum_summary(tmp_path)
    suite_path = _one_task_suite(tmp_path)
    captured_specs: list[policy_evaluation.PolicyEvaluationSpec] = []

    def fake_policy_evaluation(
        spec: policy_evaluation.PolicyEvaluationSpec,
        artifacts: policy_evaluation.PolicyEvaluationArtifactOptions,
    ) -> policy_evaluation.PolicyEvaluationResult:
        captured_specs.append(spec)
        return _fake_policy_evaluation(spec, artifacts)

    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", fake_policy_evaluation)

    result = curriculum_evaluation.run_curriculum_evaluation(
        summary_path=summary_path,
        suite_path=suite_path,
        model_scope="final-stage",
    )

    final_manifest_path = tmp_path / "training" / "stage02_manifest.json"
    evaluated_model = result.metrics["evaluated_models"][0]
    assert len(captured_specs) == 1
    assert captured_specs[0].source_manifest_path == final_manifest_path
    assert captured_specs[0].model_path == tmp_path / "training" / "stage02_model_best.zip"
    assert captured_specs[0].evaluated_model_source == "best"
    assert captured_specs[0].action_interface == "direct_rpm"
    assert captured_specs[0].rpm_delta_scale == 0.07
    assert captured_specs[0].include_dynamics_observation is True
    assert captured_specs[0].include_previous_action is True
    assert captured_specs[0].normalize_actions is True
    assert evaluated_model["source_manifest_path"] == str(final_manifest_path)
    assert evaluated_model["evaluated_model_path"].endswith("stage02_model_best.zip")
    assert evaluated_model["evaluated_model_source"] == "best"
    assert evaluated_model["action_interface"] == "direct_rpm"
    assert evaluated_model["rpm_delta_scale"] == 0.07
    assert evaluated_model["include_dynamics_observation"] is True
    assert evaluated_model["include_previous_action"] is True


def test_llm_curriculum_final_stage_suite_uses_final_stage_manifest_env_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify LLM curriculum final-stage evaluation uses the final-stage manifest flags too."""
    summary_path = _curriculum_summary(tmp_path)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))
    payload["curriculum_kind"] = "llm"
    payload["curriculum_name"] = "curriculum_llm_line_smoke"
    payload["run_name"] = "curriculum_llm_line_smoke_seed0"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suite_path = _one_task_suite(tmp_path)
    captured_specs: list[policy_evaluation.PolicyEvaluationSpec] = []

    def fake_policy_evaluation(
        spec: policy_evaluation.PolicyEvaluationSpec,
        artifacts: policy_evaluation.PolicyEvaluationArtifactOptions,
    ) -> policy_evaluation.PolicyEvaluationResult:
        captured_specs.append(spec)
        return _fake_policy_evaluation(spec, artifacts)

    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", fake_policy_evaluation)

    result = curriculum_evaluation.run_curriculum_evaluation(
        summary_path=summary_path,
        suite_path=suite_path,
        model_scope="final-stage",
    )

    evaluated_model = result.metrics["evaluated_models"][0]
    assert len(captured_specs) == 1
    assert captured_specs[0].action_interface == "direct_rpm"
    assert captured_specs[0].include_dynamics_observation is True
    assert captured_specs[0].include_previous_action is True
    assert evaluated_model["source_curriculum_kind"] == "llm"
    assert evaluated_model["action_interface"] == "direct_rpm"
    assert evaluated_model["include_dynamics_observation"] is True
    assert evaluated_model["include_previous_action"] is True


def test_curriculum_standard_evaluation_runs_stage_owned_default_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify no-suite curriculum evaluation runs every default evaluation for every stage."""
    summary_path = _curriculum_summary(tmp_path)
    generalization_suite = _suite_config(tmp_path, evaluation_name="generalization")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(curriculum_evaluation, "STANDARD_GENERALIZATION_SUITE_PATH", generalization_suite)
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", _fake_policy_evaluation)
    monkeypatch.setattr(policy_evaluation, "run_standard_scenario_evaluations", _fake_scenario_evaluation)

    result = curriculum_evaluation.run_curriculum_standard_evaluation(
        summary_path=summary_path,
        render=False,
        plots=False,
        traces=False,
    )

    summary_payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    updated_manifest = json.loads(summary_path.read_text(encoding="utf-8"))
    entries = summary_payload["evaluations"]
    assert result.metrics["profile_name"] == "standard"
    assert result.metrics["evaluation_names"] == ["own_task", "generalization", "scenarios"]
    assert summary_payload["entry_count"] == 3
    assert [(entry["evaluation_name"], entry["model_scope"], entry["entry_count"]) for entry in entries] == [
        ("own_task", "all-stages", 2),
        ("generalization", "all-stages", 4),
        ("scenarios", "final-stage", 3),
    ]
    assert updated_manifest["evaluation_index"]["path_relative"] == "evaluation_index.json"
    assert updated_manifest["evaluation_index"]["entry_count"] == 3
    assert [(entry["evaluation_name"], entry.get("model_scope")) for entry in updated_manifest["evaluation_index"]["evaluations"]] == [
        ("own_task", "all-stages"),
        ("generalization", "all-stages"),
        ("scenarios", "final-stage"),
    ]
    for evaluation_name in ("own_task", "generalization"):
        evaluation_dirs = [
            model["evaluation_dir_relative"].replace("\\", "/")
            for entry in entries
            if entry["evaluation_name"] == evaluation_name
            for model in entry["evaluated_models"]
        ]
        assert any(path.startswith(f"stages/stage01_hover_stabilization/evaluations/{evaluation_name}") for path in evaluation_dirs)
        assert any(path.startswith(f"stages/stage02_line/evaluations/{evaluation_name}") for path in evaluation_dirs)
    assert (summary_path.parent / "evaluations" / "scenarios").exists()


def test_curriculum_standard_evaluation_final_stage_scope_narrows_every_default_evaluation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify explicit final-stage model scope narrows the whole standard profile."""
    summary_path = _curriculum_summary(tmp_path)
    generalization_suite = _suite_config(tmp_path, evaluation_name="generalization")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(curriculum_evaluation, "STANDARD_GENERALIZATION_SUITE_PATH", generalization_suite)
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", _fake_policy_evaluation)
    monkeypatch.setattr(policy_evaluation, "run_standard_scenario_evaluations", _fake_scenario_evaluation)

    result = curriculum_evaluation.run_curriculum_standard_evaluation(
        summary_path=summary_path,
        model_scope="final-stage",
        render=False,
        plots=False,
        traces=False,
    )

    summary_payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    updated_manifest = json.loads(summary_path.read_text(encoding="utf-8"))
    assert [(entry["evaluation_name"], entry["model_scope"], entry["entry_count"]) for entry in summary_payload["evaluations"]] == [
        ("own_task", "final-stage", 1),
        ("generalization", "final-stage", 2),
        ("scenarios", "final-stage", 3),
    ]
    assert [(entry["evaluation_name"], entry.get("model_scope")) for entry in updated_manifest["evaluation_index"]["evaluations"]] == [
        ("own_task", "final-stage"),
        ("generalization", "final-stage"),
        ("scenarios", "final-stage"),
    ]
    stage_models = [model for entry in summary_payload["evaluations"] for model in entry["evaluated_models"] if "stage_index" in model]
    assert all(model["stage_index"] == 2 for model in stage_models)
    assert (summary_path.parent / "evaluations" / "scenarios").exists()


def test_curriculum_explicit_suite_runs_only_requested_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify explicit --suite line_eval avoids the full standard profile."""
    summary_path = _curriculum_summary(tmp_path)
    suite_path = _suite_config(tmp_path, evaluation_name="line_eval")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", _fake_policy_evaluation)

    result = curriculum_evaluation.run_curriculum_evaluation(
        summary_path=summary_path,
        suite_path=suite_path,
        render=False,
        plots=False,
        traces=False,
    )

    summary_payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    updated_manifest = json.loads(summary_path.read_text(encoding="utf-8"))
    evaluation_dirs = sorted(
        path.relative_to(summary_path.parent).as_posix() for path in (summary_path.parent / "stages").glob("*/evaluations/*") if path.is_dir()
    )
    assert summary_payload["entry_count"] == 1
    assert summary_payload["evaluations"][0]["evaluation_name"] == "line_eval"
    assert updated_manifest["evaluation_index"]["entry_count"] == 1
    assert updated_manifest["evaluation_index"]["evaluations"][0]["evaluation_name"] == "line_eval"
    assert evaluation_dirs == [
        "stages/stage01_hover_stabilization/evaluations/line_eval",
        "stages/stage02_line/evaluations/line_eval",
    ]
    assert not (summary_path.parent / "evaluations").exists()


def test_removed_benchmark_config_is_not_required() -> None:
    """Verify the removed benchmark config path is absent from active curriculum evaluation."""
    benchmark_kind = "benchmarks"
    removed_loader_name = "load_" + f"curriculum_{benchmark_kind}"
    removed_benchmark_name = f"curriculum_{benchmark_kind}.yaml"

    assert not hasattr(curriculum_evaluation, "DEFAULT_BENCHMARK_CONFIG_PATH")
    assert not hasattr(curriculum_evaluation, removed_loader_name)
    removed_benchmark_path = Path("configs") / "evaluation" / removed_benchmark_name
    assert not removed_benchmark_path.exists()


def test_curriculum_evaluation_supported_modes_do_not_expose_progression() -> None:
    """Verify progression mode is intentionally unsupported."""
    assert set(curriculum_evaluation.SUPPORTED_EVALUATION_MODES) == {"own-stage", "suite"}
