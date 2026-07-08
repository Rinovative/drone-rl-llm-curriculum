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
        },
    )

    summary_path = tmp_path / "summary.json"
    _write_json(
        summary_path,
        {
            "curriculum_name": "manual_line_v1",
            "run_name": "curriculum_manual_line_v1_seed0",
            "run_kind": "curriculum",
            "curriculum_kind": "manual",
            "seed": 0,
            "final_model_path": str(stage_root / "stage02_model.zip"),
            "stages": [
                {
                    "stage_index": 1,
                    "stage_name": "hover_stabilization",
                    "task_shape": "hover_stabilization",
                    "run_name": "manual_line_v1_stage01_hover_stabilization_seed0",
                    "model_path": str(stage_root / "stage01_model.zip"),
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
                    "run_name": "manual_line_v1_stage02_line_seed0",
                    "model_path": str(stage_root / "stage02_model.zip"),
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
        "model_path": str(spec.model_path),
        "task_config_path_used_for_evaluation": str(spec.task_config_path),
        "task_shape_used_for_evaluation": spec.task_shape,
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
    """Verify own-stage mode still produces stage-indexed evaluation dirs."""
    summary_path = _curriculum_summary(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", _fake_policy_evaluation)

    result = curriculum_evaluation.run_curriculum_evaluation(summary_path=summary_path, mode="own-stage")

    payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert payload["evaluation_mode"] == "own-stage"
    assert payload["run_kind"] == "curriculum"
    assert payload["curriculum_kind"] == "manual"
    assert payload["curriculum_run_name"] == "curriculum_manual_line_v1_seed0"
    assert payload["evaluation_suite_name"] is None
    assert payload["suite_task_count"] == 0
    assert payload["model_scope"] == "all-stages"
    assert len(payload["evaluated_models"]) == 2
    directories = [entry["evaluation_dir"].replace("\\", "/") for entry in payload["evaluated_models"]]
    assert directories[0].endswith("runs/curriculum_manual_line_v1_seed0/stages/stage01_hover_stabilization/evaluations/own_stage")
    assert directories[1].endswith("runs/curriculum_manual_line_v1_seed0/stages/stage02_line/evaluations/own_stage")
    assert payload["evaluated_models"][0]["is_final_stage"] is False
    assert payload["evaluated_models"][1]["is_final_stage"] is True
    assert payload["evaluated_models"][0]["suite_task_name"] is None


def test_curriculum_evaluation_suite_final_stage_writes_run_level_and_uses_suite_options(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify final-stage suite evaluation writes run-level task artifacts."""
    summary_path = _curriculum_summary(tmp_path)
    suite_path = _suite_config(tmp_path)
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", _fake_policy_evaluation)

    result = curriculum_evaluation.run_curriculum_evaluation(
        summary_path=summary_path,
        suite_path=suite_path,
        model_scope="final-stage",
    )

    payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    directories = [entry["evaluation_dir"].replace("\\", "/") for entry in payload["evaluated_models"]]
    assert payload["evaluation_mode"] == "suite"
    assert payload["evaluation_name"] == "final_suite"
    assert payload["evaluation_suite_name"] == "final_suite"
    assert payload["suite_task_names"] == ["line_basic", "hover_basic"]
    assert payload["suite_task_count"] == 2
    assert payload["model_scope"] == "final-stage"
    assert payload["entry_count"] == 2
    assert all(entry["stage_index"] == 2 for entry in payload["evaluated_models"])
    assert all(entry["is_final_stage"] is True for entry in payload["evaluated_models"])
    assert directories[0].endswith("runs/curriculum_manual_line_v1_seed0/evaluations/final_suite/line_basic")
    assert directories[1].endswith("runs/curriculum_manual_line_v1_seed0/evaluations/final_suite/hover_basic")
    assert not any("/stages/" in path for path in directories)
    assert {entry["suite_task_name"] for entry in payload["evaluated_models"]} == {"line_basic", "hover_basic"}
    assert all(entry["eval_steps"] == 88 for entry in payload["evaluated_models"])
    assert all(entry["seed"] == 7 for entry in payload["evaluated_models"])
    assert all(entry["render_enabled"] is False for entry in payload["evaluated_models"])
    assert all(entry["plots_enabled"] is False for entry in payload["evaluated_models"])
    assert all(entry["trace_enabled"] is False for entry in payload["evaluated_models"])
    copied_task_config = Path(payload["evaluated_models"][0]["task_config_path_used_for_evaluation"])
    assert copied_task_config.exists()
    assert copied_task_config.as_posix().endswith("config/evaluation_suites/final_suite/line_basic_task.yaml")


def test_curriculum_evaluation_stage_suite_scope_uses_stage_dirs_and_baseline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify all-stage suite evaluation uses stage dirs and optional baseline dirs."""
    summary_path = _curriculum_summary(tmp_path)
    suite_path = _suite_config(tmp_path)
    baseline = tmp_path / "baseline_model.zip"
    baseline.write_bytes(b"model")
    monkeypatch.setenv("STORAGE_ROOT", str(tmp_path / "storage"))
    monkeypatch.setattr(policy_evaluation, "run_policy_evaluation", _fake_policy_evaluation)

    result = curriculum_evaluation.run_curriculum_evaluation(
        summary_path=summary_path,
        suite_path=suite_path,
        include_baseline_model=baseline,
        baseline_label="ppo_line",
        render=True,
        plots=True,
        traces=True,
    )

    payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    directories = [entry["evaluation_dir"].replace("\\", "/") for entry in payload["evaluated_models"]]
    assert payload["model_scope"] == "all-stages"
    assert payload["entry_count"] == 6
    assert any(
        path.endswith("runs/curriculum_manual_line_v1_seed0/stages/stage01_hover_stabilization/evaluations/final_suite/line_basic")
        for path in directories
    )
    assert any(path.endswith("runs/curriculum_manual_line_v1_seed0/stages/stage02_line/evaluations/final_suite/hover_basic") for path in directories)
    assert any(
        path.endswith("runs/curriculum_manual_line_v1_seed0/evaluations/final_suite/baselines/baseline_ppo_line/line_basic") for path in directories
    )
    assert all(entry["render_enabled"] is True for entry in payload["evaluated_models"])
    assert all(entry["plots_enabled"] is True for entry in payload["evaluated_models"])
    assert all(entry["trace_enabled"] is True for entry in payload["evaluated_models"])


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

    payload = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))

    assert payload["model_scope"] == "final-stage"
    assert [entry["stage_index"] for entry in payload["evaluated_models"]] == [2]
    assert payload["evaluated_models"][0]["is_final_stage"] is True
    assert payload["evaluated_models"][0]["suite_task_name"] == "line_basic"


def test_old_curriculum_benchmarks_config_is_not_required() -> None:
    """Verify the old benchmark config path is absent from active curriculum evaluation."""
    assert not hasattr(curriculum_evaluation, "DEFAULT_BENCHMARK_CONFIG_PATH")
    assert not hasattr(curriculum_evaluation, "load_curriculum_benchmarks")
    assert not Path("configs/evaluation/curriculum_benchmarks.yaml").exists()


def test_curriculum_evaluation_supported_modes_do_not_expose_progression() -> None:
    """Verify progression mode is intentionally unsupported."""
    assert set(curriculum_evaluation.SUPPORTED_EVALUATION_MODES) == {"own-stage", "suite"}
