"""Tests for the shared policy evaluation helper."""

# ruff: noqa: S101, PT018, ARG005

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import yaml

from src.experiments.cli import experiments_cli_evaluate_policy as cli_evaluate_policy
from src.experiments.evaluation import experiments_evaluation_policy as policy_evaluation
from src.experiments.evaluation import experiments_evaluation_suites as evaluation_suites

FULL_EVALUATION_EPISODE_COUNT = 2
SUITE_EVALUATION_STEPS = 120
OWN_TASK_EVAL_STEPS = 12
OWN_TASK_SEED = 4
EVAL_RPM_DELTA_SCALE = 0.07
STANDARD_SCENARIO_LABELS = ("easy", "medium", "hard")
STANDARD_SCENARIO_COUNT = len(STANDARD_SCENARIO_LABELS)
STANDARD_SCENARIO_START_HOLD_SEC = 2.5
OBSERVATION_MISMATCH_MESSAGE = "Observation spaces do not match: model != env"


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


def _fake_render_artifact(
    spec: policy_evaluation.PolicyEvaluationSpec,
    task: dict[str, Any],
    renders_dir: Path,
    render_steps: int,
    render_fps: int,
) -> policy_evaluation._RenderArtifactResult:
    """Write a tiny fake GIF for tests that exercise render-enabled defaults."""
    del spec, task, render_steps, render_fps
    renders_dir.mkdir(parents=True, exist_ok=True)
    gif_path = renders_dir / "scenario_rollout.gif"
    gif_path.write_bytes(b"GIF89a")
    return policy_evaluation._RenderArtifactResult(  # noqa: SLF001
        gif_path=gif_path,
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
                "action": [0.0, 0.0, 0.0, 0.0],
                "terminated": False,
                "truncated": False,
            }
        ],
    )


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
            trace_records=[
                {
                    "source": "render",
                    "step_index": 0,
                    "episode_index": 0,
                    "time_sec": 0.0,
                    "actual_position_xyz_m": [0.0, 0.0, 1.0],
                    "reference_position_xyz_m": [0.0, 0.0, 1.0],
                    "position_error_m": 0.0,
                    "action": [0.0, 0.0, 0.0, 0.0],
                    "terminated": False,
                    "truncated": False,
                }
            ],
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


def test_shared_policy_evaluation_accepts_task_config_derived_from_suite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify a suite task can feed the existing one-task policy evaluation contract."""
    suite_path = tmp_path / "suite.yaml"
    suite_path.write_text(
        """evaluation_name: line_suite
seed: 0
eval_steps: 120
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
    suite = evaluation_suites.load_evaluation_suite(suite_path)
    suite_task = suite.get_task("line_basic")
    task_config = tmp_path / "line_basic_task.yaml"
    task_config.write_text(yaml.safe_dump(suite_task.to_task_config_dict(), sort_keys=False), encoding="utf-8")
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"model")

    def fake_collect(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        diagnostics_dir: Path,
    ) -> tuple[_FakeDiagnostics, dict[str, Any]]:
        assert spec.task_shape == "line"
        assert task["shape"] == "line"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        trace = diagnostics_dir / "evaluation_trace.jsonl"
        trace.write_text("{}\n", encoding="utf-8")
        return _FakeDiagnostics(metrics={}, trace_records=[]), {"evaluation_trace_path": str(trace)}

    monkeypatch.setattr(policy_evaluation, "_collect_diagnostics", fake_collect)

    result = policy_evaluation.run_policy_evaluation(
        policy_evaluation.PolicyEvaluationSpec(
            label="line_basic",
            model_role="suite_task",
            model_path=model_path,
            task_config_path=task_config,
            task_shape=suite_task.task_shape,
            output_dir=tmp_path / "out",
            eval_steps=suite.eval_steps,
            seed=suite.seed,
        ),
        policy_evaluation.PolicyEvaluationArtifactOptions(render_enabled=False, plots_enabled=False, trace_enabled=False),
    )

    assert result.task_shape == "line"
    assert result.metrics["task_config_path_used_for_evaluation"] == str(task_config)
    assert result.metrics["eval_steps"] == SUITE_EVALUATION_STEPS


def test_direct_policy_evaluation_cli_parser_accepts_run_manifest_and_suite() -> None:
    """Verify the direct policy suite CLI exposes the canonical run-owned inputs."""
    parser = cli_evaluate_policy.build_parser()
    args = parser.parse_args(
        [
            "--run-manifest",
            "storage/runs/direct_ppo_line_seed0/run_manifest.json",
            "--suite",
            "configs/evaluation/generalization_eval_suite.yaml",
            "--wandb-mode",
            "disabled",
        ]
    )

    assert args.run_manifest == Path("storage/runs/direct_ppo_line_seed0/run_manifest.json")
    assert args.suite == Path("configs/evaluation/generalization_eval_suite.yaml")
    assert args.wandb_mode == "disabled"

    default_args = parser.parse_args(["--run-manifest", "storage/runs/direct_ppo_line_seed0/run_manifest.json"])
    assert default_args.suite is None


def test_direct_policy_suite_evaluation_writes_under_direct_run_and_updates_index(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify direct PPO suite evaluation owns artifacts under the direct run."""
    storage_root = tmp_path / "storage"
    run_name = "direct_ppo_line_seed0"
    run_root = storage_root / "runs" / run_name
    model_path = run_root / "training" / "models" / f"{run_name}.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "run_kind": "direct_ppo",
                "total_timesteps": 10,
                "normalize_actions": True,
                "training": {
                    "model_path": str(model_path),
                    "model_path_relative": "training/models/direct_ppo_line_seed0.zip",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    suite_path = tmp_path / "line_eval_suite.yaml"
    suite_path.write_text(
        """evaluation_name: line_eval
seed: 3
eval_steps: 11
render:
  enabled: false
  fps: 20
  max_steps: null
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
""",
        encoding="utf-8",
    )

    def fake_collect(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        diagnostics_dir: Path,
    ) -> tuple[_FakeDiagnostics, dict[str, Any]]:
        assert spec.evaluation_name == "line_eval"
        assert spec.evaluation_suite_name == "line_eval"
        assert spec.suite_task_name == "line_basic"
        assert spec.suite_task_names == ("line_basic",)
        assert spec.suite_config_snapshot_path == run_root / "config" / "evaluation_suites" / "line_eval_eval_suite.yaml"
        assert spec.suite_config_snapshot_path_relative == "config/evaluation_suites/line_eval_eval_suite.yaml"
        assert spec.task_config_path == run_root / "config" / "evaluation_suites" / "line_eval" / "line_basic_task.yaml"
        assert task["shape"] == "line"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        trace_path = diagnostics_dir / "evaluation_trace.jsonl"
        trace_path.write_text("{}\n", encoding="utf-8")
        return _FakeDiagnostics(metrics={"episode_count": 1}, trace_records=[]), {
            "evaluation_trace_path": str(trace_path),
            "failure_report_path": str(diagnostics_dir / "failure_report.json"),
            "episode_summaries_path": str(diagnostics_dir / "episode_summaries.json"),
            "curriculum_feedback_path": str(diagnostics_dir / "curriculum_feedback.json"),
        }

    monkeypatch.setattr(policy_evaluation, "_collect_diagnostics", fake_collect)

    result = policy_evaluation.run_direct_policy_suite_evaluation(
        run_manifest_path=run_manifest_path,
        suite_path=suite_path,
        wandb_mode="disabled",
    )

    evaluation_root = run_root / "evaluations" / "line_eval"
    assert Path(result.metrics_path) == evaluation_root / "metrics" / "direct_ppo_line_seed0_line_eval_metrics.json"
    assert Path(result.manifest_path) == evaluation_root / "manifests" / "direct_ppo_line_seed0_line_eval_manifest.json"
    assert (run_root / "config" / "evaluation_suites" / "line_eval_eval_suite.yaml").read_text(encoding="utf-8") == suite_path.read_text(
        encoding="utf-8"
    )
    assert (evaluation_root / "line_basic" / "manifests" / "direct_ppo_line_seed0_line_basic_manifest.json").exists()

    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    assert metrics["run_kind"] == "direct_ppo"
    assert metrics["mode"] == "direct_policy_suite_evaluation"
    assert metrics["evaluation_name"] == "line_eval"
    assert metrics["suite_config_snapshot_path_relative"] == "config/evaluation_suites/line_eval_eval_suite.yaml"
    assert metrics["model_path_relative"] == "training/models/direct_ppo_line_seed0.zip"
    assert metrics["summary_manifest_path_relative"] == "evaluations/line_eval/manifests/direct_ppo_line_seed0_line_eval_manifest.json"
    assert (
        metrics["evaluated_models"][0]["metrics_path_relative"]
        == "evaluations/line_eval/line_basic/metrics/direct_ppo_line_seed0_line_basic_metrics.json"
    )

    updated_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["evaluation_index"]["path_relative"] == "evaluation_index.json"
    assert updated_manifest["evaluation_index"]["entry_count"] == 1
    index_entry = updated_manifest["evaluation_index"]["evaluations"][0]
    assert index_entry["evaluation_name"] == "line_eval"
    assert index_entry["aggregate_metrics_path_relative"] == "evaluations/line_eval/metrics/direct_ppo_line_seed0_line_eval_metrics.json"
    assert index_entry["evaluation_manifest_path_relative"] == "evaluations/line_eval/manifests/direct_ppo_line_seed0_line_eval_manifest.json"
    assert index_entry["task_names"] == ["line_basic"]
    assert (run_root / "evaluation_index.json").exists()


def _write_suite_config(path: Path, evaluation_name: str, task_name: str = "line_basic") -> None:
    """Write a one-task line evaluation suite."""
    path.write_text(
        f"""evaluation_name: {evaluation_name}
seed: 3
eval_steps: 11
render:
  enabled: false
  fps: 20
  max_steps: null
plots:
  enabled: false
traces:
  enabled: false
tasks:
  - task_name: {task_name}
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


def _write_direct_run_manifest(run_root: Path, run_name: str) -> Path:
    """Write a direct PPO run manifest with a training task snapshot."""
    model_path = run_root / "training" / "models" / f"{run_name}.zip"
    task_snapshot = run_root / "config" / "task_config.yaml"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    task_snapshot.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    _write_task_config(task_snapshot)
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "run_name": run_name,
                "run_kind": "direct_ppo",
                "total_timesteps": 10,
                "eval_steps": OWN_TASK_EVAL_STEPS,
                "seed": OWN_TASK_SEED,
                "normalize_actions": True,
                "training": {
                    "model_path": str(model_path),
                    "model_path_relative": f"training/models/{run_name}.zip",
                },
                "config": {
                    "task_config_snapshot_path": str(task_snapshot),
                    "task_config_snapshot_path_relative": "config/task_config.yaml",
                    "task_index": 0,
                    "task_shape": "line",
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return run_manifest_path


def test_direct_policy_suite_evaluation_uses_manifest_env_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify direct PPO suite evaluation preserves env flags from the training manifest."""
    run_name = "direct_ppo_line_seed0"
    run_root = tmp_path / "storage" / "runs" / run_name
    run_manifest_path = _write_direct_run_manifest(run_root, run_name)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    last_model_path = run_root / "training" / "models" / f"{run_name}_last.zip"
    last_model_path.write_bytes(b"last-model")
    run_manifest.update(
        {
            "action_interface": "direct_rpm",
            "rpm_delta_scale": EVAL_RPM_DELTA_SCALE,
            "include_dynamics_observation": True,
            "include_previous_action": True,
            "normalize_actions": True,
        }
    )
    run_manifest["training"].update(
        {
            "last_model_path": str(last_model_path),
            "last_model_path_relative": f"training/models/{run_name}_last.zip",
        }
    )
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suite_path = tmp_path / "line_eval_suite.yaml"
    _write_suite_config(suite_path, "line_eval", "line_basic")
    captured_specs: list[policy_evaluation.PolicyEvaluationSpec] = []

    def fake_collect(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        diagnostics_dir: Path,
    ) -> tuple[_FakeDiagnostics, dict[str, Any]]:
        del task
        captured_specs.append(spec)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        trace_path = diagnostics_dir / "evaluation_trace.jsonl"
        trace_path.write_text("{}\n", encoding="utf-8")
        return _FakeDiagnostics(metrics={"episode_count": 1}, trace_records=[]), {"evaluation_trace_path": str(trace_path)}

    monkeypatch.setattr(policy_evaluation, "_collect_diagnostics", fake_collect)

    result = policy_evaluation.run_direct_policy_suite_evaluation(
        run_manifest_path=run_manifest_path,
        suite_path=suite_path,
        wandb_mode="disabled",
    )

    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    evaluated_model = metrics["evaluated_models"][0]
    assert len(captured_specs) == 1
    assert captured_specs[0].source_manifest_path == run_manifest_path
    assert captured_specs[0].model_path == last_model_path.resolve(strict=False)
    assert captured_specs[0].evaluated_model_source == "last"
    assert metrics["evaluated_model_path"] == str(last_model_path.resolve(strict=False))
    assert metrics["evaluated_model_source"] == "last"
    assert evaluated_model["evaluated_model_path"] == str(last_model_path.resolve(strict=False))
    assert evaluated_model["evaluated_model_source"] == "last"
    assert captured_specs[0].action_interface == "direct_rpm"
    assert captured_specs[0].rpm_delta_scale == EVAL_RPM_DELTA_SCALE
    assert captured_specs[0].include_dynamics_observation is True
    assert captured_specs[0].include_previous_action is True
    assert captured_specs[0].normalize_actions is True
    assert evaluated_model["source_manifest_path"] == str(run_manifest_path)
    assert evaluated_model["action_interface"] == "direct_rpm"
    assert evaluated_model["rpm_delta_scale"] == EVAL_RPM_DELTA_SCALE
    assert evaluated_model["include_dynamics_observation"] is True
    assert evaluated_model["include_previous_action"] is True


def test_direct_policy_suite_evaluation_prefers_best_model_from_manifest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify direct PPO suite evaluation chooses manifest best models before last models."""
    run_name = "direct_ppo_line_seed0"
    run_root = tmp_path / "storage" / "runs" / run_name
    run_manifest_path = _write_direct_run_manifest(run_root, run_name)
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    best_model_path = run_root / "training" / "models" / f"{run_name}_best.zip"
    last_model_path = run_root / "training" / "models" / f"{run_name}_last.zip"
    best_model_path.write_bytes(b"best-model")
    last_model_path.write_bytes(b"last-model")
    run_manifest["training"].update(
        {
            "best_model_path": str(best_model_path),
            "best_model_path_relative": f"training/models/{run_name}_best.zip",
            "last_model_path": str(last_model_path),
            "last_model_path_relative": f"training/models/{run_name}_last.zip",
        }
    )
    run_manifest_path.write_text(json.dumps(run_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    suite_path = tmp_path / "line_eval_suite.yaml"
    _write_suite_config(suite_path, "line_eval", "line_basic")
    captured_specs: list[policy_evaluation.PolicyEvaluationSpec] = []

    def fake_collect(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        diagnostics_dir: Path,
    ) -> tuple[_FakeDiagnostics, dict[str, Any]]:
        del task
        captured_specs.append(spec)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        trace_path = diagnostics_dir / "evaluation_trace.jsonl"
        trace_path.write_text("{}\n", encoding="utf-8")
        return _FakeDiagnostics(metrics={"episode_count": 1}, trace_records=[]), {"evaluation_trace_path": str(trace_path)}

    monkeypatch.setattr(policy_evaluation, "_collect_diagnostics", fake_collect)

    result = policy_evaluation.run_direct_policy_suite_evaluation(
        run_manifest_path=run_manifest_path,
        suite_path=suite_path,
        wandb_mode="disabled",
    )

    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result.manifest_path).read_text(encoding="utf-8"))
    evaluated_model = metrics["evaluated_models"][0]
    assert len(captured_specs) == 1
    assert captured_specs[0].model_path == best_model_path.resolve(strict=False)
    assert captured_specs[0].evaluated_model_source == "best"
    assert metrics["evaluated_model_path"] == str(best_model_path.resolve(strict=False))
    assert metrics["evaluated_model_path_relative"] == f"training/models/{run_name}_best.zip"
    assert metrics["evaluated_model_source"] == "best"
    assert manifest["evaluated_model_path"] == metrics["evaluated_model_path"]
    assert manifest["evaluated_model_source"] == "best"
    assert evaluated_model["evaluated_model_path"] == str(best_model_path.resolve(strict=False))
    assert evaluated_model["evaluated_model_source"] == "best"


def test_policy_evaluation_env_builder_applies_spec_action_and_observation_flags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify evaluation env construction uses manifest-derived action and observation settings."""
    calls: dict[str, Any] = {}

    class FakeEnv:
        observation_space = "Box(env, (22,), float32)"

    def fake_make_tracking_env(task: dict[str, Any], **kwargs: Any) -> FakeEnv:
        calls["task"] = task
        calls["make_kwargs"] = kwargs
        return FakeEnv()

    def fake_normalized_action_env(env: FakeEnv) -> tuple[str, FakeEnv]:
        calls["normalized_env"] = env
        return ("normalized", env)

    monkeypatch.setattr(policy_evaluation.envs.tracking_env, "make_trajectory_tracking_env", fake_make_tracking_env)
    monkeypatch.setattr(policy_evaluation.envs.tracking_env, "make_normalized_action_env", fake_normalized_action_env)
    spec = policy_evaluation.PolicyEvaluationSpec(
        label="line_basic",
        model_role="suite_task",
        model_path=tmp_path / "model.zip",
        task_config_path=tmp_path / "task.yaml",
        task_shape="line",
        output_dir=tmp_path / "evaluation",
        eval_steps=11,
        seed=3,
        normalize_actions=False,
        action_interface="direct_rpm",
        rpm_delta_scale=EVAL_RPM_DELTA_SCALE,
        include_dynamics_observation=True,
        include_previous_action=True,
    )
    task = {"task_type": "trajectory", "shape": "line"}

    env = policy_evaluation._make_policy_evaluation_env(spec, task, record=True, max_steps=17)  # noqa: SLF001

    assert env[0] == "normalized"
    assert calls["task"] == task
    assert calls["make_kwargs"] == {
        "gui": False,
        "record": True,
        "max_steps": 17,
        "action_interface": "direct_rpm",
        "rpm_delta_scale": EVAL_RPM_DELTA_SCALE,
        "include_dynamics_observation": True,
        "include_previous_action": True,
    }
    assert calls["normalized_env"] is env[1]


def test_policy_evaluation_observation_mismatch_error_includes_manifest_context(tmp_path: Path) -> None:
    """Verify env/model observation mismatches include enough context to fix config drift."""
    model_path = tmp_path / "model.zip"
    model_path.write_bytes(b"model")
    manifest_path = tmp_path / "run_manifest.json"
    spec = policy_evaluation.PolicyEvaluationSpec(
        label="line_basic",
        model_role="suite_task",
        model_path=model_path,
        task_config_path=tmp_path / "task.yaml",
        task_shape="line",
        output_dir=tmp_path / "evaluation",
        eval_steps=11,
        seed=3,
        action_interface="direct_rpm",
        include_dynamics_observation=True,
        include_previous_action=True,
        source_manifest_path=manifest_path,
    )

    class FakeModel:
        observation_space = "Box(model, (22,), float32)"

    class FakeEnv:
        observation_space = "Box(env, (10,), float32)"

    class FakePPO:
        @staticmethod
        def load(path: str, env: object | None = None, device: str = "cpu") -> FakeModel:
            del path, device
            if env is not None:
                raise ValueError(OBSERVATION_MISMATCH_MESSAGE)
            return FakeModel()

    with pytest.raises(ValueError, match="evaluation environment observation space") as exc_info:
        policy_evaluation._load_ppo_with_evaluation_env(FakePPO, spec=spec, tracking_env=FakeEnv())  # noqa: SLF001

    message = str(exc_info.value)
    assert "model_observation_space=Box(model, (22,), float32)" in message
    assert "env_observation_space=Box(env, (10,), float32)" in message
    assert f"manifest_path={manifest_path}" in message
    assert f"model_path={model_path}" in message
    assert "action_interface=direct_rpm" in message
    assert "include_dynamics_observation=True" in message
    assert "include_previous_action=True" in message


def test_direct_policy_own_task_evaluation_uses_training_task_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify direct own_task evaluation uses the recorded training task snapshot."""
    run_name = "direct_ppo_line_seed0"
    run_root = tmp_path / "storage" / "runs" / run_name
    run_manifest_path = _write_direct_run_manifest(run_root, run_name)

    def fake_collect(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        diagnostics_dir: Path,
    ) -> tuple[_FakeDiagnostics, dict[str, Any]]:
        assert spec.evaluation_name == "own_task"
        assert spec.suite_task_name == "own_task"
        assert spec.task_config_path == run_root / "config" / "task_config.yaml"
        assert spec.output_dir == run_root / "evaluations" / "own_task"
        assert spec.eval_steps == OWN_TASK_EVAL_STEPS
        assert spec.seed == OWN_TASK_SEED
        assert task["shape"] == "line"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        trace_path = diagnostics_dir / "evaluation_trace.jsonl"
        trace_path.write_text("{}\n", encoding="utf-8")
        return _FakeDiagnostics(metrics={"episode_count": 1}, trace_records=[]), {"evaluation_trace_path": str(trace_path)}

    monkeypatch.setattr(policy_evaluation, "_collect_diagnostics", fake_collect)
    monkeypatch.setattr(policy_evaluation, "_write_render_artifact", _fake_render_artifact)

    result = policy_evaluation.run_direct_policy_own_task_evaluation(run_manifest_path=run_manifest_path, wandb_mode="disabled")

    assert Path(result.metrics_path) == run_root / "evaluations" / "own_task" / "metrics" / "direct_ppo_line_seed0_own_task_metrics.json"
    updated_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["evaluation_index"]["path_relative"] == "evaluation_index.json"
    index_entry = updated_manifest["evaluation_index"]["evaluations"][0]
    assert index_entry["evaluation_name"] == "own_task"
    assert index_entry["suite_name"] is None
    assert index_entry["aggregate_metrics_relative"] == "evaluations/own_task/metrics/direct_ppo_line_seed0_own_task_metrics.json"
    assert index_entry["evaluation_manifest_relative"] == "evaluations/own_task/manifests/direct_ppo_line_seed0_own_task_manifest.json"
    assert (run_root / "evaluation_index.json").exists()


@pytest.mark.parametrize(
    ("distribution_path", "expected_shape", "expected_source", "expected_task_name", "expected_is_show"),
    [
        (
            "configs/tasks/task_distribution_basic_training_show.yaml",
            "basic_training_show",
            "task_distribution_base_task",
            None,
            True,
        ),
        (
            "configs/tasks/task_distribution_tracking_medium.yaml",
            "polyline",
            "task_distribution_own_task_representative",
            "tracking_medium_representative",
            False,
        ),
    ],
)
def test_direct_policy_own_task_evaluation_resolves_distribution_representative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    distribution_path: str,
    expected_shape: str,
    expected_source: str,
    expected_task_name: str | None,
    expected_is_show: bool,
) -> None:
    """Verify distribution-trained direct runs do not fall back to the selected line task."""
    run_name = f"direct_ppo_{expected_shape}_seed0"
    run_root = tmp_path / "storage" / "runs" / run_name
    run_manifest_path = _write_direct_run_manifest(run_root, run_name)
    manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    manifest["task_distribution_config_path"] = distribution_path
    manifest["config"]["task_distribution_config_path"] = distribution_path
    manifest["config"]["task_index"] = 0
    manifest["config"]["task_shape"] = "line"
    run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def fake_collect(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        diagnostics_dir: Path,
    ) -> tuple[_FakeDiagnostics, dict[str, Any]]:
        assert spec.evaluation_name == "own_task"
        assert spec.task_config_path.parent == run_root / "config" / "own_task"
        assert spec.own_task_source == expected_source
        assert spec.own_task_distribution_config_path == Path(distribution_path).resolve(strict=False)
        assert spec.own_task_shape == expected_shape
        assert spec.own_task_is_show is expected_is_show
        assert spec.own_task_fallback_used is False
        assert spec.own_task_fallback_reason is None
        assert task["shape"] == expected_shape
        if expected_task_name is not None:
            assert task["task_name"] == expected_task_name
        assert task["start_hold_enabled"] is True
        assert task["final_hold_enabled"] is True
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        trace_path = diagnostics_dir / "evaluation_trace.jsonl"
        trace_path.write_text("{}\n", encoding="utf-8")
        return _FakeDiagnostics(metrics={"episode_count": 1}, trace_records=[]), {"evaluation_trace_path": str(trace_path)}

    monkeypatch.setattr(policy_evaluation, "_collect_diagnostics", fake_collect)
    monkeypatch.setattr(policy_evaluation, "_write_render_artifact", _fake_render_artifact)

    result = policy_evaluation.run_direct_policy_own_task_evaluation(run_manifest_path=run_manifest_path, wandb_mode="disabled")
    metrics = json.loads(Path(result.metrics_path).read_text(encoding="utf-8"))

    assert metrics["own_task_source"] == expected_source
    assert metrics["own_task_shape"] == expected_shape
    assert metrics["own_task_is_show"] is expected_is_show
    assert metrics["own_task_fallback_used"] is False
    assert metrics["own_task_fallback_reason"] is None
    assert metrics["own_task_distribution_config_path"] == str(Path(distribution_path).resolve(strict=False))


def test_standard_scenario_evaluation_writes_easy_medium_hard_metrics_and_diagnostics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify standard scenarios run as evaluations with metrics and diagnostics artifacts."""
    run_root = tmp_path / "storage" / "runs" / "policy_run"
    model_path = run_root / "models" / "policy.zip"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model")
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest_path.write_text(
        json.dumps(
            {
                "run_name": "policy_run",
                "run_kind": "direct_ppo",
                "action_interface": "pid_position",
                "normalize_actions": True,
                "include_dynamics_observation": True,
                "include_previous_action": True,
                "config": {
                    "training_config_path": "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml",
                    "action_interface": "pid_position",
                    "normalize_actions": True,
                    "include_dynamics_observation": True,
                    "include_previous_action": True,
                },
            }
        ),
        encoding="utf-8",
    )
    captured: list[tuple[str, float, float, bool, bool, str, bool]] = []

    def fake_run_scenario_render(
        settings: policy_evaluation.scenario_render.ScenarioRenderSettings | None = None,
    ) -> policy_evaluation.scenario_render.ScenarioRenderResult:
        assert settings is not None
        assert settings.output_dir is not None
        captured.append(
            (
                settings.scenario_name or "",
                settings.start_hold_sec,
                settings.final_hold_sec,
                settings.include_dynamics_observation,
                settings.include_previous_action,
                settings.action_interface,
                settings.normalize_actions,
            )
        )
        renders_dir = settings.output_dir / "renders"
        manifests_dir = settings.output_dir / "manifests"
        traces_dir = settings.output_dir / "traces"
        plots_dir = settings.output_dir / "plots"
        for directory in (renders_dir, manifests_dir, traces_dir, plots_dir):
            directory.mkdir(parents=True, exist_ok=True)
        gif_path = renders_dir / "scenario.gif"
        trace_path = traces_dir / "scenario_rollout_trace.jsonl"
        plot_path = plots_dir / "position_error.png"
        manifest_path = manifests_dir / "scenario_manifest.json"
        gif_path.write_bytes(b"gif")
        trace_path.write_text("{}\n", encoding="utf-8")
        plot_path.write_bytes(b"plot")
        manifest_path.write_text(
            json.dumps({"gif_path": str(gif_path), "trace_path": str(trace_path), "plot_paths": {"position_error": str(plot_path)}}),
            encoding="utf-8",
        )
        return policy_evaluation.scenario_render.ScenarioRenderResult(
            gif_path=str(gif_path),
            manifest_path=str(manifest_path),
            metrics={
                "scenario_name": settings.scenario_name,
                "mean_position_error_m": 0.10,
                "final_position_error_m": 0.20,
                "max_position_error_m": 0.30,
                "terminated": True,
                "truncated": False,
                "termination_reason": "tracking_reference_complete",
                "completed_reference": True,
                "completed_reference_motion": True,
                "completed_phase_holds": True,
                "completed_final_hold": True,
                "ended_normally": True,
                "warnings": [],
            },
            warnings=(),
        )

    monkeypatch.setattr(policy_evaluation.scenario_render, "run_scenario_render", fake_run_scenario_render)

    result = policy_evaluation.run_standard_scenario_evaluations(
        run_root=run_root,
        run_name="policy_run",
        model_path=model_path,
        model_run_name="policy_run",
        source_run_kind="direct_ppo",
        source_curriculum_kind=None,
        model_scope="direct",
        evaluated_model_source="test",
        run_manifest_path=run_manifest_path,
    )

    assert result.metrics["scenario_labels"] == list(STANDARD_SCENARIO_LABELS)
    assert captured == [
        ("show_easy", STANDARD_SCENARIO_START_HOLD_SEC, 1.0, True, True, "pid_position", True),
        ("show_medium", STANDARD_SCENARIO_START_HOLD_SEC, 1.2, True, True, "pid_position", True),
        ("show_hard", STANDARD_SCENARIO_START_HOLD_SEC, 1.5, True, True, "pid_position", True),
    ]
    assert result.metrics["entry_count"] == STANDARD_SCENARIO_COUNT
    for scenario_label in STANDARD_SCENARIO_LABELS:
        scenario_root = run_root / "evaluations" / "scenarios" / scenario_label
        metrics_path = scenario_root / "metrics" / f"policy_run_{scenario_label}_scenario_metrics.json"
        diagnostics_path = scenario_root / "diagnostics" / f"policy_run_{scenario_label}_scenario_diagnostics.json"
        assert metrics_path.exists()
        assert diagnostics_path.exists()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        assert metrics["evaluation_name"] == "scenarios"
        assert metrics["scenario_label"] == scenario_label
        assert metrics["model_path"] == str(model_path)
        assert metrics["evaluated_model_source"] == "test"
        assert metrics["include_dynamics_observation"] is True
        assert metrics["include_previous_action"] is True
        assert metrics["normalize_actions"] is True
        assert metrics["action_interface"] == "pid_position"
        assert metrics["scenario_duration_sec"] > 0.0
        assert metrics["scenario_reference_path_length_m"] > 0.0
        assert metrics["scenario_reference_mean_speed_mps"] > 0.0
        assert metrics["scenario_phase_count"] == len(metrics["scenario_segments"])
        assert metrics["start_hold_enabled"] is True
        assert metrics["start_hold_sec"] == STANDARD_SCENARIO_START_HOLD_SEC
        assert metrics["final_hold_enabled"] is True
        assert metrics["final_hold_sec"] > 0.0
        assert metrics["failure_overall_status"] == "passed"
        assert diagnostics["failure_overall_status"] == "passed"
        assert diagnostics["failure_primary_mode"] == "none"
        manifest = json.loads((scenario_root / "manifests" / "scenario_manifest.json").read_text(encoding="utf-8"))
        assert manifest["scenario_complete"] is True
        assert manifest["policy_rollout_render_required"] is True
        assert manifest["render_source"] == "evaluated_policy_rollout_trace"
        assert manifest["scenario_completion_requirements"] == {
            "metrics": True,
            "diagnostics": True,
            "manifest": True,
            "trace": True,
            "plot": True,
            "render": True,
        }

    entries_by_label = {entry["scenario_label"]: entry for entry in result.metrics["evaluated_models"]}
    assert set(entries_by_label) == {"easy", "medium", "hard"}
    assert entries_by_label["easy"]["metrics_path_relative"] == "evaluations/scenarios/easy/metrics/policy_run_easy_scenario_metrics.json"
    assert entries_by_label["easy"]["diagnostics_path_relative"] == "evaluations/scenarios/easy/diagnostics/policy_run_easy_scenario_diagnostics.json"


def test_direct_policy_standard_evaluation_runs_default_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify no-suite direct evaluation runs own_task plus standard suites."""
    run_name = "direct_ppo_line_seed0"
    run_root = tmp_path / "storage" / "runs" / run_name
    run_manifest_path = _write_direct_run_manifest(run_root, run_name)
    generalization_suite = tmp_path / "generalization_eval_suite.yaml"
    _write_suite_config(generalization_suite, "generalization", "line_generalization")
    monkeypatch.setattr(policy_evaluation, "STANDARD_GENERALIZATION_SUITE_PATH", generalization_suite)

    seen: list[str | None] = []

    def fake_collect(
        spec: policy_evaluation.PolicyEvaluationSpec,
        task: dict[str, Any],
        diagnostics_dir: Path,
    ) -> tuple[_FakeDiagnostics, dict[str, Any]]:
        del task
        seen.append(spec.evaluation_name)
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        trace_path = diagnostics_dir / "evaluation_trace.jsonl"
        trace_path.write_text("{}\n", encoding="utf-8")
        return _FakeDiagnostics(metrics={"episode_count": 1}, trace_records=[]), {"evaluation_trace_path": str(trace_path)}

    def fake_scenarios(run_manifest_path: Path, wandb_mode: str = "disabled") -> policy_evaluation.PolicyScenarioEvaluationResult:
        assert wandb_mode == "disabled"
        seen.append("scenarios")
        metrics_path = run_root / "evaluations" / "scenarios" / "metrics" / f"{run_name}_scenarios_metrics.json"
        manifest_path = run_root / "evaluations" / "scenarios" / "manifests" / f"{run_name}_scenarios_manifest.json"
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        metrics = {"evaluation_name": "scenarios", "scenario_labels": ["easy", "medium", "hard"], "entry_count": 3}
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        manifest_path.write_text(json.dumps(metrics), encoding="utf-8")
        policy_evaluation.update_run_evaluation_index(
            run_manifest_path,
            {
                "index_key": "standard_scenario_evaluation:scenarios",
                "run_name": run_name,
                "run_kind": "direct_ppo",
                "mode": "standard_scenario_evaluation",
                "evaluation_name": "scenarios",
                "aggregate_metrics_path": str(metrics_path),
                "aggregate_metrics_path_relative": "evaluations/scenarios/metrics/direct_ppo_line_seed0_scenarios_metrics.json",
                "evaluation_manifest_path": str(manifest_path),
                "evaluation_manifest_path_relative": "evaluations/scenarios/manifests/direct_ppo_line_seed0_scenarios_manifest.json",
                "task_names": ["easy", "medium", "hard"],
                "evaluated_models": [],
            },
        )
        return policy_evaluation.PolicyScenarioEvaluationResult(str(metrics_path), str(manifest_path), metrics)

    monkeypatch.setattr(policy_evaluation, "_collect_diagnostics", fake_collect)
    monkeypatch.setattr(policy_evaluation, "_write_render_artifact", _fake_render_artifact)
    monkeypatch.setattr(policy_evaluation, "run_direct_policy_scenario_evaluation", fake_scenarios)

    result = policy_evaluation.run_direct_policy_standard_evaluation(run_manifest_path=run_manifest_path, wandb_mode="disabled")

    assert result.metrics_path == str(run_root / "evaluation_index.json")
    assert result.metrics["evaluation_names"] == ["own_task", "generalization", "scenarios"]
    assert seen == ["own_task", "generalization", "scenarios"]
    updated_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    assert updated_manifest["evaluation_index"]["path_relative"] == "evaluation_index.json"
    assert [entry["evaluation_name"] for entry in updated_manifest["evaluation_index"]["evaluations"]] == [
        "own_task",
        "generalization",
        "scenarios",
    ]
