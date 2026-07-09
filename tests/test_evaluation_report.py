"""Tests for final-report artifact and metrics helpers."""

# ruff: noqa: S101, PLR2004

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src import evaluation


def _matrix_rows() -> list[dict[str, object]]:
    """Return a small planned-run matrix fixture."""
    return [
        {
            "lane": 1,
            "experiment_id": "direct_ppo_pid_dynprev_m-taskdist_medium_seed0",
            "kind": "direct_ppo",
            "config_path": "configs/training/ppo_tracking_pid_dynprev_m-taskdist_medium.yaml",
            "expected_run_name": "direct_ppo_pid_dynprev_m-taskdist_medium_seed0",
            "unit_count": 1,
            "method": "Direct PPO",
            "action_interface": "pid_position",
            "training_target": "m-taskdist_medium",
            "ppo_variant": "default",
        },
        {
            "lane": 2,
            "experiment_id": "curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0",
            "kind": "manual_curriculum",
            "config_path": "configs/curricula/curriculum_directrpm_dynprev_m-taskdist_medium.yaml",
            "expected_run_name": "curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0",
            "unit_count": 5,
            "method": "Manual curriculum",
            "action_interface": "direct_rpm",
            "training_target": "manual_curriculum",
            "ppo_variant": "default",
        },
    ]


def _write_manifest(root: Path, run_name: str, payload: dict[str, object] | None = None) -> Path:
    """Write a compact run manifest fixture."""
    run_root = root / run_name
    run_root.mkdir(parents=True)
    manifest_path = run_root / evaluation.report.RUN_MANIFEST_FILENAME
    manifest = {"run_name": run_name, "run_kind": "direct_ppo", "action_interface": "pid_position"}
    if payload:
        manifest.update(payload)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path


def test_default_artifact_root_is_direct_storage_runs() -> None:
    """Verify the report helper defaults to direct run folders under /workspace/storage/runs."""
    assert evaluation.report.artifact_root() == evaluation.report.DEFAULT_ARTIFACT_ROOT
    assert str(evaluation.report.DEFAULT_ARTIFACT_ROOT) == "/workspace/storage/runs"


def test_real_experiment_matrix_loads_expected_submission_rows() -> None:
    """Verify the active planned experiment matrix still has the expected report shape."""
    rows = evaluation.report.load_experiment_matrix()

    assert len(rows) == 20
    assert {row["method"] for row in rows} == {"Direct PPO", "Manual curriculum", "LLM curriculum"}
    assert {row["action_interface"] for row in rows} == {"pid_position", "direct_rpm"}
    assert "direct_ppo_pid_dynprev_m-taskdist_medium_seed0" in evaluation.report.expected_run_names(rows)


def test_summarize_run_artifacts_reports_available_and_missing_runs(tmp_path: Path) -> None:
    """Verify planned run folders are matched directly under the artifact root."""
    rows = _matrix_rows()
    root = tmp_path / "runs"
    _write_manifest(root, str(rows[0]["expected_run_name"]))
    metrics_dir = root / str(rows[0]["expected_run_name"]) / "training" / "metrics"
    metrics_dir.mkdir(parents=True)
    (metrics_dir / "policy_metrics.json").write_text(json.dumps({"mean_position_error_tracking_m": 0.2}), encoding="utf-8")

    table = evaluation.report.summarize_run_artifacts(root=root, matrix_rows=rows)

    assert table[0]["artifact_status"] == "available"
    assert table[0]["metrics_file_count"] == 1
    assert table[1]["artifact_status"] == "missing_run_folder"
    assert table[1]["manifest_path"] is None


def test_load_metric_records_reads_selected_scalar_metrics(tmp_path: Path) -> None:
    """Verify metrics JSON loading keeps scalar report metrics without fabricating values."""
    root = tmp_path / "runs"
    metrics_dir = root / "direct_ppo_pid_dynprev_m-taskdist_medium_seed0" / "training" / "metrics"
    metrics_dir.mkdir(parents=True)
    metrics_path = metrics_dir / "policy_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "mean_position_error_tracking_m": 0.25,
                "success_rate": 0.75,
                "failure_modes": ["not_scalar"],
            }
        ),
        encoding="utf-8",
    )

    rows = evaluation.report.load_metric_records(
        root=root,
        metric_keys=("mean_position_error_tracking_m", "success_rate", "failure_modes", "missing_metric"),
    )

    assert rows == [
        {
            "run_name": "direct_ppo_pid_dynprev_m-taskdist_medium_seed0",
            "metric_file": "policy_metrics.json",
            "path": str(metrics_path),
            "path_relative_to_root": "direct_ppo_pid_dynprev_m-taskdist_medium_seed0/training/metrics/policy_metrics.json",
            "artifact_status": "available",
            "mean_position_error_tracking_m": 0.25,
            "success_rate": 0.75,
        }
    ]


def test_build_metric_comparison_table_adds_planned_metadata_and_sorts(tmp_path: Path) -> None:
    """Verify comparison rows combine available metrics with planned experiment labels."""
    rows = _matrix_rows()
    root = tmp_path / "runs"
    for run_name, value in (
        (str(rows[0]["expected_run_name"]), 0.4),
        (str(rows[1]["expected_run_name"]), 0.2),
    ):
        metrics_dir = root / run_name / "evaluations" / "own_task" / "metrics"
        metrics_dir.mkdir(parents=True)
        (metrics_dir / "rollout_metrics.json").write_text(
            json.dumps({"evaluation_name": "own_task", "mean_position_error_tracking_m": value}),
            encoding="utf-8",
        )

    table = evaluation.report.build_metric_comparison_table(root=root, matrix_rows=rows)

    assert [row["run_name"] for row in table] == [str(rows[1]["expected_run_name"]), str(rows[0]["expected_run_name"])]
    assert table[0]["method"] == "Manual curriculum"
    assert table[0]["action_interface"] == "direct_rpm"
    assert table[0]["mean_tracking_error"] == pytest.approx(0.2)


def test_find_media_artifacts_returns_direct_run_relative_paths(tmp_path: Path) -> None:
    """Verify media discovery reports relative paths below direct run folders."""
    root = tmp_path / "runs"
    media_dir = root / "direct_ppo_pid_dynprev_m-taskdist_medium_seed0" / "plots"
    media_dir.mkdir(parents=True)
    (media_dir / "trajectory_xy.png").write_bytes(b"png")
    (media_dir / "rollout.gif").write_bytes(b"gif")
    (media_dir / "notes.txt").write_text("ignore", encoding="utf-8")

    rows = evaluation.report.find_media_artifacts(root=root, max_items=10)

    assert [row["media_file"] for row in rows] == ["rollout.gif", "trajectory_xy.png"]
    assert all(str(row["path_relative_to_root"]).startswith("direct_ppo_pid") for row in rows)


def _write_metrics(root: Path, run_name: str, relative_dir: str, payload: dict[str, object]) -> Path:
    """Write a synthetic metrics JSON fixture below a run directory."""
    metrics_dir = root / run_name / relative_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{run_name}_{relative_dir.replace('/', '_')}_metrics.json"
    metrics_path.write_text(json.dumps(payload), encoding="utf-8")
    return metrics_path


def test_build_report_metric_table_normalizes_valid_metrics_and_skips_unusable_files(tmp_path: Path) -> None:
    """Verify report metric aggregation includes valid metrics and skips invalid or nonmetric files."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_gamma095_seed0"
    metrics_path = _write_metrics(
        root,
        run_name,
        "evaluations/own_task",
        {
            "source_run_name": run_name,
            "source_run_kind": "direct_ppo",
            "action_interface": "pid_position",
            "evaluation_name": "own_task",
            "suite_task_name": "own_task",
            "task_shape_used_for_evaluation": "polyline",
            "mean_position_error_tracking_m": 0.12,
            "mean_position_error_m": 0.15,
            "final_position_error_m": 0.20,
            "max_position_error_m": 0.35,
            "mean_eval_reward": -0.4,
            "final_eval_reward": -0.2,
            "eval_terminated_count": 2,
            "eval_truncated_count": 1,
            "failure_overall_status": "failed",
            "failure_primary_mode": "attitude_instability",
        },
    )
    bad_dir = root / "bad_run" / "training" / "metrics"
    bad_dir.mkdir(parents=True)
    (bad_dir / "bad_run_metrics.json").write_text("{not json", encoding="utf-8")
    (bad_dir / "aggregate_metrics.json").write_text(json.dumps({"evaluation_name": "own_task"}), encoding="utf-8")
    (bad_dir / "evaluation_index_metrics.json").write_text(json.dumps({"entry_count": 1}), encoding="utf-8")
    (bad_dir / "run_manifest_metrics.json").write_text(json.dumps({"run_name": "bad_run"}), encoding="utf-8")

    rows = evaluation.report.build_report_metric_table(root=root)

    assert len(rows) == 1
    assert rows[0] == {
        "run_name": run_name,
        "method": "Direct PPO",
        "action_interface": "pid_position",
        "variant": "gamma095",
        "training_target": "m-taskdist_medium",
        "evaluation_name": "own_task",
        "suite_task_name": "own_task",
        "task_shape": "polyline",
        "mean_tracking_error_m": 0.12,
        "completion_ratio": None,
        "completion_adjusted_tracking_error_m": None,
        "completed_tracking_steps": None,
        "planned_tracking_steps": None,
        "completed_rollout_steps": None,
        "planned_rollout_steps": None,
        "rollout_completion_ratio": None,
        "mean_position_error_m": 0.15,
        "final_position_error_m": 0.20,
        "max_position_error_m": 0.35,
        "mean_eval_reward": -0.4,
        "final_eval_reward": -0.2,
        "terminated_count": 2,
        "truncated_count": 1,
        "failure_status": "failed",
        "primary_failure": "attitude_instability",
        "metrics_file": str(metrics_path),
    }


def test_build_report_metric_table_computes_completion_fields_from_counts(tmp_path: Path) -> None:
    """Verify report rows compute completion metrics when step counts are present."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    _write_metrics(
        root,
        run_name,
        "evaluations/generalization/line_basic",
        {
            "evaluation_name": "generalization",
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 0.05,
            "completed_tracking_steps": 2,
            "planned_tracking_steps": 10,
        },
    )

    rows = evaluation.report.build_report_metric_table(root=root)

    assert len(rows) == 1
    assert rows[0]["completed_tracking_steps"] == 2
    assert rows[0]["planned_tracking_steps"] == 10
    assert rows[0]["completion_ratio"] == pytest.approx(0.2)
    assert rows[0]["completion_adjusted_tracking_error_m"] == pytest.approx(0.25)


def test_build_report_metric_table_uses_completion_ratio_floor(tmp_path: Path) -> None:
    """Verify report-side adjusted error uses the same 0.05 denominator floor."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    _write_metrics(
        root,
        run_name,
        "evaluations/generalization/line_basic",
        {
            "evaluation_name": "generalization",
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 0.1,
            "completed_tracking_steps": 1,
            "planned_tracking_steps": 100,
        },
    )

    rows = evaluation.report.build_report_metric_table(root=root)

    assert rows[0]["completion_ratio"] == pytest.approx(0.01)
    assert rows[0]["completion_adjusted_tracking_error_m"] == pytest.approx(2.0)


def test_build_report_metric_table_infers_completion_fields_from_trace(tmp_path: Path) -> None:
    """Verify old metrics can derive completion fields from available trace diagnostics."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    metrics_path = _write_metrics(
        root,
        run_name,
        "evaluations/generalization/line_basic",
        {
            "evaluation_name": "generalization",
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 0.05,
            "eval_steps": 10,
        },
    )
    diagnostics_dir = metrics_path.parent.parent / "diagnostics"
    diagnostics_dir.mkdir(parents=True)
    trace_rows = [
        {
            "episode_index": 0,
            "is_tracking_phase": True,
            "exclude_start_hold_from_tracking_metrics": False,
            "exclude_final_hold_from_tracking_metrics": False,
            "tracking_phase_end_step": 10,
        },
        {
            "episode_index": 0,
            "is_tracking_phase": True,
            "exclude_start_hold_from_tracking_metrics": False,
            "exclude_final_hold_from_tracking_metrics": False,
            "tracking_phase_end_step": 10,
        },
    ]
    (diagnostics_dir / "evaluation_trace.jsonl").write_text("\n".join(json.dumps(row) for row in trace_rows), encoding="utf-8")

    rows = evaluation.report.build_report_metric_table(root=root)

    assert rows[0]["completed_tracking_steps"] == 2
    assert rows[0]["planned_tracking_steps"] == 10
    assert rows[0]["completion_ratio"] == pytest.approx(0.2)
    assert rows[0]["completion_adjusted_tracking_error_m"] == pytest.approx(0.25)


def test_build_report_metric_table_infers_method_action_and_variant_from_paths(tmp_path: Path) -> None:
    """Verify deterministic inference for report labels when metrics metadata is sparse."""
    root = tmp_path / "runs"
    run_names = (
        "direct_ppo_pid_dynprev_m-taskdist_medium_gamma095_seed0",
        "curriculum_manual_directrpm_dynprev_m-taskdist_medium_smooth001_seed0",
        "curriculum_llm_pid_dynprev_m-taskdist_medium_seed0",
        "direct_ppo_directrpm_dynprev_basic_show_seed0",
        "direct_ppo_pid_dynprev_basic_show_seed0",
    )
    for index, run_name in enumerate(run_names):
        _write_metrics(root, run_name, "training", {"mean_position_error_m": 0.1 + index})

    rows = {row["run_name"]: row for row in evaluation.report.build_report_metric_table(root=root)}

    assert rows[run_names[0]]["method"] == "Direct PPO"
    assert rows[run_names[0]]["action_interface"] == "pid_position"
    assert rows[run_names[0]]["variant"] == "gamma095"
    assert rows[run_names[0]]["training_target"] == "m-taskdist_medium"
    assert rows[run_names[1]]["method"] == "Manual curriculum"
    assert rows[run_names[1]]["action_interface"] == "direct_rpm"
    assert rows[run_names[1]]["variant"] == "smooth001"
    assert rows[run_names[1]]["training_target"] == "manual_curriculum"
    assert rows[run_names[2]]["method"] == "LLM curriculum"
    assert rows[run_names[2]]["action_interface"] == "pid_position"
    assert rows[run_names[2]]["variant"] == "default"
    assert rows[run_names[2]]["training_target"] == "llm_curriculum"
    assert rows[run_names[3]]["method"] == "Direct PPO"
    assert rows[run_names[3]]["action_interface"] == "direct_rpm"
    assert rows[run_names[3]]["variant"] == "basic_show"
    assert rows[run_names[3]]["variant"] != "default"
    assert rows[run_names[3]]["training_target"] == "direct_basic_show"
    assert rows[run_names[4]]["method"] == "Direct PPO"
    assert rows[run_names[4]]["action_interface"] == "pid_position"
    assert rows[run_names[4]]["variant"] == "basic_show"
    assert rows[run_names[4]]["variant"] != "default"
    assert rows[run_names[4]]["training_target"] == "direct_basic_show"


def test_compact_report_metric_table_uses_display_columns_without_metrics_file(tmp_path: Path) -> None:
    """Verify compact report rows contain display fields and omit verbose file paths."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    _write_metrics(root, run_name, "evaluations/generalization/line_basic", {"mean_position_error_m": 0.25})

    rows = evaluation.report.compact_report_metric_table(root=root)

    assert len(rows) == 1
    assert tuple(rows[0]) == evaluation.report.compact_report_columns()
    assert rows[0]["run_name"] == run_name
    assert rows[0]["training_target"] == "m-taskdist_medium"
    assert rows[0]["evaluation_name"] == "generalization"
    assert rows[0]["mean_tracking_error_m"] == pytest.approx(0.25)
    assert "failure_status" not in rows[0]
    assert "primary_failure" in rows[0]
    assert "metrics_file" not in rows[0]


def test_find_default_runs_root_prefers_local_storage_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default runs-root discovery supports notebook execution from the repo root."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "storage" / "runs").mkdir(parents=True)

    assert evaluation.report.find_default_runs_root() == Path("storage/runs")


def test_build_aggregated_report_metric_table_summarizes_generalization_rows(tmp_path: Path) -> None:
    """Verify run-level aggregation averages task metrics and excludes non-generalization rows."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    common_payload = {
        "source_run_name": run_name,
        "source_run_kind": "direct_ppo",
        "action_interface": "pid_position",
        "evaluation_name": "generalization",
    }
    _write_metrics(
        root,
        run_name,
        "evaluations/generalization/line_basic",
        {
            **common_payload,
            "suite_task_name": "line_basic",
            "task_shape": "line",
            "mean_position_error_tracking_m": 0.2,
            "completed_tracking_steps": 10,
            "planned_tracking_steps": 10,
            "mean_position_error_m": 0.3,
            "final_position_error_m": 0.4,
            "max_position_error_m": 0.5,
            "mean_eval_reward": -0.2,
            "final_eval_reward": -0.1,
            "eval_terminated_count": 1,
            "eval_truncated_count": 0,
            "failure_overall_status": "successful",
            "failure_primary_mode": "no_failure_detected",
        },
    )
    _write_metrics(
        root,
        run_name,
        "evaluations/generalization/circle_basic",
        {
            **common_payload,
            "suite_task_name": "circle_basic",
            "task_shape": "circle",
            "mean_position_error_tracking_m": 0.4,
            "completed_tracking_steps": 5,
            "planned_tracking_steps": 10,
            "mean_position_error_m": 0.5,
            "final_position_error_m": 0.6,
            "max_position_error_m": 0.7,
            "mean_eval_reward": -0.4,
            "final_eval_reward": -0.3,
            "eval_terminated_count": 2,
            "eval_truncated_count": 3,
            "failure_overall_status": "failed",
            "failure_primary_mode": "overshoot",
        },
    )
    _write_metrics(
        root,
        run_name,
        "evaluations/own_task",
        {**common_payload, "evaluation_name": "own_task", "suite_task_name": "own_task", "mean_position_error_tracking_m": 99.0},
    )
    _write_metrics(
        root,
        run_name,
        "evaluations/scenarios/easy",
        {**common_payload, "evaluation_name": "scenarios", "scenario_label": "easy", "mean_position_error_tracking_m": 99.0},
    )
    _write_metrics(
        root,
        run_name,
        "training",
        {**common_payload, "evaluation_name": "training", "suite_task_name": "training", "mean_position_error_tracking_m": 99.0},
    )

    rows = evaluation.report.build_aggregated_report_metric_table(root=root)

    assert len(rows) == 1
    row = rows[0]
    assert row["run_name"] == run_name
    assert row["method"] == "Direct PPO"
    assert row["action_interface"] == "pid_position"
    assert row["variant"] == "default"
    assert row["training_target"] == "m-taskdist_medium"
    assert row["evaluated_task_count"] == 2
    assert row["mean_tracking_error_m"] == pytest.approx(0.3)
    assert row["completion_ratio"] == pytest.approx(0.75)
    assert row["completion_adjusted_tracking_error_m"] == pytest.approx(0.5)
    assert row["mean_position_error_m"] == pytest.approx(0.4)
    assert row["final_position_error_m"] == pytest.approx(0.5)
    assert row["max_position_error_m"] == pytest.approx(0.6)
    assert row["mean_eval_reward"] == pytest.approx(-0.3)
    assert row["final_eval_reward"] == pytest.approx(-0.2)
    assert row["terminated_count"] == 3
    assert row["truncated_count"] == 3
    assert row["failure_status"] == "failed, successful"
    assert row["primary_failure"] == "no_failure_detected, overshoot"


def test_build_aggregated_report_metric_table_includes_basic_show_generalization_runs(tmp_path: Path) -> None:
    """Verify basic-show direct runs are not mistaken for show/scenario evaluations."""
    root = tmp_path / "runs"
    basic_run_name = "direct_ppo_pid_dynprev_basic_show_seed0"
    medium_run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    common_payload = {
        "source_run_kind": "direct_ppo",
        "action_interface": "pid_position",
        "evaluation_name": "generalization",
    }
    for task_name, value in (("line_basic", 0.2), ("circle_basic", 0.4)):
        _write_metrics(
            root,
            basic_run_name,
            f"evaluations/generalization/{task_name}",
            {
                **common_payload,
                "source_run_name": basic_run_name,
                "suite_task_name": task_name,
                "mean_position_error_tracking_m": value,
                "eval_terminated_count": 1,
                "failure_overall_status": "successful",
            },
        )
    _write_metrics(
        root,
        basic_run_name,
        "evaluations/own_task",
        {
            **common_payload,
            "source_run_name": basic_run_name,
            "evaluation_name": "own_task",
            "suite_task_name": "own_task",
            "mean_position_error_tracking_m": 99.0,
        },
    )
    _write_metrics(
        root,
        basic_run_name,
        "evaluations/scenarios/show_easy",
        {
            **common_payload,
            "source_run_name": basic_run_name,
            "evaluation_name": "show_easy",
            "scenario_label": "show_easy",
            "mean_position_error_tracking_m": 99.0,
        },
    )
    _write_metrics(
        root,
        basic_run_name,
        "training",
        {
            **common_payload,
            "source_run_name": basic_run_name,
            "evaluation_name": "training",
            "suite_task_name": "training",
            "mean_position_error_tracking_m": 99.0,
        },
    )
    _write_metrics(
        root,
        medium_run_name,
        "evaluations/generalization/line_basic",
        {
            **common_payload,
            "source_run_name": medium_run_name,
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 0.6,
        },
    )

    rows = evaluation.report.build_aggregated_report_metric_table(root=root)
    rows_by_run = {row["run_name"]: row for row in rows}

    assert set(rows_by_run) == {basic_run_name, medium_run_name}
    assert rows_by_run[basic_run_name]["training_target"] == "direct_basic_show"
    assert rows_by_run[basic_run_name]["variant"] == "basic_show"
    assert rows_by_run[basic_run_name]["variant"] != "default"
    assert rows_by_run[basic_run_name]["evaluated_task_count"] == 2
    assert rows_by_run[basic_run_name]["mean_tracking_error_m"] == pytest.approx(0.3)
    assert rows_by_run[basic_run_name]["terminated_count"] == 2
    assert rows_by_run[medium_run_name]["training_target"] == "m-taskdist_medium"
    assert rows_by_run[medium_run_name]["evaluated_task_count"] == 1


def test_build_aggregated_report_metric_table_sorts_missing_adjusted_metric_last(tmp_path: Path) -> None:
    """Verify rows without completion-adjusted metrics sort after ranked rows."""
    root = tmp_path / "runs"
    ranked_run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_ranked_seed0"
    old_run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_old_seed0"
    _write_metrics(
        root,
        old_run_name,
        "evaluations/generalization/line_basic",
        {
            "evaluation_name": "generalization",
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 0.01,
        },
    )
    _write_metrics(
        root,
        ranked_run_name,
        "evaluations/generalization/line_basic",
        {
            "evaluation_name": "generalization",
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 0.2,
            "completion_ratio": 1.0,
            "completion_adjusted_tracking_error_m": 0.2,
        },
    )

    rows = evaluation.report.build_aggregated_report_metric_table(root=root)

    assert [row["run_name"] for row in rows] == [ranked_run_name, old_run_name]
    assert rows[-1]["completion_adjusted_tracking_error_m"] is None


def test_build_aggregated_report_metric_table_handles_missing_optional_metrics(tmp_path: Path) -> None:
    """Verify missing optional aggregate fields do not crash or fabricate values."""
    root = tmp_path / "runs"
    run_name = "curriculum_manual_directrpm_dynprev_m-taskdist_medium_seed0"
    _write_metrics(
        root,
        run_name,
        "evaluations/line_eval/line_basic",
        {
            "evaluation_name": "line_eval",
            "suite_task_name": "line_basic",
            "mean_position_error_m": 0.25,
        },
    )

    rows = evaluation.report.build_aggregated_report_metric_table(root=root)

    assert len(rows) == 1
    assert rows[0]["method"] == "Manual curriculum"
    assert rows[0]["action_interface"] == "direct_rpm"
    assert rows[0]["training_target"] == "manual_curriculum"
    assert rows[0]["evaluated_task_count"] == 1
    assert rows[0]["mean_tracking_error_m"] == pytest.approx(0.25)
    assert rows[0]["mean_position_error_m"] == pytest.approx(0.25)
    assert rows[0]["completion_ratio"] is None
    assert rows[0]["completion_adjusted_tracking_error_m"] is None
    assert rows[0]["final_position_error_m"] is None
    assert rows[0]["terminated_count"] is None
    assert rows[0]["truncated_count"] is None


def test_build_aggregated_report_metric_table_keeps_final_manual_curriculum_stage(tmp_path: Path) -> None:
    """Verify manual curriculum aggregation excludes earlier stage evaluations."""
    root = tmp_path / "runs"
    run_name = "curriculum_manual_pid_dynprev_m-taskdist_medium_seed0"
    _write_metrics(
        root,
        run_name,
        "stages/stage01_hover/evaluations/generalization/line_basic",
        {
            "source_run_name": "curriculum_manual_pid_dynprev_m-taskdist_medium_stage01_hover_seed0",
            "source_run_kind": "curriculum_stage",
            "source_curriculum_kind": "manual",
            "source_stage": {"stage_index": 1, "stage_name": "hover"},
            "evaluation_name": "generalization",
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 99.0,
            "eval_terminated_count": 9,
            "eval_truncated_count": 9,
        },
    )
    for task_name, value in (("line_basic", 0.2), ("circle_basic", 0.4)):
        _write_metrics(
            root,
            run_name,
            f"stages/stage03_medium/evaluations/generalization/{task_name}",
            {
                "source_run_name": "curriculum_manual_pid_dynprev_m-taskdist_medium_stage03_medium_seed0",
                "source_run_kind": "curriculum_stage",
                "source_curriculum_kind": "manual",
                "source_stage": {"stage_index": 3, "stage_name": "medium"},
                "model_scope": "final-stage",
                "model_role": "stage",
                "evaluation_name": "generalization",
                "suite_task_name": task_name,
                "mean_position_error_tracking_m": value,
                "eval_terminated_count": 1,
                "eval_truncated_count": 2,
            },
        )

    rows = evaluation.report.build_aggregated_report_metric_table(root=root)

    assert len(rows) == 1
    assert rows[0]["run_name"] == run_name
    assert rows[0]["method"] == "Manual curriculum"
    assert rows[0]["training_target"] == "manual_curriculum"
    assert rows[0]["evaluated_task_count"] == 2
    assert rows[0]["mean_tracking_error_m"] == pytest.approx(0.3)
    assert rows[0]["terminated_count"] == 2
    assert rows[0]["truncated_count"] == 4


def test_build_aggregated_report_metric_table_keeps_final_llm_curriculum_stage(tmp_path: Path) -> None:
    """Verify LLM curriculum aggregation excludes earlier stage evaluations."""
    root = tmp_path / "runs"
    run_name = "llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0"
    _write_metrics(
        root,
        run_name,
        "stages/stage02_line/evaluations/generalization/line_basic",
        {
            "source_run_name": "llm_curriculum_directrpm_dynprev_m-taskdist_medium_stage02_line_seed0",
            "source_run_kind": "curriculum_stage",
            "source_curriculum_kind": "llm",
            "source_stage": {"stage_index": 2, "stage_name": "line"},
            "evaluation_name": "generalization",
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 99.0,
        },
    )
    _write_metrics(
        root,
        run_name,
        "stages/stage10_hover/evaluations/generalization/line_basic",
        {
            "source_run_name": "llm_curriculum_directrpm_dynprev_m-taskdist_medium_stage10_hover_seed0",
            "source_run_kind": "curriculum_stage",
            "source_curriculum_kind": "llm",
            "source_stage": {"stage_index": 10, "stage_name": "hover"},
            "model_scope": "final-stage",
            "model_role": "stage",
            "evaluation_name": "generalization",
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 0.5,
        },
    )

    rows = evaluation.report.build_aggregated_report_metric_table(root=root)

    assert len(rows) == 1
    assert rows[0]["run_name"] == run_name
    assert rows[0]["method"] == "LLM curriculum"
    assert rows[0]["action_interface"] == "direct_rpm"
    assert rows[0]["training_target"] == "llm_curriculum"
    assert rows[0]["evaluated_task_count"] == 1
    assert rows[0]["mean_tracking_error_m"] == pytest.approx(0.5)


def test_build_aggregated_scenario_metric_table_summarizes_show_rows_and_excludes_other_categories(tmp_path: Path) -> None:
    """Verify show/OOD aggregation summarizes scenarios without mixing fixed/generalization rows."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    common_payload = {
        "source_run_name": run_name,
        "source_run_kind": "direct_ppo",
        "action_interface": "pid_position",
    }
    scenario_values = (
        ("show_easy", 0.2, 1),
        ("show_medium", 0.4, 2),
        ("show_hard", 0.6, 3),
        ("scenario", 0.8, 4),
        ("scenarios", 1.0, 5),
    )
    for scenario_name, value, count in scenario_values:
        _write_metrics(
            root,
            run_name,
            f"evaluations/scenarios/{scenario_name}",
            {
                **common_payload,
                "evaluation_name": scenario_name,
                "scenario_label": scenario_name,
                "mean_position_error_tracking_m": value,
                "completion_ratio": value,
                "completion_adjusted_tracking_error_m": value + 0.05,
                "rollout_completion_ratio": min(value + 0.1, 1.0),
                "mean_position_error_m": value + 0.1,
                "final_position_error_m": value + 0.2,
                "max_position_error_m": value + 0.3,
                "mean_eval_reward": -value,
                "final_eval_reward": -(value + 0.1),
                "eval_terminated_count": count,
                "eval_truncated_count": count + 10,
                "failure_overall_status": "failed" if count % 2 == 0 else "successful",
                "failure_primary_mode": "overshoot" if count % 2 == 0 else "no_failure_detected",
            },
        )
    _write_metrics(
        root,
        run_name,
        "evaluations/generalization/line_basic",
        {
            **common_payload,
            "evaluation_name": "generalization",
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 99.0,
        },
    )
    _write_metrics(
        root,
        run_name,
        "evaluations/own_task",
        {
            **common_payload,
            "evaluation_name": "own_task",
            "suite_task_name": "own_task",
            "mean_position_error_tracking_m": 99.0,
        },
    )
    _write_metrics(
        root,
        run_name,
        "training",
        {
            **common_payload,
            "evaluation_name": "training",
            "suite_task_name": "training",
            "mean_position_error_tracking_m": 99.0,
        },
    )

    scenario_rows = evaluation.report.build_aggregated_scenario_metric_table(root=root)
    generalization_rows = evaluation.report.build_aggregated_report_metric_table(root=root)

    assert len(scenario_rows) == 1
    row = scenario_rows[0]
    assert row["run_name"] == run_name
    assert row["method"] == "Direct PPO"
    assert row["training_target"] == "m-taskdist_medium"
    assert row["action_interface"] == "pid_position"
    assert row["variant"] == "default"
    assert row["evaluated_scenario_count"] == 5
    assert row["mean_tracking_error_m"] == pytest.approx(0.6)
    assert row["completion_ratio"] == pytest.approx(0.6)
    assert row["completion_adjusted_tracking_error_m"] == pytest.approx(0.65)
    assert row["rollout_completion_ratio"] == pytest.approx(0.68)
    assert row["mean_position_error_m"] == pytest.approx(0.7)
    assert row["final_position_error_m"] == pytest.approx(0.8)
    assert row["max_position_error_m"] == pytest.approx(0.9)
    assert row["mean_eval_reward"] == pytest.approx(-0.6)
    assert row["final_eval_reward"] == pytest.approx(-0.7)
    assert row["terminated_count"] == 15
    assert row["truncated_count"] == 65
    assert row["failure_status"] == "failed, successful"
    assert row["primary_failure"] == "no_failure_detected, overshoot"
    assert len(generalization_rows) == 1
    assert generalization_rows[0]["evaluated_task_count"] == 1
    assert generalization_rows[0]["mean_tracking_error_m"] == pytest.approx(99.0)


def test_scenario_report_metrics_infer_completion_from_stale_trace(tmp_path: Path) -> None:
    """Verify stale scenario artifacts can infer completion from trace metadata."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    metrics_path = _write_metrics(
        root,
        run_name,
        "evaluations/scenarios/show_easy",
        {
            "evaluation_name": "show_easy",
            "scenario_label": "show_easy",
            "mean_position_error_m": 0.9,
            "effective_max_steps": 6,
            "reference_motion_steps": 4,
        },
    )
    trace_path = metrics_path.parent.parent / "traces" / evaluation.report.SCENARIO_TRACE_FILENAME
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "\n".join(
            json.dumps(record)
            for record in (
                {
                    "step_index": 0,
                    "is_start_hold": True,
                    "is_phase_hold": False,
                    "is_final_hold": False,
                    "is_tracking_phase": False,
                    "position_error_m": 0.8,
                },
                {
                    "step_index": 1,
                    "is_start_hold": False,
                    "is_phase_hold": False,
                    "is_final_hold": False,
                    "is_tracking_phase": True,
                    "position_error_m": 0.1,
                },
                {
                    "step_index": 2,
                    "is_start_hold": False,
                    "is_phase_hold": False,
                    "is_final_hold": False,
                    "is_tracking_phase": True,
                    "position_error_m": 0.3,
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    detailed = evaluation.report.build_report_metric_table(root=root)
    scenario_rows = evaluation.report.build_aggregated_scenario_metric_table(root=root)

    assert len(detailed) == 1
    assert detailed[0]["mean_tracking_error_m"] == pytest.approx(0.2)
    assert detailed[0]["completed_tracking_steps"] == 2
    assert detailed[0]["planned_tracking_steps"] == 4
    assert detailed[0]["completion_ratio"] == pytest.approx(0.5)
    assert detailed[0]["completion_adjusted_tracking_error_m"] == pytest.approx(0.4)
    assert detailed[0]["completed_rollout_steps"] == 3
    assert detailed[0]["planned_rollout_steps"] == 6
    assert detailed[0]["rollout_completion_ratio"] == pytest.approx(0.5)
    assert scenario_rows[0]["completion_ratio"] == pytest.approx(0.5)
    assert scenario_rows[0]["completion_adjusted_tracking_error_m"] == pytest.approx(0.4)
    assert scenario_rows[0]["rollout_completion_ratio"] == pytest.approx(0.5)


def test_scenario_report_metrics_do_not_fabricate_tracking_completion_without_trace(tmp_path: Path) -> None:
    """Verify old scenario metrics without traces keep tracking completion fields empty."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    _write_metrics(
        root,
        run_name,
        "evaluations/scenarios/show_easy",
        {
            "evaluation_name": "show_easy",
            "scenario_label": "show_easy",
            "mean_position_error_m": 0.9,
            "effective_max_steps": 6,
            "reference_motion_steps": 4,
        },
    )

    detailed = evaluation.report.build_report_metric_table(root=root)
    scenario_rows = evaluation.report.build_aggregated_scenario_metric_table(root=root)

    assert len(detailed) == 1
    assert detailed[0]["completion_ratio"] is None
    assert detailed[0]["completion_adjusted_tracking_error_m"] is None
    assert detailed[0]["completed_tracking_steps"] is None
    assert detailed[0]["planned_tracking_steps"] is None
    assert detailed[0]["rollout_completion_ratio"] is None
    assert scenario_rows[0]["completion_ratio"] is None
    assert scenario_rows[0]["completion_adjusted_tracking_error_m"] is None
    assert scenario_rows[0]["rollout_completion_ratio"] is None


def test_build_aggregated_scenario_metric_table_keeps_final_curriculum_stage(tmp_path: Path) -> None:
    """Verify scenario aggregation excludes intermediate curriculum stages."""
    root = tmp_path / "runs"
    run_name = "llm_curriculum_directrpm_dynprev_m-taskdist_medium_seed0"
    _write_metrics(
        root,
        run_name,
        "stages/stage02_line/evaluations/scenarios/show_easy",
        {
            "source_run_name": "llm_curriculum_directrpm_dynprev_m-taskdist_medium_stage02_line_seed0",
            "source_run_kind": "curriculum_stage",
            "source_curriculum_kind": "llm",
            "source_stage": {"stage_index": 2, "stage_name": "line"},
            "evaluation_name": "show_easy",
            "scenario_label": "show_easy",
            "mean_position_error_tracking_m": 99.0,
            "eval_terminated_count": 9,
        },
    )
    for scenario_name, value in (("show_easy", 0.2), ("show_hard", 0.4)):
        _write_metrics(
            root,
            run_name,
            f"stages/stage10_hover/evaluations/scenarios/{scenario_name}",
            {
                "source_run_name": "llm_curriculum_directrpm_dynprev_m-taskdist_medium_stage10_hover_seed0",
                "source_run_kind": "curriculum_stage",
                "source_curriculum_kind": "llm",
                "source_stage": {"stage_index": 10, "stage_name": "hover"},
                "model_scope": "final-stage",
                "model_role": "stage",
                "evaluation_name": scenario_name,
                "scenario_label": scenario_name,
                "mean_position_error_tracking_m": value,
                "eval_terminated_count": 1,
                "eval_truncated_count": 2,
            },
        )

    rows = evaluation.report.build_aggregated_scenario_metric_table(root=root)

    assert len(rows) == 1
    assert rows[0]["run_name"] == run_name
    assert rows[0]["method"] == "LLM curriculum"
    assert rows[0]["training_target"] == "llm_curriculum"
    assert rows[0]["action_interface"] == "direct_rpm"
    assert rows[0]["evaluated_scenario_count"] == 2
    assert rows[0]["mean_tracking_error_m"] == pytest.approx(0.3)
    assert rows[0]["terminated_count"] == 2
    assert rows[0]["truncated_count"] == 4


def test_notebook_results_cell_does_not_write_csv() -> None:
    """Verify notebook report display cells do not export CSV files."""
    notebook = json.loads(Path("Drone_RL_LLM_Curriculum.ipynb").read_text(encoding="utf-8"))
    code_source = "\n".join("".join(cell.get("source", [])) for cell in notebook.get("cells", []) if cell.get("cell_type") == "code")

    assert "to_csv" not in code_source
    assert "report_metric_comparison.csv" not in code_source
    assert "build_aggregated_report_metric_table" in code_source
    assert "build_aggregated_scenario_metric_table" in code_source
    assert "Fixed evaluation summary" in code_source
    assert "Show / OOD scenario summary" in code_source
    assert "completion_adjusted_tracking_error_m" in code_source
    assert "tracks well briefly but fails early" in code_source


def test_compact_aggregated_report_metric_table_uses_display_columns(tmp_path: Path) -> None:
    """Verify compact aggregate rows expose the notebook display columns only."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_gamma095_seed0"
    _write_metrics(
        root,
        run_name,
        "evaluations/generalization/line_basic",
        {
            "evaluation_name": "generalization",
            "suite_task_name": "line_basic",
            "mean_position_error_tracking_m": 0.1,
            "final_eval_reward": -0.5,
        },
    )

    rows = evaluation.report.compact_aggregated_report_metric_table(root=root)

    assert len(rows) == 1
    assert tuple(rows[0]) == evaluation.report.compact_aggregated_report_columns()
    assert rows[0]["run_name"] == run_name
    assert rows[0]["variant"] == "gamma095"
    assert rows[0]["training_target"] == "m-taskdist_medium"
    assert rows[0]["evaluated_task_count"] == 1
    assert rows[0]["mean_tracking_error_m"] == pytest.approx(0.1)
    assert "completion_ratio" in evaluation.report.compact_aggregated_report_columns()
    assert "completion_adjusted_tracking_error_m" in evaluation.report.compact_aggregated_report_columns()
    assert "failure_status" not in evaluation.report.compact_aggregated_report_columns()
    assert "primary_failure" in evaluation.report.compact_aggregated_report_columns()
    assert "failure_status" not in rows[0]
    assert "primary_failure" in rows[0]
    assert "final_eval_reward" not in rows[0]
    assert "metrics_file" not in rows[0]


def test_compact_aggregated_scenario_metric_table_uses_display_columns(tmp_path: Path) -> None:
    """Verify compact scenario rows expose the notebook display columns only."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_directrpm_dynprev_basic_show_seed0"
    _write_metrics(
        root,
        run_name,
        "evaluations/scenarios/show_easy",
        {
            "evaluation_name": "show_easy",
            "scenario_label": "show_easy",
            "mean_position_error_tracking_m": 0.2,
            "completion_ratio": 0.5,
            "completion_adjusted_tracking_error_m": 0.4,
            "rollout_completion_ratio": 0.8,
            "final_eval_reward": -0.5,
        },
    )

    rows = evaluation.report.compact_aggregated_scenario_metric_table(root=root)

    assert len(rows) == 1
    assert tuple(rows[0]) == evaluation.report.compact_aggregated_scenario_columns()
    assert rows[0]["run_name"] == run_name
    assert rows[0]["training_target"] == "direct_basic_show"
    assert rows[0]["variant"] == "basic_show"
    assert rows[0]["variant"] != "default"
    assert rows[0]["action_interface"] == "direct_rpm"
    assert rows[0]["evaluated_scenario_count"] == 1
    assert rows[0]["mean_tracking_error_m"] == pytest.approx(0.2)
    assert rows[0]["completion_ratio"] == pytest.approx(0.5)
    assert rows[0]["completion_adjusted_tracking_error_m"] == pytest.approx(0.4)
    assert rows[0]["rollout_completion_ratio"] == pytest.approx(0.8)
    assert "completion_ratio" in evaluation.report.compact_aggregated_scenario_columns()
    assert "completion_adjusted_tracking_error_m" in evaluation.report.compact_aggregated_scenario_columns()
    assert "rollout_completion_ratio" in evaluation.report.compact_aggregated_scenario_columns()
    assert "failure_status" not in evaluation.report.compact_aggregated_scenario_columns()
    assert "primary_failure" in evaluation.report.compact_aggregated_scenario_columns()
    assert "failure_status" not in rows[0]
    assert "primary_failure" in rows[0]
    assert "final_eval_reward" not in rows[0]
    assert "metrics_file" not in rows[0]
