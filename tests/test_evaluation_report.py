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
            "training_target": "tracking_medium",
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
            "training_target": "tracking_medium",
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
        "evaluation_name": "own_task",
        "suite_task_name": "own_task",
        "task_shape": "polyline",
        "mean_tracking_error_m": 0.12,
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


def test_build_report_metric_table_infers_method_action_and_variant_from_paths(tmp_path: Path) -> None:
    """Verify deterministic inference for report labels when metrics metadata is sparse."""
    root = tmp_path / "runs"
    run_names = (
        "direct_ppo_pid_dynprev_m-taskdist_medium_gamma095_seed0",
        "curriculum_manual_directrpm_dynprev_m-taskdist_medium_smooth001_seed0",
        "curriculum_llm_pid_dynprev_m-taskdist_medium_seed0",
    )
    for index, run_name in enumerate(run_names):
        _write_metrics(root, run_name, "training", {"mean_position_error_m": 0.1 + index})

    rows = {row["run_name"]: row for row in evaluation.report.build_report_metric_table(root=root)}

    assert rows[run_names[0]]["method"] == "Direct PPO"
    assert rows[run_names[0]]["action_interface"] == "pid_position"
    assert rows[run_names[0]]["variant"] == "gamma095"
    assert rows[run_names[1]]["method"] == "Manual curriculum"
    assert rows[run_names[1]]["action_interface"] == "direct_rpm"
    assert rows[run_names[1]]["variant"] == "smooth001"
    assert rows[run_names[2]]["method"] == "LLM curriculum"
    assert rows[run_names[2]]["action_interface"] == "pid_position"
    assert rows[run_names[2]]["variant"] == "default"


def test_compact_report_metric_table_uses_display_columns_without_metrics_file(tmp_path: Path) -> None:
    """Verify compact report rows contain display fields and omit verbose file paths."""
    root = tmp_path / "runs"
    run_name = "direct_ppo_pid_dynprev_m-taskdist_medium_seed0"
    _write_metrics(root, run_name, "evaluations/generalization/line_basic", {"mean_position_error_m": 0.25})

    rows = evaluation.report.compact_report_metric_table(root=root)

    assert len(rows) == 1
    assert tuple(rows[0]) == evaluation.report.compact_report_columns()
    assert rows[0]["run_name"] == run_name
    assert rows[0]["evaluation_name"] == "generalization"
    assert rows[0]["mean_tracking_error_m"] == pytest.approx(0.25)
    assert "metrics_file" not in rows[0]


def test_find_default_runs_root_prefers_local_storage_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify default runs-root discovery supports notebook execution from the repo root."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "storage" / "runs").mkdir(parents=True)

    assert evaluation.report.find_default_runs_root() == Path("storage/runs")
