"""Tests for final-report artifact and metrics helpers."""

# ruff: noqa: S101, PLR2004

from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from src import evaluation

if TYPE_CHECKING:
    from pathlib import Path


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

    assert len(rows) == 18
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
