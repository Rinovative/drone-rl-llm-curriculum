"""
===============================================================================
evaluation_report.py
===============================================================================
Load compact final-report metadata from planned runs and generated artifacts.

Responsibilities:
  - Read the source-controlled experiment matrix for planned run metadata
  - Inspect run-name folders directly under the configured artifact root
  - Load selected scalar metrics from existing metrics JSON files
  - Report missing artifacts explicitly without fabricating results

Design principles:
  - Default to /workspace/storage/runs because generated runs live directly there
  - Keep notebook-facing helpers small, typed, and side-effect free
  - Treat generated artifacts as optional evidence, not source-controlled truth

Boundaries:
  - Training, evaluation rollout execution, rendering, and notebook presentation belong elsewhere
  - This module must not start simulators, PPO training, or long evaluation jobs
===============================================================================

"""

from __future__ import annotations

import csv
import fnmatch
import json
import re
from math import isfinite
from pathlib import Path
from typing import Any

MATRIX_SCRIPT_PATH = Path("scripts/experiment_matrix.sh")
MATRIX_TSV_PATH = Path("docs/experiments/overnight_lane_assignment.tsv")
DEFAULT_ARTIFACT_ROOT = Path("/workspace/storage/runs")
RUN_MANIFEST_FILENAME = "run_manifest.json"
METRICS_GLOB = "*_metrics.json"
EVALUATION_TRACE_FILENAME = "evaluation_trace.jsonl"
SCENARIO_TRACE_FILENAME = "scenario_rollout_trace.jsonl"
EPISODE_SUMMARIES_FILENAME = "episode_summaries.json"
COMPLETION_ADJUSTMENT_MIN_RATIO = 0.05
MEDIA_SUFFIXES = (".gif", ".png", ".jpg", ".jpeg", ".mp4")
REPORT_METRIC_KEYS = (
    "evaluation_name",
    "evaluation_suite_name",
    "suite_task_name",
    "mean_position_error_tracking_m",
    "completed_tracking_steps",
    "planned_tracking_steps",
    "completion_ratio",
    "completion_adjusted_tracking_error_m",
    "completed_rollout_steps",
    "planned_rollout_steps",
    "rollout_completion_ratio",
    "mean_position_error_m",
    "final_position_error_m",
    "max_position_error_m",
    "rmse_position_error_m",
    "success_rate",
    "crash_rate",
    "mean_eval_reward",
    "final_eval_reward",
    "mean_reward",
    "final_reward",
    "failure_overall_status",
    "failure_primary_mode",
    "eval_terminated_count",
    "eval_truncated_count",
    "episode_count",
)
REPORT_METRIC_OUTPUT_COLUMNS = (
    "run_name",
    "method",
    "action_interface",
    "variant",
    "training_target",
    "evaluation_name",
    "suite_task_name",
    "task_shape",
    "mean_tracking_error_m",
    "completion_ratio",
    "completion_adjusted_tracking_error_m",
    "completed_tracking_steps",
    "planned_tracking_steps",
    "completed_rollout_steps",
    "planned_rollout_steps",
    "rollout_completion_ratio",
    "mean_position_error_m",
    "final_position_error_m",
    "max_position_error_m",
    "mean_eval_reward",
    "final_eval_reward",
    "terminated_count",
    "truncated_count",
    "failure_status",
    "primary_failure",
    "metrics_file",
)
COMPACT_REPORT_METRIC_COLUMNS = (
    "run_name",
    "method",
    "action_interface",
    "variant",
    "training_target",
    "evaluation_name",
    "suite_task_name",
    "task_shape",
    "mean_tracking_error_m",
    "completion_ratio",
    "completion_adjusted_tracking_error_m",
    "rollout_completion_ratio",
    "mean_position_error_m",
    "final_position_error_m",
    "max_position_error_m",
    "mean_eval_reward",
    "primary_failure",
)
_AGGREGATED_REPORT_GROUP_COLUMNS = (
    "run_name",
    "method",
    "action_interface",
    "variant",
    "training_target",
)
_AGGREGATED_REPORT_MEAN_COLUMNS = (
    "mean_tracking_error_m",
    "completion_ratio",
    "completion_adjusted_tracking_error_m",
    "rollout_completion_ratio",
    "mean_position_error_m",
    "final_position_error_m",
    "max_position_error_m",
    "mean_eval_reward",
    "final_eval_reward",
)
_AGGREGATED_REPORT_COUNT_COLUMNS = (
    "terminated_count",
    "truncated_count",
)
AGGREGATED_REPORT_METRIC_COLUMNS = (
    *_AGGREGATED_REPORT_GROUP_COLUMNS,
    "evaluated_task_count",
    *_AGGREGATED_REPORT_MEAN_COLUMNS,
    *_AGGREGATED_REPORT_COUNT_COLUMNS,
    "failure_status",
    "primary_failure",
)
COMPACT_AGGREGATED_REPORT_METRIC_COLUMNS = (
    "run_name",
    "method",
    "action_interface",
    "variant",
    "training_target",
    "evaluated_task_count",
    "mean_tracking_error_m",
    "completion_ratio",
    "completion_adjusted_tracking_error_m",
    "rollout_completion_ratio",
    "mean_position_error_m",
    "final_position_error_m",
    "max_position_error_m",
    "terminated_count",
    "truncated_count",
    "primary_failure",
)
_AGGREGATED_SCENARIO_GROUP_COLUMNS = (
    "run_name",
    "method",
    "training_target",
    "action_interface",
    "variant",
)
_AGGREGATED_SCENARIO_REPORT_MEAN_COLUMNS = _AGGREGATED_REPORT_MEAN_COLUMNS
AGGREGATED_SCENARIO_REPORT_METRIC_COLUMNS = (
    *_AGGREGATED_SCENARIO_GROUP_COLUMNS,
    "evaluated_scenario_count",
    *_AGGREGATED_SCENARIO_REPORT_MEAN_COLUMNS,
    *_AGGREGATED_REPORT_COUNT_COLUMNS,
    "failure_status",
    "primary_failure",
)
COMPACT_AGGREGATED_SCENARIO_REPORT_METRIC_COLUMNS = (
    "run_name",
    "method",
    "training_target",
    "action_interface",
    "variant",
    "evaluated_scenario_count",
    "mean_tracking_error_m",
    "completion_ratio",
    "completion_adjusted_tracking_error_m",
    "rollout_completion_ratio",
    "mean_position_error_m",
    "final_position_error_m",
    "max_position_error_m",
    "terminated_count",
    "truncated_count",
    "primary_failure",
)
_REPORT_METRIC_FIELD_SOURCES = {
    "mean_tracking_error_m": ("mean_tracking_error_m", "mean_position_error_tracking_m", "mean_position_error_m"),
    "completed_tracking_steps": ("completed_tracking_steps",),
    "planned_tracking_steps": ("planned_tracking_steps",),
    "completion_ratio": ("completion_ratio", "tracking_completion_ratio"),
    "completion_adjusted_tracking_error_m": ("completion_adjusted_tracking_error_m",),
    "completed_rollout_steps": ("completed_rollout_steps", "steps", "total_steps"),
    "planned_rollout_steps": ("planned_rollout_steps", "effective_max_steps", "eval_steps"),
    "rollout_completion_ratio": ("rollout_completion_ratio", "rollout_step_fraction"),
    "mean_position_error_m": ("mean_position_error_m", "mean_position_error_tracking_m"),
    "final_position_error_m": ("final_position_error_m",),
    "max_position_error_m": ("max_position_error_m",),
    "mean_eval_reward": ("mean_eval_reward", "mean_reward", "eval_mean_reward"),
    "final_eval_reward": ("final_eval_reward", "final_reward", "eval_final_reward"),
    "terminated_count": ("terminated_count", "eval_terminated_count", "terminated"),
    "truncated_count": ("truncated_count", "eval_truncated_count", "truncated"),
    "failure_status": ("failure_status", "failure_overall_status", "overall_status"),
    "primary_failure": ("primary_failure", "failure_primary_mode", "primary_failure_mode"),
}
_REPORT_TASK_SHAPE_KEYS = (
    "task_shape",
    "task_shape_used_for_evaluation",
    "own_task_shape",
    "training_task_shape",
    "task_distribution_base_task_shape",
)
_REPORT_STEP_COUNT_COLUMNS = (
    "completed_tracking_steps",
    "planned_tracking_steps",
    "completed_rollout_steps",
    "planned_rollout_steps",
)
_REPORT_COMPLETION_COLUMNS = (
    "completed_tracking_steps",
    "planned_tracking_steps",
    "completion_ratio",
    "completion_adjusted_tracking_error_m",
    "completed_rollout_steps",
    "planned_rollout_steps",
    "rollout_completion_ratio",
)
_REPORT_USABLE_METRIC_KEYS = (
    *(
        key
        for keys in _REPORT_METRIC_FIELD_SOURCES.values()
        for key in keys
        if key not in {"actual_eval_steps", "eval_steps", "terminated", "truncated"}
    ),
    "terminated",
    "truncated",
)


def load_experiment_matrix(path: str | Path = MATRIX_SCRIPT_PATH) -> list[dict[str, Any]]:
    """
    Load the planned experiment matrix from the active script or legacy TSV.

    Parameters
    ----------
    path
        Active shell matrix or legacy TSV file with planned experiment rows.

    Returns
    -------
    list[dict[str, Any]]
        Matrix rows with numeric lane and unit-count fields normalized when present.

    """
    matrix_path = Path(path)
    if not matrix_path.exists():
        return []
    if matrix_path.suffix == ".sh":
        return _load_experiment_matrix_script(matrix_path)
    return _load_experiment_matrix_tsv(matrix_path)


def expected_run_names(matrix_rows: list[dict[str, Any]] | None = None) -> tuple[str, ...]:
    """Return planned run names from matrix rows in matrix order."""
    rows = load_experiment_matrix() if matrix_rows is None else matrix_rows
    return tuple(_row_run_name(row) for row in rows if _row_run_name(row))


def _load_experiment_matrix_tsv(matrix_path: Path) -> list[dict[str, Any]]:
    """Load a legacy TSV experiment matrix."""
    with matrix_path.open("r", encoding="utf-8", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle, delimiter="\t")]
    return _normalize_matrix_rows(rows)


def _load_experiment_matrix_script(matrix_path: Path) -> list[dict[str, Any]]:
    """Load the active shell-script experiment matrix without executing it."""
    script_text = matrix_path.read_text(encoding="utf-8")
    kind_by_id = _read_case_echo_map(script_text, "experiment_kind")
    config_by_id = _read_case_echo_map(script_text, "experiment_config")
    units_by_pattern = _read_case_echo_map(script_text, "experiment_units")
    priority_by_id = _read_case_echo_map(script_text, "experiment_priority")
    notes_by_id = _read_case_echo_map(script_text, "experiment_notes")
    experiments_by_lane = _read_case_echo_map(script_text, "lane_experiments")

    rows: list[dict[str, Any]] = []
    for lane in sorted(experiments_by_lane, key=_lane_sort_key):
        for experiment_id in experiments_by_lane[lane].split():
            kind = kind_by_id.get(experiment_id, "")
            rows.append(
                {
                    "lane": lane,
                    "experiment_id": experiment_id,
                    "kind": kind,
                    "curriculum_kind": _curriculum_kind(kind),
                    "config_path": config_by_id.get(experiment_id, ""),
                    "expected_run_name": experiment_id,
                    "unit_count": _lookup_case_value(units_by_pattern, experiment_id),
                    "priority": priority_by_id.get(experiment_id, ""),
                    "notes": notes_by_id.get(experiment_id, ""),
                }
            )
    return _normalize_matrix_rows(rows)


def _normalize_matrix_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Add report-facing derived fields to planned matrix rows."""
    for row in rows:
        row["lane"] = _int_or_none(row.get("lane"))
        row["unit_count"] = _int_or_none(row.get("unit_count"))
        run_name = _row_run_name(row)
        row["run_name"] = run_name
        row["expected_run_name"] = str(row.get("expected_run_name") or run_name)
        row["action_interface"] = infer_action_interface(run_name)
        row["training_target"] = infer_training_target(run_name)
        row["ppo_variant"] = infer_ppo_variant(run_name)
        row["method"] = infer_method_label(str(row.get("kind") or ""))
    return rows


def _read_case_echo_map(script_text: str, function_name: str) -> dict[str, str]:
    """Extract simple ``case`` entries that echo a quoted string."""
    function_match = re.search(rf"^{re.escape(function_name)}\(\) \{{\n(?P<body>.*?)^\}}", script_text, re.MULTILINE | re.DOTALL)
    if function_match is None:
        return {}
    entries = re.findall(r'^\s*([^\n)]+)\)\s+echo\s+"([^"]*)"\s+;;', function_match.group("body"), re.MULTILINE)
    return {key.strip(): value for key, value in entries if key.strip() != "*"}


def _lookup_case_value(case_map: dict[str, str], key: str) -> str | None:
    """Resolve an exact or shell-pattern case value for a key."""
    if key in case_map:
        return case_map[key]
    for pattern, value in case_map.items():
        if fnmatch.fnmatchcase(key, pattern):
            return value
    return None


def _lane_sort_key(value: str) -> tuple[int, int | str]:
    """Sort numeric lane labels before any nonnumeric labels."""
    return (0, int(value)) if value.isdigit() else (1, value)


def _curriculum_kind(kind: str) -> str:
    """Return the compact curriculum kind used by runner summaries."""
    return {"manual_curriculum": "manual", "llm_curriculum": "llm"}.get(kind, "")


def artifact_root(root: str | Path | None = None) -> Path:
    """Return the artifact root, defaulting to direct run folders under /workspace/storage/runs."""
    return DEFAULT_ARTIFACT_ROOT if root is None else Path(root).expanduser()


def find_default_runs_root() -> Path:
    """Find the most likely local runs root for notebook/report metrics."""
    for candidate in (Path("storage/runs"), Path("../storage/runs"), DEFAULT_ARTIFACT_ROOT):
        if candidate.is_dir():
            return candidate
    return DEFAULT_ARTIFACT_ROOT


def build_report_metric_table(root: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Build normalized final-report metric rows from existing run artifacts.

    Parameters
    ----------
    root
        Runs root to scan recursively. When omitted, common local roots such as
        ``storage/runs`` and ``/workspace/storage/runs`` are checked.

    Returns
    -------
    list[dict[str, Any]]
        Pandas-friendly rows with stable comparison columns. Invalid JSON,
        manifests, indexes and metrics files without scalar report metrics are skipped.

    """
    resolved_root = find_default_runs_root() if root is None else Path(root).expanduser()
    rows: list[dict[str, Any]] = []
    for path in _iter_report_metric_files(resolved_root):
        payload = _read_json_mapping(path)
        if payload is None or not _has_report_metric_fields(payload):
            continue
        rows.append(_report_metric_row(payload=payload, path=path, root=resolved_root))
    return sorted(rows, key=_report_sort_key)


def compact_report_columns() -> tuple[str, ...]:
    """Return the default compact columns for notebook report display."""
    return COMPACT_REPORT_METRIC_COLUMNS


def compact_report_metric_table(
    root: str | Path | None = None,
    *,
    columns: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a compact report metric table with only display-friendly columns.

    Parameters
    ----------
    root
        Runs root to scan recursively. When omitted, ``find_default_runs_root`` is used.
    columns
        Optional output columns. Missing columns are ignored rather than raising.

    Returns
    -------
    list[dict[str, Any]]
        Compact rows suitable for notebook display.

    """
    selected_columns = compact_report_columns() if columns is None else columns
    return [{column: row.get(column) for column in selected_columns if column in row} for row in build_report_metric_table(root=root)]


def build_aggregated_report_metric_table(root: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Build one fixed/generalization comparison row per run.

    Parameters
    ----------
    root
        Runs root to scan recursively. When omitted, ``find_default_runs_root`` is used.

    Returns
    -------
    list[dict[str, Any]]
        Run-level rows where numeric evaluation metrics are averaged over fixed or
        generalization task rows, termination counts are summed and failures are summarized.

    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in build_report_metric_table(root=root):
        if not _is_generalization_report_metric_row(row):
            continue
        key = tuple(row.get(column) for column in _AGGREGATED_REPORT_GROUP_COLUMNS)
        groups.setdefault(key, []).append(row)

    table: list[dict[str, Any]] = []
    for key, candidate_rows in groups.items():
        rows = _final_model_report_rows(candidate_rows)
        aggregate = dict.fromkeys(AGGREGATED_REPORT_METRIC_COLUMNS)
        aggregate.update(dict(zip(_AGGREGATED_REPORT_GROUP_COLUMNS, key, strict=True)))
        aggregate["evaluated_task_count"] = len(rows)
        for column in _AGGREGATED_REPORT_MEAN_COLUMNS:
            aggregate[column] = _mean_numeric(row.get(column) for row in rows)
        for column in _AGGREGATED_REPORT_COUNT_COLUMNS:
            aggregate[column] = _sum_numeric_counts(row.get(column) for row in rows)
        aggregate["failure_status"] = _unique_summary(row.get("failure_status") for row in rows)
        aggregate["primary_failure"] = _unique_summary(row.get("primary_failure") for row in rows)
        table.append(aggregate)
    return sorted(table, key=_aggregated_report_sort_key)


def compact_aggregated_report_columns() -> tuple[str, ...]:
    """Return the default compact columns for aggregated notebook report display."""
    return COMPACT_AGGREGATED_REPORT_METRIC_COLUMNS


def compact_aggregated_report_metric_table(
    root: str | Path | None = None,
    *,
    columns: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a compact aggregated report table with display-friendly columns.

    Parameters
    ----------
    root
        Runs root to scan recursively. When omitted, ``find_default_runs_root`` is used.
    columns
        Optional output columns. Missing columns are ignored rather than raising.

    Returns
    -------
    list[dict[str, Any]]
        Compact one-row-per-run metric summaries for notebook display.

    """
    selected_columns = compact_aggregated_report_columns() if columns is None else columns
    return [{column: row.get(column) for column in selected_columns if column in row} for row in build_aggregated_report_metric_table(root=root)]


def build_aggregated_scenario_metric_table(root: str | Path | None = None) -> list[dict[str, Any]]:
    """
    Build one show/OOD scenario comparison row per run.

    Parameters
    ----------
    root
        Runs root to scan recursively. When omitted, ``find_default_runs_root`` is used.

    Returns
    -------
    list[dict[str, Any]]
        Run-level rows where numeric scenario metrics are averaged over show/OOD
        scenario rows, termination counts are summed and failures are summarized.

    """
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in build_report_metric_table(root=root):
        if not _is_scenario_report_metric_row(row):
            continue
        key = tuple(row.get(column) for column in _AGGREGATED_SCENARIO_GROUP_COLUMNS)
        groups.setdefault(key, []).append(row)

    table: list[dict[str, Any]] = []
    for key, candidate_rows in groups.items():
        rows = _final_model_report_rows(candidate_rows)
        aggregate = dict.fromkeys(AGGREGATED_SCENARIO_REPORT_METRIC_COLUMNS)
        aggregate.update(dict(zip(_AGGREGATED_SCENARIO_GROUP_COLUMNS, key, strict=True)))
        aggregate["evaluated_scenario_count"] = len(rows)
        for column in _AGGREGATED_SCENARIO_REPORT_MEAN_COLUMNS:
            aggregate[column] = _mean_numeric(row.get(column) for row in rows)
        for column in _AGGREGATED_REPORT_COUNT_COLUMNS:
            aggregate[column] = _sum_numeric_counts(row.get(column) for row in rows)
        aggregate["failure_status"] = _unique_summary(row.get("failure_status") for row in rows)
        aggregate["primary_failure"] = _unique_summary(row.get("primary_failure") for row in rows)
        table.append(aggregate)
    return sorted(table, key=_aggregated_report_sort_key)


def compact_aggregated_scenario_columns() -> tuple[str, ...]:
    """Return the default compact columns for aggregated scenario report display."""
    return COMPACT_AGGREGATED_SCENARIO_REPORT_METRIC_COLUMNS


def compact_aggregated_scenario_metric_table(
    root: str | Path | None = None,
    *,
    columns: tuple[str, ...] | None = None,
) -> list[dict[str, Any]]:
    """
    Build a compact show/OOD scenario report table with display-friendly columns.

    Parameters
    ----------
    root
        Runs root to scan recursively. When omitted, ``find_default_runs_root`` is used.
    columns
        Optional output columns. Missing columns are ignored rather than raising.

    Returns
    -------
    list[dict[str, Any]]
        Compact one-row-per-run scenario metric summaries for notebook display.

    """
    selected_columns = compact_aggregated_scenario_columns() if columns is None else columns
    return [{column: row.get(column) for column in selected_columns if column in row} for row in build_aggregated_scenario_metric_table(root=root)]


def summarize_run_artifacts(
    root: str | Path | None = None,
    *,
    matrix_rows: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """
    Summarize planned run folders and local artifact availability.

    Parameters
    ----------
    root
        Artifact root containing direct ``<run_name>/`` folders. Defaults to ``/workspace/storage/runs``.
    matrix_rows
        Optional preloaded planned experiment matrix rows.

    Returns
    -------
    list[dict[str, Any]]
        One row per planned run with explicit missing/incomplete statuses.

    """
    rows = load_experiment_matrix() if matrix_rows is None else matrix_rows
    resolved_root = artifact_root(root)
    summaries: list[dict[str, Any]] = []
    for row in rows:
        run_name = _row_run_name(row)
        run_root = resolved_root / run_name
        manifest_path = run_root / RUN_MANIFEST_FILENAME
        metrics = list(_iter_metric_files(run_root)) if run_root.exists() else []
        media = list(_iter_media_files(run_root)) if run_root.exists() else []
        summaries.append(
            {
                "run_name": run_name,
                "method": row.get("method") or infer_method_label(str(row.get("kind") or "")),
                "action_interface": row.get("action_interface") or infer_action_interface(run_name),
                "training_target": row.get("training_target") or infer_training_target(run_name),
                "ppo_variant": row.get("ppo_variant") or infer_ppo_variant(run_name),
                "run_root": str(run_root),
                "artifact_status": _run_artifact_status(run_root, manifest_path, metrics),
                "manifest_path": str(manifest_path) if manifest_path.exists() else None,
                "metrics_file_count": len(metrics),
                "media_file_count": len(media),
            }
        )
    return summaries


def find_metric_artifacts(root: str | Path | None = None, *, run_name: str | None = None, max_items: int = 100) -> list[dict[str, Any]]:
    """
    Find metrics JSON files under direct run-name folders.

    Parameters
    ----------
    root
        Artifact root containing direct ``<run_name>/`` folders. Defaults to ``/workspace/storage/runs``.
    run_name
        Optional run directory name to restrict the search.
    max_items
        Maximum number of metrics files to report.

    Returns
    -------
    list[dict[str, Any]]
        Metrics artifact rows. Missing roots return an empty list.

    """
    resolved_root = artifact_root(root)
    search_roots = [resolved_root / run_name] if run_name else _direct_run_dirs(resolved_root)
    rows: list[dict[str, Any]] = []
    for search_root in search_roots:
        for path in _iter_metric_files(search_root):
            if len(rows) >= max_items:
                return rows
            rows.append(_artifact_row(path=path, root=resolved_root, artifact_key="metric_file"))
    return rows


def load_metric_records(
    root: str | Path | None = None,
    *,
    run_name: str | None = None,
    metric_keys: tuple[str, ...] = REPORT_METRIC_KEYS,
    max_items: int = 100,
) -> list[dict[str, Any]]:
    """
    Load selected scalar values from generated metrics JSON files.

    Parameters
    ----------
    root
        Artifact root containing direct ``<run_name>/`` folders. Defaults to ``/workspace/storage/runs``.
    run_name
        Optional run directory name to restrict the search.
    metric_keys
        Metric keys to copy when present as scalar JSON values.
    max_items
        Maximum number of metrics files to read.

    Returns
    -------
    list[dict[str, Any]]
        One row per metrics file. Invalid JSON files are reported without raising.

    """
    records: list[dict[str, Any]] = []
    for artifact in find_metric_artifacts(root=root, run_name=run_name, max_items=max_items):
        path = Path(str(artifact["path"]))
        payload = _read_json_mapping(path)
        record = {
            "run_name": artifact.get("run_name"),
            "metric_file": artifact.get("metric_file"),
            "path": artifact.get("path"),
            "path_relative_to_root": artifact.get("path_relative_to_root"),
        }
        if payload is None:
            records.append({**record, "artifact_status": "invalid_metrics_json"})
            continue
        record["artifact_status"] = "available"
        for key in metric_keys:
            if key not in payload:
                continue
            value = payload[key]
            if _is_scalar_json_value(value):
                record[key] = value
        records.append(record)
    return records


def build_metric_comparison_table(
    root: str | Path | None = None,
    *,
    matrix_rows: list[dict[str, Any]] | None = None,
    max_items: int = 100,
) -> list[dict[str, Any]]:
    """
    Build a compact run comparison table from available metrics JSON files.

    Parameters
    ----------
    root
        Artifact root containing direct ``<run_name>/`` folders. Defaults to ``/workspace/storage/runs``.
    matrix_rows
        Optional preloaded planned experiment matrix rows.
    max_items
        Maximum number of metrics files to read.

    Returns
    -------
    list[dict[str, Any]]
        Metrics rows enriched with planned method/action/target labels and sorted by the clearest available error metric.

    """
    rows = load_experiment_matrix() if matrix_rows is None else matrix_rows
    planned = {_row_run_name(row): row for row in rows}
    table: list[dict[str, Any]] = []
    for record in load_metric_records(root=root, max_items=max_items):
        run_name = str(record.get("run_name") or "")
        plan = planned.get(run_name, {})
        table.append(
            {
                "run_name": run_name,
                "method": plan.get("method") or infer_method_label(str(plan.get("kind") or "")),
                "action_interface": plan.get("action_interface") or infer_action_interface(run_name),
                "training_target": plan.get("training_target") or infer_training_target(run_name),
                "ppo_variant": plan.get("ppo_variant") or infer_ppo_variant(run_name),
                "evaluation_name": record.get("evaluation_name") or record.get("evaluation_suite_name") or record.get("suite_task_name"),
                "mean_tracking_error": _first_present(record, ("mean_position_error_tracking_m", "mean_position_error_m")),
                "final_error": record.get("final_position_error_m"),
                "max_error": record.get("max_position_error_m"),
                "termination_status": record.get("failure_overall_status") or record.get("failure_primary_mode"),
                "metric_file": record.get("metric_file"),
                "artifact_status": record.get("artifact_status"),
                "path_relative_to_root": record.get("path_relative_to_root"),
            }
        )
    return sorted(table, key=_comparison_sort_key)


def find_media_artifacts(root: str | Path | None = None, *, run_name: str | None = None, max_items: int = 40) -> list[dict[str, Any]]:
    """
    Find plot, GIF, and video files under direct run-name folders.

    Parameters
    ----------
    root
        Artifact root containing direct ``<run_name>/`` folders. Defaults to ``/workspace/storage/runs``.
    run_name
        Optional run directory name to restrict the search.
    max_items
        Maximum number of media files to report.

    Returns
    -------
    list[dict[str, Any]]
        Media artifact rows. Missing roots return an empty list.

    """
    resolved_root = artifact_root(root)
    search_roots = [resolved_root / run_name] if run_name else _direct_run_dirs(resolved_root)
    rows: list[dict[str, Any]] = []
    for search_root in search_roots:
        for path in _iter_media_files(search_root):
            if len(rows) >= max_items:
                return rows
            rows.append(_artifact_row(path=path, root=resolved_root, artifact_key="media_file"))
    return rows


def infer_method_label(kind: str) -> str:
    """Return a report-facing method label for an experiment kind or run name."""
    return _known_method_label(kind) or kind or "unknown"


def infer_action_interface(text: str) -> str:
    """Infer the action interface from a run name or config path."""
    normalized = text.lower()
    if "directrpm" in normalized or "direct_rpm" in normalized:
        return "direct_rpm"
    if "pid" in normalized:
        return "pid_position"
    return "unknown"


def infer_training_target(text: str) -> str:
    """Infer the main training target label from a run name or config path."""
    normalized = text.lower()
    if "manual_curriculum" in normalized or "curriculum_manual" in normalized:
        return "manual_curriculum"
    if "llm_curriculum" in normalized or "curriculum_llm" in normalized:
        return "llm_curriculum"
    if "basic_show" in normalized or "basic_training_show" in normalized:
        return "direct_basic_show"
    if "m-taskdist_medium" in normalized or "taskdist_medium" in normalized or "tracking_medium" in normalized:
        return "m-taskdist_medium"
    return "configured_task"


def infer_ppo_variant(run_name: str) -> str:
    """Infer the compact PPO variant label from a planned run name or metric field."""
    normalized = run_name.lower()
    if "basic_show" in normalized or "basic_training_show" in normalized:
        return "basic_show"
    if "net256" in normalized:
        return "net256"
    if "low_lr" in normalized or "low-lr" in normalized:
        return "low_lr"
    if "gamma095" in normalized or "gamma_0.95" in normalized or "gamma=0.95" in normalized:
        return "gamma095"
    if "smooth001" in normalized or "smooth_0.01" in normalized or "smooth=0.01" in normalized:
        return "smooth001"
    if "ent005" in normalized or "ent_coef_0.005" in normalized or "ent=0.005" in normalized:
        return "ent005"
    if "clip010" in normalized or "clip_range_0.10" in normalized or "clip=0.10" in normalized:
        return "clip010"
    if "targetkl015" in normalized or "target_kl_0.015" in normalized or "targetkl=0.015" in normalized:
        return "targetkl015"
    return "default"


def _known_method_label(text: str) -> str | None:
    """Infer a known report method label from compact metadata or path text."""
    normalized = text.lower().replace("-", "_")
    if normalized in {"manual", "manual_curriculum"} or "manual_curriculum" in normalized or "curriculum_manual" in normalized:
        return "Manual curriculum"
    if normalized in {"llm", "llm_curriculum"} or "llm_curriculum" in normalized or "curriculum_llm" in normalized:
        return "LLM curriculum"
    if normalized in {"direct", "direct_ppo"} or "direct_ppo" in normalized:
        return "Direct PPO"
    return None


def _iter_report_metric_files(root: Path) -> tuple[Path, ...]:
    """Return candidate report metrics files below a runs root."""
    if not root.exists():
        return ()
    return tuple(sorted(path for path in root.rglob(METRICS_GLOB) if path.is_file() and not _is_index_or_manifest_file(path)))


def _is_index_or_manifest_file(path: Path) -> bool:
    """Return whether a JSON path is clearly an index or manifest artifact."""
    name = path.name.lower()
    return "index" in name or "manifest" in name


def _has_report_metric_fields(payload: dict[str, Any]) -> bool:
    """Return whether a metrics payload has at least one scalar report metric."""
    return any(key in payload and _is_scalar_json_value(payload[key]) for key in _REPORT_USABLE_METRIC_KEYS)


def _report_metric_row(payload: dict[str, Any], path: Path, root: Path) -> dict[str, Any]:
    """Normalize one metrics payload into the final-report comparison schema."""
    artifact = _artifact_row(path=path, root=root, artifact_key="metrics_file_name")
    raw_run_name = _first_present(payload, ("source_run_name", "run_name", "training_run_name", "model_run_name"))
    run_name = _report_display_run_name(
        raw_run_name=str(raw_run_name or ""),
        artifact_run_name=str(artifact.get("run_name") or ""),
        payload=payload,
        path=path,
    )
    context = _report_context(payload=payload, path=path, run_name=run_name)
    row = dict.fromkeys(REPORT_METRIC_OUTPUT_COLUMNS)
    row.update(
        {
            "run_name": run_name,
            "method": _report_method_label(payload=payload, context=context),
            "action_interface": _report_action_interface(payload=payload, context=context),
            "variant": _report_variant_label(payload=payload, context=context),
            "training_target": infer_training_target(context),
            "evaluation_name": _report_evaluation_name(payload=payload, path=path),
            "suite_task_name": _report_suite_task_name(payload),
            "task_shape": _first_present(payload, _REPORT_TASK_SHAPE_KEYS),
            "metrics_file": str(path),
        }
    )
    for column, keys in _REPORT_METRIC_FIELD_SOURCES.items():
        value = _first_present(payload, keys)
        row[column] = _count_value(value) if column in {"terminated_count", "truncated_count", *_REPORT_STEP_COUNT_COLUMNS} else value
    row.update(_report_completion_fields(payload=payload, row=row, metrics_path=path))
    return row


def _report_completion_fields(payload: dict[str, Any], row: dict[str, Any], metrics_path: Path) -> dict[str, Any]:
    """Return normalized completion fields, inferring from diagnostics when possible."""
    fields = {column: row.get(column) for column in _REPORT_COMPLETION_COLUMNS}
    inferred = _infer_completion_fields_from_artifacts(payload=payload, metrics_path=metrics_path)
    for column, value in inferred.items():
        if fields.get(column) is None:
            fields[column] = value

    completed_tracking_steps = _count_value(fields.get("completed_tracking_steps"))
    planned_tracking_steps = _count_value(fields.get("planned_tracking_steps"))
    completion_ratio = _as_float(fields.get("completion_ratio"))
    if completion_ratio is None:
        completion_ratio = _completion_ratio(completed=completed_tracking_steps, planned=planned_tracking_steps)

    mean_tracking_error_m = row.get("mean_tracking_error_m")
    native_tracking_error = _first_present(payload, ("mean_tracking_error_m", "mean_position_error_tracking_m"))
    inferred_tracking_error = _as_float(inferred.get("mean_tracking_error_m"))
    extra_fields: dict[str, Any] = {}
    if native_tracking_error is None and inferred_tracking_error is not None:
        mean_tracking_error_m = inferred_tracking_error
        extra_fields["mean_tracking_error_m"] = inferred_tracking_error

    completion_adjusted_tracking_error_m = _as_float(fields.get("completion_adjusted_tracking_error_m"))
    if completion_adjusted_tracking_error_m is None:
        completion_adjusted_tracking_error_m = _completion_adjusted_tracking_error(
            mean_tracking_error_m=mean_tracking_error_m,
            completion_ratio=completion_ratio,
        )

    completed_rollout_steps = _count_value(fields.get("completed_rollout_steps"))
    planned_rollout_steps = _count_value(fields.get("planned_rollout_steps"))
    rollout_completion_ratio = _as_float(fields.get("rollout_completion_ratio"))
    if rollout_completion_ratio is None:
        rollout_completion_ratio = _completion_ratio(completed=completed_rollout_steps, planned=planned_rollout_steps)

    return {
        **extra_fields,
        "completed_tracking_steps": completed_tracking_steps,
        "planned_tracking_steps": planned_tracking_steps,
        "completion_ratio": completion_ratio,
        "completion_adjusted_tracking_error_m": completion_adjusted_tracking_error_m,
        "completed_rollout_steps": completed_rollout_steps,
        "planned_rollout_steps": planned_rollout_steps,
        "rollout_completion_ratio": rollout_completion_ratio,
    }


def _infer_completion_fields_from_artifacts(payload: dict[str, Any], metrics_path: Path) -> dict[str, Any]:
    """Infer completion fields from trace or episode-summary diagnostics when available."""
    eval_steps = _count_value(_first_present(payload, ("eval_steps", "planned_rollout_steps", "effective_max_steps", "requested_max_steps")))
    for default_filename in (EVALUATION_TRACE_FILENAME, SCENARIO_TRACE_FILENAME):
        for trace_path in _candidate_diagnostic_paths(
            payload=payload,
            metrics_path=metrics_path,
            path_keys=("evaluation_trace_path", "trace_path"),
            default_filename=default_filename,
        ):
            records = _read_trace_records(trace_path)
            if records:
                return _completion_fields_from_trace(records=records, eval_steps=eval_steps, payload=payload)

    for summaries_path in _candidate_diagnostic_paths(
        payload=payload,
        metrics_path=metrics_path,
        path_keys=("episode_summaries_path",),
        default_filename=EPISODE_SUMMARIES_FILENAME,
    ):
        summaries = _read_json_list(summaries_path)
        if summaries:
            return _completion_fields_from_episode_summaries(summaries)
    return {}


def _candidate_diagnostic_paths(
    payload: dict[str, Any],
    metrics_path: Path,
    path_keys: tuple[str, ...],
    default_filename: str,
) -> tuple[Path, ...]:
    """Return existing diagnostic artifact candidates related to one metrics file."""
    candidates: list[Path] = []
    for key in path_keys:
        value = payload.get(key)
        if value:
            candidates.extend(_resolve_artifact_path_candidates(value=value, metrics_path=metrics_path))
    manifest_payloads = _candidate_manifest_payloads(payload=payload, metrics_path=metrics_path)
    for manifest_payload in manifest_payloads:
        for key in path_keys:
            value = manifest_payload.get(key)
            if value:
                candidates.extend(_resolve_artifact_path_candidates(value=value, metrics_path=metrics_path))
        for directory_key in ("diagnostics_dir", "traces_dir"):
            value = manifest_payload.get(directory_key)
            if value:
                candidates.extend(
                    directory / default_filename for directory in _resolve_artifact_path_candidates(value=value, metrics_path=metrics_path)
                )
    diagnostics_dir = payload.get("diagnostics_dir")
    if diagnostics_dir:
        candidates.extend(
            directory / default_filename for directory in _resolve_artifact_path_candidates(value=diagnostics_dir, metrics_path=metrics_path)
        )
    traces_dir = payload.get("traces_dir")
    if traces_dir:
        candidates.extend(
            directory / default_filename for directory in _resolve_artifact_path_candidates(value=traces_dir, metrics_path=metrics_path)
        )
    candidates.append(metrics_path.parent.parent / "diagnostics" / default_filename)
    candidates.append(metrics_path.parent.parent / "traces" / default_filename)

    existing: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = candidate.expanduser().resolve(strict=False).as_posix()
        if key in seen or not candidate.is_file():
            continue
        seen.add(key)
        existing.append(candidate)
    return tuple(existing)


def _candidate_manifest_payloads(payload: dict[str, Any], metrics_path: Path) -> tuple[dict[str, Any], ...]:
    """Return existing manifest JSON payloads referenced by a metrics payload."""
    manifests: list[dict[str, Any]] = []
    for key in ("manifest_path", "evaluation_manifest_path", "source_manifest_path"):
        value = payload.get(key)
        if not value:
            continue
        for candidate in _resolve_artifact_path_candidates(value=value, metrics_path=metrics_path):
            manifest = _read_json_mapping(candidate)
            if manifest:
                manifests.append(manifest)
    return tuple(manifests)


def _resolve_artifact_path_candidates(value: Any, metrics_path: Path) -> list[Path]:
    """Return plausible paths for an absolute or relative artifact field."""
    raw_path = Path(str(value)).expanduser()
    if raw_path.is_absolute():
        return [raw_path]
    return [metrics_path.parent / raw_path, metrics_path.parent.parent / raw_path, Path.cwd() / raw_path]


def _read_trace_records(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL evaluation trace, returning an empty list when unavailable."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            return []
        if isinstance(payload, dict):
            records.append(dict(payload))
    return records


def _read_json_list(path: Path) -> list[dict[str, Any]]:
    """Read a JSON list of objects, returning an empty list when unavailable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(payload, list):
        return []
    return [dict(item) for item in payload if isinstance(item, dict)]


def _read_json_mapping(path: Path) -> dict[str, Any]:
    """Read a JSON object, returning an empty dict when unavailable."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, dict) else {}


def _completion_fields_from_trace(records: list[dict[str, Any]], eval_steps: int | None, payload: dict[str, Any]) -> dict[str, Any]:
    """Infer completion fields from detailed trace records."""
    episode_records = _records_by_episode(records)
    completed_by_episode = [_completed_tracking_steps(episode) for episode in episode_records]
    completed_tracking_steps = (
        None if any(value is None for value in completed_by_episode) else int(sum(value for value in completed_by_episode if value is not None))
    )
    planned_by_episode = [_planned_tracking_steps(episode, payload=payload) for episode in episode_records]
    planned_tracking_steps = (
        None if any(value is None for value in planned_by_episode) else int(sum(value for value in planned_by_episode if value is not None))
    )
    completed_rollout_steps = len(records)
    planned_rollout_steps = eval_steps or _planned_rollout_steps_from_trace(records=records, payload=payload)
    tracking_records = [record for episode in episode_records for record in _tracking_records_for_mean(episode)]
    mean_tracking_error_m = _mean_trace_position_error(tracking_records)
    return {
        "mean_tracking_error_m": mean_tracking_error_m,
        "completed_tracking_steps": completed_tracking_steps,
        "planned_tracking_steps": planned_tracking_steps,
        "completion_ratio": _completion_ratio(completed=completed_tracking_steps, planned=planned_tracking_steps),
        "completed_rollout_steps": int(completed_rollout_steps),
        "planned_rollout_steps": planned_rollout_steps,
        "rollout_completion_ratio": _completion_ratio(completed=completed_rollout_steps, planned=planned_rollout_steps),
    }


def _completion_fields_from_episode_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    """Infer completion fields from episode summary diagnostics when present."""
    completed_tracking_steps = _sum_optional_counts(summary.get("completed_tracking_steps") for summary in summaries)
    planned_tracking_steps = _sum_optional_counts(summary.get("planned_tracking_steps") for summary in summaries)
    completed_rollout_steps = _sum_optional_counts(summary.get("completed_rollout_steps") for summary in summaries)
    planned_rollout_steps = _sum_optional_counts(summary.get("planned_rollout_steps") for summary in summaries)
    return {
        "completed_tracking_steps": completed_tracking_steps,
        "planned_tracking_steps": planned_tracking_steps,
        "completion_ratio": _completion_ratio(completed=completed_tracking_steps, planned=planned_tracking_steps),
        "completed_rollout_steps": completed_rollout_steps,
        "planned_rollout_steps": planned_rollout_steps,
        "rollout_completion_ratio": _completion_ratio(completed=completed_rollout_steps, planned=planned_rollout_steps),
    }


def _records_by_episode(records: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group trace rows by episode while preserving trace order."""
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(_count_value(record.get("episode_index")) or 0, []).append(record)
    return [grouped[index] for index in sorted(grouped)]


def _completed_tracking_steps(records: list[dict[str, Any]]) -> int | None:
    """Return the number of trace rows included in tracking-error metrics."""
    if not _has_tracking_phase_metadata(records):
        return None
    if _uses_tracking_phase_filter(records):
        return len([record for record in records if bool(record.get("is_tracking_phase", True))])
    return len(records)


def _tracking_records_for_mean(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return trace records aligned with the tracking-error phase definition."""
    if _uses_tracking_phase_filter(records):
        return [record for record in records if bool(record.get("is_tracking_phase", True))]
    return records if _has_tracking_phase_metadata(records) else []


def _planned_tracking_steps(records: list[dict[str, Any]], payload: dict[str, Any]) -> int | None:
    """Infer planned tracking rows from phase metadata."""
    if not records or not _has_tracking_phase_metadata(records):
        return None
    first = records[0]
    direct_value = _count_value(
        _first_present(
            first,
            ("planned_tracking_steps", "reference_motion_steps"),
        )
    ) or _count_value(_first_present(payload, ("planned_tracking_steps", "reference_motion_steps")))
    if direct_value is not None and direct_value > 0:
        return int(direct_value)
    if _uses_tracking_phase_filter(records):
        start_step = _count_value(first.get("tracking_phase_start_step")) or 0
        end_step = _count_value(first.get("tracking_phase_end_step"))
        return int(end_step - start_step) if end_step is not None and end_step > start_step else None
    end_step = _count_value(first.get("tracking_phase_end_step"))
    return int(end_step) if end_step is not None and end_step > 0 else None


def _planned_rollout_steps_from_trace(records: list[dict[str, Any]], payload: dict[str, Any]) -> int | None:
    """Infer planned rollout steps from trace or metrics metadata."""
    if records:
        first = records[0]
        value = _count_value(_first_present(first, ("planned_rollout_steps", "effective_max_steps", "eval_steps")))
        if value is not None and value > 0:
            return int(value)
    value = _count_value(_first_present(payload, ("planned_rollout_steps", "effective_max_steps", "eval_steps", "requested_max_steps")))
    return int(value) if value is not None and value > 0 else None


def _has_tracking_phase_metadata(records: list[dict[str, Any]]) -> bool:
    """Return whether trace rows expose enough metadata for tracking completion."""
    if not records:
        return False
    first = records[0]
    return any(
        key in first
        for key in (
            "planned_tracking_steps",
            "reference_motion_steps",
            "tracking_phase_end_step",
            "is_tracking_phase",
            "exclude_start_hold_from_tracking_metrics",
            "exclude_final_hold_from_tracking_metrics",
        )
    )


def _uses_tracking_phase_filter(records: list[dict[str, Any]]) -> bool:
    """Return whether existing tracking-error metrics filter to active tracking rows."""
    return any(
        "is_tracking_phase" in record
        or bool(record.get("exclude_start_hold_from_tracking_metrics", False))
        or bool(record.get("exclude_final_hold_from_tracking_metrics", False))
        for record in records
    )


def _mean_trace_position_error(trace_records: list[dict[str, Any]]) -> float | None:
    """Return the mean position error from trace rows when numeric values exist."""
    values = []
    for record in trace_records:
        value = _as_float(record.get("position_error_m", record.get("position_error")))
        if value is not None:
            values.append(value)
    return _mean_numeric(values)


def _sum_optional_counts(values: Any) -> int | None:
    """Sum count-like values, returning None if any value is missing."""
    counts = [_count_value(value) for value in values]
    if not counts or any(value is None for value in counts):
        return None
    return int(sum(value for value in counts if value is not None))


def _completion_ratio(completed: int | None, planned: int | None) -> float | None:
    """Return bounded completion fraction from completed and planned steps."""
    if completed is None or planned is None or planned <= 0:
        return None
    return float(min(1.0, max(0.0, completed / planned)))


def _completion_adjusted_tracking_error(mean_tracking_error_m: Any, completion_ratio: float | None) -> float | None:
    """Return tracking error divided by the completion-ratio floor."""
    mean_error = _as_float(mean_tracking_error_m)
    if mean_error is None or completion_ratio is None:
        return None
    return float(mean_error / max(completion_ratio, COMPLETION_ADJUSTMENT_MIN_RATIO))


def _report_display_run_name(raw_run_name: str, artifact_run_name: str, payload: dict[str, Any], path: Path) -> str:
    """Return the report-facing run name, collapsing curriculum stages to their parent run."""
    if _is_curriculum_stage_context(raw_run_name=raw_run_name, payload=payload, path=path):
        return artifact_run_name or _strip_curriculum_stage_suffix(raw_run_name) or raw_run_name
    return raw_run_name or artifact_run_name


def _is_curriculum_stage_context(raw_run_name: str, payload: dict[str, Any], path: Path) -> bool:
    """Return whether metrics describe a curriculum stage model or stage artifact."""
    run_kind = _first_present(payload, ("source_run_kind", "run_kind"))
    if run_kind == "curriculum_stage":
        return True
    if any(key in payload for key in ("source_stage", "stage_index", "curriculum_stage_index")):
        return True
    normalized_path = path.as_posix().lower()
    return "/stages/stage" in normalized_path or re.search(r"_stage\d+_", raw_run_name.lower()) is not None


def _strip_curriculum_stage_suffix(run_name: str) -> str:
    """Strip a ``_stageNN_<name>_seedN`` suffix from a curriculum stage run name."""
    match = re.match(r"^(?P<prefix>.+)_stage\d+_.+_seed(?P<seed>\d+)$", run_name)
    if match is None:
        return ""
    return f"{match.group('prefix')}_seed{match.group('seed')}"


def _report_context(payload: dict[str, Any], path: Path, run_name: str) -> str:
    """Build a deterministic text context for label inference."""
    context_values = [
        run_name,
        path.as_posix(),
        payload.get("source_config_path"),
        payload.get("training_config_path"),
        payload.get("task_config_path"),
    ]
    return " ".join(str(value) for value in context_values if value)


def _report_method_label(payload: dict[str, Any], context: str) -> str:
    """Return the best method label from metrics metadata or path context."""
    curriculum_kind = _first_present(payload, ("source_curriculum_kind", "curriculum_kind"))
    run_kind = _first_present(payload, ("source_run_kind", "run_kind", "model_role"))
    if curriculum_kind and str(run_kind or "") in {"curriculum", "curriculum_stage", "baseline"}:
        known = _known_method_label(f"{curriculum_kind}_curriculum")
        if known is not None:
            return known
    if run_kind:
        known = _known_method_label(str(run_kind))
        if known is not None:
            return known
    return _known_method_label(context) or "unknown"


def _report_action_interface(payload: dict[str, Any], context: str) -> str:
    """Return the action interface from metrics metadata or path context."""
    action_interface = payload.get("action_interface")
    if isinstance(action_interface, str) and action_interface:
        return infer_action_interface(action_interface)
    return infer_action_interface(context)


def _report_variant_label(payload: dict[str, Any], context: str) -> str:
    """Return a compact PPO variant from metrics metadata or path context."""
    for key in ("variant", "ppo_profile", "ppo_variant"):
        value = payload.get(key)
        if isinstance(value, str) and value:
            inferred = infer_ppo_variant(value)
            if inferred != "default" or value.lower() in {"default", "baseline"}:
                return inferred
    return infer_ppo_variant(context)


def _report_evaluation_name(payload: dict[str, Any], path: Path) -> Any:
    """Return the evaluation name from metrics metadata or artifact layout."""
    value = _first_present(payload, ("evaluation_name", "evaluation_suite_name", "evaluation_suite", "evaluated_task_name"))
    if value:
        return value
    path_value = _path_component_after(path, "evaluations")
    if path_value:
        return path_value
    if "training" in path.parts:
        return "training"
    return None


def _report_suite_task_name(payload: dict[str, Any]) -> Any:
    """Return the suite task or scenario name when present."""
    return _first_present(payload, ("suite_task_name", "evaluated_task_name", "scenario_label", "scenario_name"))


def _path_component_after(path: Path, component: str) -> str | None:
    """Return the path part immediately after a named component."""
    parts = path.parts
    try:
        index = parts.index(component)
    except ValueError:
        return None
    next_index = index + 1
    return parts[next_index] if next_index < len(parts) else None


def _count_value(value: Any) -> int | None:
    """Convert count-like metric values to integers."""
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and isfinite(value):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _report_sort_key(row: dict[str, Any]) -> tuple[str, str, str, str]:
    """Sort report rows deterministically by run, evaluation, task and metrics path."""
    return (
        str(row.get("run_name") or ""),
        str(row.get("evaluation_name") or ""),
        str(row.get("suite_task_name") or ""),
        str(row.get("metrics_file") or ""),
    )


def _is_generalization_report_metric_row(row: dict[str, Any]) -> bool:
    """Return whether a detailed report row belongs in the fixed/generalization aggregate."""
    return _infer_report_evaluation_category(row) in {"fixed", "generalization"} and _has_suite_task_name(row)


def _is_scenario_report_metric_row(row: dict[str, Any]) -> bool:
    """Return whether a detailed report row belongs in the show/OOD scenario aggregate."""
    return _infer_report_evaluation_category(row) == "scenario" and _has_suite_task_name(row)


def _final_model_report_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only final-stage rows when a grouped model has staged evaluations."""
    stage_indexes = [_report_stage_index(row) for row in rows]
    known_stage_indexes = [stage_index for stage_index in stage_indexes if stage_index is not None]
    if not known_stage_indexes:
        return rows
    final_stage_index = max(known_stage_indexes)
    return [row for row, stage_index in zip(rows, stage_indexes, strict=True) if stage_index == final_stage_index]


def _has_suite_task_name(row: dict[str, Any]) -> bool:
    """Return whether a detailed metrics row represents one suite task."""
    suite_task_name = row.get("suite_task_name")
    return isinstance(suite_task_name, str) and bool(suite_task_name) and suite_task_name != "own_task"


def _infer_report_evaluation_category(row: dict[str, Any]) -> str:
    """Infer the report evaluation category from row labels and artifact paths."""
    text = _row_search_text(row, ("evaluation_name", "evaluation_suite_name", "suite_task_name", "metrics_file")).lower().replace("\\", "/")
    evaluation_name = str(row.get("evaluation_name") or "").lower()
    suite_task_name = str(row.get("suite_task_name") or "").lower()
    metrics_file = str(row.get("metrics_file") or "").lower().replace("\\", "/")
    if not text:
        return "unknown"
    if "own_task" in text:
        return "own_task"
    if "policy_render" in text or "render" in text:
        return "render"
    if "training" in text and "/evaluations/" not in text:
        return "training"
    if "generalization" in text:
        return "generalization"
    if "fixed" in text or "benchmark" in text or "line_eval" in text:
        return "fixed"
    if "scenario" in evaluation_name or "scenario" in suite_task_name or "/scenarios/" in metrics_file:
        return "scenario"
    if evaluation_name.startswith("show") or suite_task_name.startswith("show") or "/evaluations/show" in metrics_file:
        return "scenario"
    if "/evaluations/" in text and "/metrics/" in text and row.get("suite_task_name"):
        return "fixed"
    return "unknown"


def _report_stage_index(row: dict[str, Any]) -> int | None:
    """Infer a curriculum stage index from a report row path or run name."""
    text = _row_search_text(row, ("metrics_file", "run_name")).lower().replace("\\", "/")
    match = re.search(r"(?:/|_)stage(?P<stage>\d+)", text)
    if match is None:
        return None
    return int(match.group("stage"))


def _row_search_text(row: dict[str, Any], keys: tuple[str, ...]) -> str:
    """Return lower-level row text used for deterministic category inference."""
    return " ".join(str(row.get(key) or "") for key in keys).strip()


def _mean_numeric(values: Any) -> float | None:
    """Return the mean of finite numeric values, ignoring missing values."""
    numeric_values = [_as_float(value) for value in values]
    numeric_values = [value for value in numeric_values if value is not None]
    if not numeric_values:
        return None
    return float(sum(numeric_values) / len(numeric_values))


def _sum_numeric_counts(values: Any) -> int | None:
    """Return the sum of count-like values, or None when no counts are present."""
    counts = [_count_value(value) for value in values]
    counts = [value for value in counts if value is not None]
    if not counts:
        return None
    return int(sum(counts))


def _unique_summary(values: Any) -> str | None:
    """Return a compact deterministic summary of unique non-empty values."""
    unique_values = sorted({str(value) for value in values if value not in {None, ""}})
    if not unique_values:
        return None
    return ", ".join(unique_values)


def _aggregated_report_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    """Sort aggregated rows by completion-aware stability, then display labels."""
    adjusted_error = _as_float(row.get("completion_adjusted_tracking_error_m"))
    completion_ratio = _as_float(row.get("completion_ratio"))
    terminated_count = _count_value(row.get("terminated_count"))
    truncated_count = _count_value(row.get("truncated_count"))
    mean_tracking_error = _as_float(row.get("mean_tracking_error_m"))
    return (
        1 if adjusted_error is None else 0,
        float("inf") if adjusted_error is None else adjusted_error,
        -(completion_ratio if completion_ratio is not None else -1.0),
        _sort_count(terminated_count),
        _sort_count(truncated_count),
        1 if mean_tracking_error is None else 0,
        float("inf") if mean_tracking_error is None else mean_tracking_error,
        str(row.get("method") or ""),
        str(row.get("action_interface") or ""),
        str(row.get("training_target") or ""),
        str(row.get("variant") or ""),
        str(row.get("run_name") or ""),
    )


def _sort_count(value: int | None) -> int:
    """Return a sortable count with missing values after known counts."""
    return 10**12 if value is None else int(value)


def _direct_run_dirs(root: Path) -> list[Path]:
    """Return direct child directories under an artifact root."""
    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def _iter_metric_files(root: Path) -> tuple[Path, ...]:
    """Return metrics JSON files below a run root."""
    if not root.exists():
        return ()
    return tuple(sorted(path for path in root.rglob(METRICS_GLOB) if path.is_file()))


def _iter_media_files(root: Path) -> tuple[Path, ...]:
    """Return report-friendly media files below a run root."""
    if not root.exists():
        return ()
    return tuple(sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES))


def _artifact_row(path: Path, root: Path, artifact_key: str) -> dict[str, Any]:
    """Return a compact artifact row for a path below an artifact root."""
    relative = _relative_to_or_str(path, root)
    parts = Path(relative).parts
    return {
        "run_name": parts[0] if parts else None,
        artifact_key: path.name,
        "path": str(path),
        "path_relative_to_root": relative,
    }


def _run_artifact_status(run_root: Path, manifest_path: Path, metrics: list[Path]) -> str:
    """Return a concise artifact availability status for one planned run."""
    if not run_root.exists():
        return "missing_run_folder"
    if manifest_path.exists():
        return "available"
    if metrics:
        return "metrics_without_manifest"
    return "incomplete_run_folder"


def _row_run_name(row: dict[str, Any]) -> str:
    """Return the planned run name encoded by a matrix row."""
    return str(row.get("expected_run_name") or row.get("run_name") or row.get("experiment_id") or "")


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning None for absent or unreadable artifacts."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, dict) else None


def _first_present(row: dict[str, Any], keys: tuple[str, ...]) -> Any:
    """Return the first present non-None value for the given keys."""
    for key in keys:
        value = row.get(key)
        if value is not None:
            return value
    return None


def _comparison_sort_key(row: dict[str, Any]) -> tuple[int, float, str]:
    """Sort rows by available mean tracking error while keeping missing metrics last."""
    metric = _as_float(row.get("mean_tracking_error"))
    if metric is None:
        return (1, 0.0, str(row.get("run_name") or ""))
    return (0, metric, str(row.get("run_name") or ""))


def _relative_to_or_str(path: Path, root: Path) -> str:
    """Return a root-relative path when possible."""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _is_scalar_json_value(value: Any) -> bool:
    """Return whether a value is safe to copy into a compact metrics table."""
    return value is None or isinstance(value, str | int | float | bool)


def _int_or_none(value: Any) -> int | None:
    """Convert a value to int, returning None when conversion is not meaningful."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float | None:
    """Convert finite numeric-looking values to float."""
    if value is None or isinstance(value, bool):
        return None
    try:
        resolved = float(value)
    except (TypeError, ValueError):
        return None
    if not isfinite(resolved):
        return None
    return resolved


__all__ = [
    "AGGREGATED_REPORT_METRIC_COLUMNS",
    "AGGREGATED_SCENARIO_REPORT_METRIC_COLUMNS",
    "COMPACT_AGGREGATED_REPORT_METRIC_COLUMNS",
    "COMPACT_AGGREGATED_SCENARIO_REPORT_METRIC_COLUMNS",
    "COMPACT_REPORT_METRIC_COLUMNS",
    "DEFAULT_ARTIFACT_ROOT",
    "MATRIX_SCRIPT_PATH",
    "MATRIX_TSV_PATH",
    "MEDIA_SUFFIXES",
    "METRICS_GLOB",
    "REPORT_METRIC_KEYS",
    "REPORT_METRIC_OUTPUT_COLUMNS",
    "RUN_MANIFEST_FILENAME",
    "artifact_root",
    "build_aggregated_report_metric_table",
    "build_aggregated_scenario_metric_table",
    "build_metric_comparison_table",
    "build_report_metric_table",
    "compact_aggregated_report_columns",
    "compact_aggregated_report_metric_table",
    "compact_aggregated_scenario_columns",
    "compact_aggregated_scenario_metric_table",
    "compact_report_columns",
    "compact_report_metric_table",
    "expected_run_names",
    "find_default_runs_root",
    "find_media_artifacts",
    "find_metric_artifacts",
    "infer_action_interface",
    "infer_method_label",
    "infer_ppo_variant",
    "infer_training_target",
    "load_experiment_matrix",
    "load_metric_records",
    "summarize_run_artifacts",
]
