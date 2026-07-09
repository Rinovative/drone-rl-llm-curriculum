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
MEDIA_SUFFIXES = (".gif", ".png", ".jpg", ".jpeg", ".mp4")
REPORT_METRIC_KEYS = (
    "evaluation_name",
    "evaluation_suite_name",
    "suite_task_name",
    "mean_position_error_tracking_m",
    "mean_position_error_m",
    "final_position_error_m",
    "max_position_error_m",
    "rmse_position_error_m",
    "success_rate",
    "crash_rate",
    "failure_overall_status",
    "failure_primary_mode",
    "eval_terminated_count",
    "eval_truncated_count",
    "episode_count",
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
    """Return a report-facing method label for an experiment kind."""
    return {
        "direct_ppo": "Direct PPO",
        "manual_curriculum": "Manual curriculum",
        "llm_curriculum": "LLM curriculum",
    }.get(kind, kind or "unknown")


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
    if "basic_show" in normalized or "basic_training_show" in normalized:
        return "basic_training_show"
    if "taskdist_medium" in normalized or "tracking_medium" in normalized or "m-taskdist_medium" in normalized:
        return "tracking_medium"
    return "configured_task"


def infer_ppo_variant(run_name: str) -> str:
    """Infer the PPO variant label from a planned run name."""
    if "net256" in run_name:
        return "net256"
    if "low_lr" in run_name:
        return "low_lr"
    if "gamma095" in run_name:
        return "gamma_0.95"
    if "ent005" in run_name:
        return "ent_coef_0.005"
    if "clip010" in run_name:
        return "clip_range_0.10"
    if "targetkl015" in run_name:
        return "target_kl_0.015"
    return "default"


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
    "DEFAULT_ARTIFACT_ROOT",
    "MATRIX_SCRIPT_PATH",
    "MATRIX_TSV_PATH",
    "MEDIA_SUFFIXES",
    "METRICS_GLOB",
    "REPORT_METRIC_KEYS",
    "RUN_MANIFEST_FILENAME",
    "artifact_root",
    "build_metric_comparison_table",
    "expected_run_names",
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
