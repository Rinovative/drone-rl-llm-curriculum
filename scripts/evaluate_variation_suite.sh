#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: bash scripts/evaluate_variation_suite.sh <run-manifest-path> [suite-config]" >&2
  exit 2
fi

RUN_MANIFEST="$1"
SUITE_CONFIG="${2:-configs/evaluation/evaluation_task_suite_variation.yaml}"
PYTHON_BIN="${PYTHON_BIN:-python}"
TMP_ROOT="$(mktemp -d "${TMPDIR:-/tmp}/drone_eval_suite.XXXXXX")"
STATUS_TSV="${TMP_ROOT}/task_status.tsv"
trap 'rm -rf "$TMP_ROOT"' EXIT

if [[ ! -f "$RUN_MANIFEST" ]]; then
  echo "run manifest not found: $RUN_MANIFEST" >&2
  exit 2
fi
if [[ ! -f "$SUITE_CONFIG" ]]; then
  echo "suite config not found: $SUITE_CONFIG" >&2
  exit 2
fi

TASK_LIST="$TMP_ROOT/tasks.tsv"
"$PYTHON_BIN" - "$SUITE_CONFIG" "$TMP_ROOT" > "$TASK_LIST" <<'PYTASKS'
from pathlib import Path
import sys
import yaml

suite_path = Path(sys.argv[1])
tmp_root = Path(sys.argv[2])
payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
if not isinstance(payload, dict):
    raise SystemExit("suite root must be a mapping")
tasks = payload.get("tasks")
if not isinstance(tasks, list):
    raise SystemExit("suite tasks must be a list")
for raw_task in tasks:
    if not isinstance(raw_task, dict):
        raise SystemExit("suite task entry must be a mapping")
    task_name = str(raw_task.get("task_name") or "")
    if not task_name:
        raise SystemExit("suite task_name must be non-empty")
    one_task_payload = dict(payload)
    one_task_payload["tasks"] = [raw_task]
    one_task_path = tmp_root / f"{task_name}.yaml"
    one_task_path.write_text(yaml.safe_dump(one_task_payload, sort_keys=False), encoding="utf-8")
    print(f"{task_name}\t{one_task_path}")
PYTASKS

RUN_KIND="$($PYTHON_BIN - "$RUN_MANIFEST" <<'PYKIND'
from pathlib import Path
import json
import sys
payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("run_kind", ""))
PYKIND
)"

RUN_ROOT="$($PYTHON_BIN - "$RUN_MANIFEST" <<'PYROOT'
from pathlib import Path
import sys
print(Path(sys.argv[1]).resolve(strict=False).parent)
PYROOT
)"
SUITE_NAME="$($PYTHON_BIN - "$SUITE_CONFIG" <<'PYSUITE'
from pathlib import Path
import sys
import yaml
payload = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(payload.get("evaluation_name", Path(sys.argv[1]).stem))
PYSUITE
)"
LOG_DIR="${RUN_ROOT}/evaluations/${SUITE_NAME}/logs"
mkdir -p "$LOG_DIR"
printf 'task_name\tstatus\tlog_path\n' > "$STATUS_TSV"

FAILED=0
while IFS=$'\t' read -r TASK_NAME TASK_SUITE; do
  [[ -n "$TASK_NAME" ]] || continue
  LOG_PATH="${LOG_DIR}/${TASK_NAME}.log"
  if [[ "$RUN_KIND" == "direct_ppo" ]]; then
    CMD=("$PYTHON_BIN" -m src.experiments.cli.experiments_cli_evaluate_policy --run-manifest "$RUN_MANIFEST" --suite "$TASK_SUITE")
  elif [[ "$RUN_KIND" == "curriculum" ]]; then
    CMD=("$PYTHON_BIN" -m src.experiments.cli.experiments_cli_evaluate_curriculum --summary "$RUN_MANIFEST" --suite "$TASK_SUITE" --model-scope final-stage)
  else
    echo "unsupported run_kind in manifest: $RUN_KIND" >&2
    exit 2
  fi
  {
    printf 'task=%s\n' "$TASK_NAME"
    printf 'command:'
    printf ' %q' "${CMD[@]}"
    printf '\nstarted_at=%s\n' "$(date --iso-8601=seconds)"
    "${CMD[@]}"
    STATUS=$?
    printf 'finished_at=%s\nexit_status=%s\n' "$(date --iso-8601=seconds)" "$STATUS"
    exit "$STATUS"
  } > "$LOG_PATH" 2>&1
  STATUS=$?
  if [[ "$STATUS" -eq 0 ]]; then
    printf '%s\tsuccess\t%s\n' "$TASK_NAME" "$LOG_PATH" >> "$STATUS_TSV"
  else
    printf '%s\tfailed\t%s\n' "$TASK_NAME" "$LOG_PATH" >> "$STATUS_TSV"
    FAILED=1
  fi
done < "$TASK_LIST"

"$PYTHON_BIN" - "$RUN_MANIFEST" "$SUITE_CONFIG" "$STATUS_TSV" <<'PYSUMMARY'
from __future__ import annotations

from pathlib import Path
import csv
import json
import sys
import yaml

run_manifest_path = Path(sys.argv[1]).resolve(strict=False)
suite_path = Path(sys.argv[2])
status_tsv = Path(sys.argv[3])
run_root = run_manifest_path.parent
manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
suite_payload = yaml.safe_load(suite_path.read_text(encoding="utf-8"))
suite_name = str(suite_payload.get("evaluation_name") or suite_path.stem)
eval_root = run_root / "evaluations" / suite_name
eval_root.mkdir(parents=True, exist_ok=True)
statuses = list(csv.DictReader(status_tsv.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
metric_keys = [
    "failure_overall_status",
    "failure_primary_mode",
    "failure_modes",
    "mean_position_error_tracking_m",
    "final_position_error_m",
    "max_position_error_m",
    "action_saturation_fraction",
    "real_action_saturation_fraction",
    "xy_tracking_ratio",
    "actual_xy_span_m",
    "reference_xy_span_m",
    "eval_terminated_count",
    "eval_truncated_count",
    "episode_count",
    "evaluation_suite_name",
    "suite_task_name",
]
summary_rows = []
for status in statuses:
    task_name = status["task_name"]
    metric_candidates = sorted(run_root.glob(f"**/{suite_name}/{task_name}/metrics/*_metrics.json"))
    metrics = {}
    if metric_candidates:
        metrics = json.loads(metric_candidates[-1].read_text(encoding="utf-8"))
    row = {
        "evaluation_suite_name": suite_name,
        "suite_config": str(suite_path),
        "task_name": task_name,
        "task_status": status["status"],
        "task_log_path": status["log_path"],
        "metrics_path": str(metric_candidates[-1]) if metric_candidates else None,
    }
    for key in metric_keys:
        value = metrics.get(key)
        if isinstance(value, (list, dict)):
            value = json.dumps(value, sort_keys=True)
        row[key] = value
    row["evaluated_task_name"] = task_name
    summary_rows.append(row)

summary_json_path = eval_root / "suite_summary.json"
summary_tsv_path = eval_root / "suite_summary.tsv"
summary_csv_path = eval_root / "suite_summary.csv"
summary_payload = {
    "run_name": manifest.get("run_name"),
    "run_kind": manifest.get("run_kind"),
    "evaluation_name": suite_name,
    "evaluation_suite_name": suite_name,
    "suite_config_path": str(suite_path),
    "task_count": len(summary_rows),
    "failed_task_count": sum(1 for row in summary_rows if row["task_status"] != "success"),
    "summary_tsv_path": str(summary_tsv_path),
    "summary_csv_path": str(summary_csv_path),
    "tasks": summary_rows,
}
summary_json_path.write_text(json.dumps(summary_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
fieldnames = list(summary_rows[0]) if summary_rows else ["task_name", "task_status"]
for path, delimiter in ((summary_tsv_path, "\t"), (summary_csv_path, ",")):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=delimiter)
        writer.writeheader()
        writer.writerows(summary_rows)

index_path = run_root / "evaluation_index.json"
index_payload = json.loads(index_path.read_text(encoding="utf-8")) if index_path.exists() else {"run_name": manifest.get("run_name"), "evaluations": []}
entries = index_payload.get("evaluations") if isinstance(index_payload.get("evaluations"), list) else []
entry = {
    "index_key": f"suite_summary:{suite_name}",
    "run_name": manifest.get("run_name"),
    "run_kind": manifest.get("run_kind"),
    "mode": "suite_task_loop_summary",
    "evaluation_name": suite_name,
    "evaluation_suite_name": suite_name,
    "suite_config_path": str(suite_path),
    "task_names": [row["task_name"] for row in summary_rows],
    "failed_task_count": summary_payload["failed_task_count"],
    "summary_json_path": str(summary_json_path),
    "summary_tsv_path": str(summary_tsv_path),
    "summary_csv_path": str(summary_csv_path),
}
entries = [candidate for candidate in entries if not isinstance(candidate, dict) or candidate.get("index_key") != entry["index_key"]]
entries.append(entry)
index_payload = {
    "run_name": manifest.get("run_name"),
    "run_kind": manifest.get("run_kind"),
    "index_path": str(index_path),
    "entry_count": len(entries),
    "evaluations": entries,
}
index_path.write_text(json.dumps(index_payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
manifest["evaluation_index"] = {
    "path": str(index_path),
    "entry_count": len(entries),
    "evaluations": entries,
}
run_manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"summary_json_path": str(summary_json_path), "summary_tsv_path": str(summary_tsv_path), "failed_task_count": summary_payload["failed_task_count"]}, indent=2, sort_keys=True))
PYSUMMARY

exit "$FAILED"
