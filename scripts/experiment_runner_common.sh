#!/usr/bin/env bash
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck source=scripts/experiment_matrix.sh
source "${SCRIPT_DIR}/experiment_matrix.sh"

PYTHON_BIN="${PYTHON_BIN:-python}"
STORAGE_ROOT_DIR="${STORAGE_ROOT:-storage}"
VARIATION_SUITE="configs/evaluation/evaluation_task_suite_variation.yaml"
BROAD_SUITE="configs/evaluation/evaluation_task_suite_broad.yaml"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export TORCH_NUM_THREADS="${TORCH_NUM_THREADS:-1}"

print_thread_settings() {
  printf 'Thread settings: OMP=%s MKL=%s OPENBLAS=%s NUMEXPR=%s TORCH=%s\n' \
    "$OMP_NUM_THREADS" "$MKL_NUM_THREADS" "$OPENBLAS_NUM_THREADS" "$NUMEXPR_NUM_THREADS" "$TORCH_NUM_THREADS"
}

lane_log_root() {
  printf 'storage/logs/overnight_lanes/%s/lane_%s' "$LANE_RUN_ID" "$LANE_ID"
}

marker_path() {
  local experiment_id="$1"
  local phase="$2"
  printf '%s/markers/%s.%s.done' "$LANE_LOG_ROOT" "$experiment_id" "$phase"
}

manifest_path_for_run() {
  local run_name="$1"
  printf '%s/runs/%s/run_manifest.json' "$STORAGE_ROOT_DIR" "$run_name"
}

init_lane() {
  cd "$REPO_ROOT" || return 1
  LANE_RUN_ID="${LANE_RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
  LANE_LOG_ROOT="$(lane_log_root)"
  mkdir -p "$LANE_LOG_ROOT/markers"
  LANE_SUMMARY_TSV="$LANE_LOG_ROOT/lane_summary.tsv"
  LANE_SUMMARY_MD="$LANE_LOG_ROOT/lane_summary.md"
  printf 'experiment_id\tkind\tcurriculum_kind\tconfig_path\texpected_run_name\trun_name\twandb_group\twandb_name\tunit_count\ttrain_status\teval_status\tvariation_eval_status\tbroad_eval_status\trender_status\tmanifest_path\tnotes\n' > "$LANE_SUMMARY_TSV"
  print_thread_settings | tee "$LANE_LOG_ROOT/startup.log"
  printf 'LANE_RUN_ID=%s\nLANE_ID=%s\nLOG_ROOT=%s\n' "$LANE_RUN_ID" "$LANE_ID" "$LANE_LOG_ROOT" | tee -a "$LANE_LOG_ROOT/startup.log"
}

run_and_log() {
  local log_path="$1"
  shift
  {
    printf 'cwd=%s\n' "$PWD"
    printf 'command:'
    printf ' %q' "$@"
    printf '\nstarted_at=%s\n' "$(date --iso-8601=seconds)"
    "$@"
    local status=$?
    printf 'finished_at=%s\nexit_status=%s\n' "$(date --iso-8601=seconds)" "$status"
    return "$status"
  } > "$log_path" 2>&1
}

training_command() {
  local kind="$1"
  local config_path="$2"
  TRAIN_CMD=("$PYTHON_BIN")
  case "$kind" in
    direct_ppo)
      TRAIN_CMD+=( -m src.experiments.cli.experiments_cli_train_tracking --config "$config_path" )
      ;;
    manual_curriculum)
      TRAIN_CMD+=( -m src.experiments.cli.experiments_cli_train_curriculum --config "$config_path" )
      ;;
    llm_curriculum)
      TRAIN_CMD+=( -m src.experiments.cli.experiments_cli_train_llm_curriculum --config "$config_path" )
      ;;
    *)
      return 1
      ;;
  esac
  if [[ -n "${WANDB_MODE_OVERRIDE:-}" ]]; then
    TRAIN_CMD+=( --wandb-mode "$WANDB_MODE_OVERRIDE" )
  fi
}

evaluation_command() {
  local kind="$1"
  local manifest_path="$2"
  EVAL_CMD=("$PYTHON_BIN")
  case "$kind" in
    direct_ppo)
      EVAL_CMD+=( -m src.experiments.cli.experiments_cli_evaluate_policy --run-manifest "$manifest_path" )
      ;;
    manual_curriculum|llm_curriculum)
      EVAL_CMD+=( -m src.experiments.cli.experiments_cli_evaluate_curriculum --summary "$manifest_path" --model-scope final-stage )
      ;;
    *)
      return 1
      ;;
  esac
}

llm_server_available() {
  local config_path="$1"
  "$PYTHON_BIN" - "$config_path" <<'PYCHECK'
from pathlib import Path
from urllib.request import urlopen
import sys
import yaml

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
llm_config = config.get("llm", {}) if isinstance(config, dict) else {}
provider = str(llm_config.get("provider") or "")
api_base = str(llm_config.get("api_base") or "")
if provider != "openai_compatible":
    raise SystemExit(0)
if not api_base:
    raise SystemExit(2)
url = api_base.rstrip("/") + "/models"
try:
    with urlopen(url, timeout=2.0) as response:  # noqa: S310 - operator-provided localhost preflight URL.
        if 200 <= int(response.status) < 500:
            raise SystemExit(0)
except Exception as exc:  # noqa: BLE001 - shell preflight reports any connectivity failure.
    print(f"local LLM unavailable at {url}: {exc}")
    raise SystemExit(1)
raise SystemExit(1)
PYCHECK
}

experiment_curriculum_kind() {
  case "$1" in
    manual_curriculum) echo "manual" ;;
    llm_curriculum) echo "llm" ;;
    *) echo "" ;;
  esac
}

experiment_wandb_group() {
  local kind="$1"
  local run_name="$2"
  local variant seed action_interface task_distribution
  case "$kind" in
    manual_curriculum) echo "curriculum/manual/${run_name}" ;;
    llm_curriculum) echo "${run_name}" ;;
    direct_ppo)
      variant="${run_name#direct_ppo_}"
      seed="${variant##*_seed}"
      variant="${variant%_seed*}"
      if [[ "$run_name" == *directrpm* ]]; then
        action_interface="direct_rpm"
      else
        action_interface="pid_position"
      fi
      if [[ "$run_name" == *taskdist_medium* ]]; then
        task_distribution="tracking_medium"
      else
        task_distribution="fixed"
      fi
      echo "direct_ppo/${action_interface}/${task_distribution}/${variant}/seed${seed}"
      ;;
    *) echo "" ;;
  esac
}

append_summary_row() {
  local experiment_id="$1"
  local kind="$2"
  local config_path="$3"
  local run_name="$4"
  local units="$5"
  local train_status="$6"
  local eval_status="$7"
  local variation_status="$8"
  local broad_status="$9"
  local render_status="${10}"
  local manifest_path="${11}"
  local notes="${12}"
  local curriculum_kind wandb_group wandb_name
  curriculum_kind="$(experiment_curriculum_kind "$kind")"
  wandb_group="$(experiment_wandb_group "$kind" "$run_name")"
  wandb_name="$run_name"
  notes="${notes//$'\t'/ }"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$experiment_id" "$kind" "$curriculum_kind" "$config_path" "$run_name" "$run_name" "$wandb_group" "$wandb_name" "$units" "$train_status" "$eval_status" \
    "$variation_status" "$broad_status" "$render_status" "$manifest_path" "$notes" >> "$LANE_SUMMARY_TSV"
  write_markdown_summary
}

write_markdown_summary() {
  "$PYTHON_BIN" - "$LANE_SUMMARY_TSV" "$LANE_SUMMARY_MD" <<'PYMD'
from pathlib import Path
import csv
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
rows = list(csv.DictReader(src.read_text(encoding="utf-8").splitlines(), delimiter="\t"))
headers = ["experiment_id", "kind", "curriculum_kind", "run_name", "wandb_group", "wandb_name", "unit_count", "train_status", "eval_status", "variation_eval_status", "broad_eval_status", "render_status", "manifest_path", "notes"]
lines = ["# Lane Summary", "", "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
for row in rows:
    values = [str(row.get(header, "")).replace("|", "\\|") for header in headers]
    lines.append("| " + " | ".join(values) + " |")
dst.write_text("\n".join(lines) + "\n", encoding="utf-8")
PYMD
}

run_experiment() {
  local experiment_id="$1"
  local kind config_path run_name units notes manifest_path
  kind="$(experiment_kind "$experiment_id")" || return 1
  config_path="$(experiment_config "$experiment_id")" || return 1
  run_name="$(experiment_run_name "$experiment_id")"
  units="$(experiment_units "$experiment_id")"
  notes="$(experiment_notes "$experiment_id")"
  manifest_path="$(manifest_path_for_run "$run_name")"

  local train_status="not_run"
  local eval_status="not_run"
  local variation_status="not_run"
  local broad_status="not_run"
  local render_status="not_run"
  local failure_notes="${notes}"

  if [[ ! -f "$config_path" ]]; then
    append_summary_row "$experiment_id" "$kind" "$config_path" "$run_name" "$units" "missing_config" "skipped" "skipped" "skipped" "skipped" "$manifest_path" "missing config: $config_path"
    return 0
  fi

  if [[ "$kind" == "llm_curriculum" ]]; then
    local llm_log="$LANE_LOG_ROOT/${experiment_id}.llm_preflight.log"
    if ! llm_server_available "$config_path" > "$llm_log" 2>&1; then
      append_summary_row "$experiment_id" "$kind" "$config_path" "$run_name" "$units" "skipped_llm_unavailable" "skipped" "skipped" "skipped" "skipped" "$manifest_path" "local LLM unavailable; see ${llm_log}"
      return 0
    fi
  fi

  local train_marker eval_marker variation_marker broad_marker render_marker
  train_marker="$(marker_path "$experiment_id" train)"
  eval_marker="$(marker_path "$experiment_id" eval)"
  variation_marker="$(marker_path "$experiment_id" variation_eval)"
  broad_marker="$(marker_path "$experiment_id" broad_eval)"
  render_marker="$(marker_path "$experiment_id" render)"

  if [[ -f "$train_marker" ]]; then
    train_status="done_marker"
  elif [[ -f "$manifest_path" ]]; then
    train_status="existing_manifest"
    touch "$train_marker"
  else
    training_command "$kind" "$config_path"
    if run_and_log "$LANE_LOG_ROOT/${experiment_id}.train.log" "${TRAIN_CMD[@]}"; then
      if [[ -f "$manifest_path" ]]; then
        train_status="success"
        touch "$train_marker"
      else
        train_status="failed_missing_manifest"
        failure_notes="${failure_notes}; training exited 0 but manifest was not found"
      fi
    else
      train_status="failed"
      failure_notes="${failure_notes}; training failed"
    fi
  fi

  if [[ -f "$manifest_path" ]]; then
    if [[ -f "$eval_marker" ]]; then
      eval_status="done_marker"
    else
      evaluation_command "$kind" "$manifest_path"
      if run_and_log "$LANE_LOG_ROOT/${experiment_id}.eval.log" "${EVAL_CMD[@]}"; then
        eval_status="success"
        touch "$eval_marker"
      else
        eval_status="failed"
        failure_notes="${failure_notes}; standard evaluation failed"
      fi
    fi

    if [[ -f "$variation_marker" ]]; then
      variation_status="done_marker"
    else
      if run_and_log "$LANE_LOG_ROOT/${experiment_id}.variation_eval.log" bash scripts/evaluate_variation_suite.sh "$manifest_path" "$VARIATION_SUITE"; then
        variation_status="success"
        touch "$variation_marker"
      else
        variation_status="failed"
        failure_notes="${failure_notes}; variation evaluation failed"
      fi
    fi

    if [[ -f "$broad_marker" ]]; then
      broad_status="done_marker"
    else
      if run_and_log "$LANE_LOG_ROOT/${experiment_id}.broad_eval.log" bash scripts/evaluate_variation_suite.sh "$manifest_path" "$BROAD_SUITE"; then
        broad_status="success"
        touch "$broad_marker"
      else
        broad_status="failed"
        failure_notes="${failure_notes}; broad evaluation failed"
      fi
    fi

    if [[ -f "$render_marker" ]]; then
      render_status="done_marker"
    else
      if run_and_log "$LANE_LOG_ROOT/${experiment_id}.render.log" bash scripts/render_run_gifs.sh "$manifest_path"; then
        render_status="success"
        touch "$render_marker"
      else
        render_status="failed"
        failure_notes="${failure_notes}; render status report failed"
      fi
    fi
  else
    eval_status="skipped_no_manifest"
    variation_status="skipped_no_manifest"
    broad_status="skipped_no_manifest"
    render_status="skipped_no_manifest"
  fi

  append_summary_row "$experiment_id" "$kind" "$config_path" "$run_name" "$units" "$train_status" "$eval_status" "$variation_status" "$broad_status" "$render_status" "$manifest_path" "$failure_notes"
}

run_lane() {
  LANE_ID="$1"
  init_lane || return 1
  local ids id
  ids="$(lane_experiments "$LANE_ID")" || return 1
  for id in $ids; do
    printf '\n=== lane %s experiment %s ===\n' "$LANE_ID" "$id" | tee -a "$LANE_LOG_ROOT/startup.log"
    run_experiment "$id"
  done
  printf 'Lane %s complete. Summary: %s\n' "$LANE_ID" "$LANE_SUMMARY_TSV"
}
