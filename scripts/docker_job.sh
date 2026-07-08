#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STORAGE_DIR="${PROJECT_DIR}/../storage"

usage() {
  echo "Usage: scripts/docker_job.sh <repo-script-path> [args...]" >&2
  echo "Example: scripts/docker_job.sh src/experiments/cli/experiments_cli_train_tracking.py --config configs/training/ppo_tracking_smoke.yaml" >&2
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

mkdir -p "${STORAGE_DIR}"
STORAGE_DIR="$(cd "${STORAGE_DIR}" && pwd)"

LOG_DIR="${STORAGE_DIR}/docker_logs"
mkdir -p "${LOG_DIR}"

SCRIPT_PATH="$1"
shift

SCRIPT_HOST_PATH="${PROJECT_DIR}/${SCRIPT_PATH}"
if [ ! -f "${SCRIPT_HOST_PATH}" ]; then
  echo "Script not found: ${SCRIPT_HOST_PATH}"
  exit 1
fi

echo "Current GPU usage:"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total \
  --format=csv,noheader,nounits
echo "------------------------------------------------------------"

AUTO_GPU="$(
  nvidia-smi --query-gpu=index,memory.used \
    --format=csv,noheader,nounits \
    | sort -t, -k2 -n \
    | head -n1 \
    | cut -d',' -f1 \
    | xargs
)"

read -r -p "Select GPU (press Enter for ${AUTO_GPU}): " GPU_ID
GPU_ID="${GPU_ID:-$AUTO_GPU}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
SCRIPT_TAG="$(basename "${SCRIPT_PATH}" .py)"
LOG_BASENAME="${TIMESTAMP}__${SCRIPT_TAG}__gpu${GPU_ID}.log"

cd "${PROJECT_DIR}"

if command -v runTSGPU.py >/dev/null 2>&1; then
  runTSGPU.py -g"${GPU_ID}" -- scripts/_docker_run.sh \
    "${GPU_ID}" \
    "${SCRIPT_PATH}" \
    "${LOG_BASENAME}" \
    "$@"

  echo "Queued Docker job on GPU ${GPU_ID}: ${SCRIPT_PATH}"
  echo "Queue: runTSGPU.py -g${GPU_ID} -s"
else
  echo "runTSGPU.py not found. Running Docker job directly."
  scripts/_docker_run.sh \
    "${GPU_ID}" \
    "${SCRIPT_PATH}" \
    "${LOG_BASENAME}" \
    "$@"
fi

echo "Log: tail -f ${LOG_DIR}/${LOG_BASENAME}"