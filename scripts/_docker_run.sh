#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SCRIPT_PATH="$2"
LOG_BASENAME="$3"
shift 3

IMAGE_NAME="drone-rl-llm-curriculum"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STORAGE_DIR="${PROJECT_DIR}/../storage"

mkdir -p "${STORAGE_DIR}"
STORAGE_DIR="$(cd "${STORAGE_DIR}" && pwd)"

DOCKER_HOME="${STORAGE_DIR}/.docker_home"
LOG_FILE="/workspace/storage/docker_logs/${LOG_BASENAME}"

mkdir -p \
  "${STORAGE_DIR}/runs" \
  "${STORAGE_DIR}/docker_logs" \
  "${STORAGE_DIR}/tmp" \
  "${DOCKER_HOME}"

SCRIPT_HOST_PATH="${PROJECT_DIR}/${SCRIPT_PATH}"
if [ ! -f "${SCRIPT_HOST_PATH}" ]; then
  echo "Script not found: ${SCRIPT_HOST_PATH}"
  exit 1
fi

if [[ "${SCRIPT_PATH}" == src/*.py ]]; then
  MODULE_NAME="${SCRIPT_PATH%.py}"
  MODULE_NAME="${MODULE_NAME//\//.}"
  PYTHON_COMMAND="python -m ${MODULE_NAME}"
else
  PYTHON_COMMAND="python '/workspace/repo/${SCRIPT_PATH}'"
fi

cat > "${DOCKER_HOME}/passwd" <<PASSWD
root:x:0:0:root:/root:/bin/bash
rino:x:$(id -u):$(id -g):Rino Albertin:/workspace/storage/.docker_home:/bin/bash
PASSWD

cat > "${DOCKER_HOME}/group" <<GROUP
root:x:0:
rino:x:$(id -g):
GROUP

chmod 644 "${DOCKER_HOME}/passwd" "${DOCKER_HOME}/group"

WANDB_API_KEY_VALUE="${WANDB_API_KEY:-}"
if [ -z "${WANDB_API_KEY_VALUE}" ] && [ -f "${HOME}/wandb_key.txt" ]; then
  WANDB_API_KEY_VALUE="$(tr -d '\r\n' < "${HOME}/wandb_key.txt")"
fi

WANDB_ENV_ARGS=()
if [ -n "${WANDB_API_KEY_VALUE}" ]; then
  WANDB_ENV_ARGS+=("-e" "WANDB_API_KEY=${WANDB_API_KEY_VALUE}")
fi

THREAD_ENV_ARGS=(
  "-e" "OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}"
  "-e" "MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}"
  "-e" "OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}"
  "-e" "NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}"
  "-e" "TORCH_NUM_THREADS=${TORCH_NUM_THREADS:-1}"
)

SSH_ARGS=()
if [ -d "${HOME}/.ssh" ]; then
  SSH_ARGS=(-v "${HOME}/.ssh:/workspace/storage/.docker_home/.ssh:ro")
fi

docker run --rm \
  --gpus "device=${GPU_ID}" \
  --user "$(id -u):$(id -g)" \
  --shm-size=16G \
  --workdir /workspace/repo \
  -e HOME=/workspace/storage/.docker_home \
  -e PROJECT_ROOT=/workspace/repo \
  -e STORAGE_ROOT=/workspace/storage \
  -e RUNS_DIR=/workspace/storage/runs \
  -e DOCKER_LOGS_DIR=/workspace/storage/docker_logs \
  -e TMP_DIR=/workspace/storage/tmp \
  "${WANDB_ENV_ARGS[@]}" \
  "${THREAD_ENV_ARGS[@]}" \
  -e PYTHONPATH=/workspace/repo \
  -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  -v "${DOCKER_HOME}/passwd:/etc/passwd:ro" \
  -v "${DOCKER_HOME}/group:/etc/group:ro" \
  -v "${PROJECT_DIR}:/workspace/repo:rw" \
  -v "${STORAGE_DIR}:/workspace/storage:rw" \
  "${SSH_ARGS[@]}" \
  "${IMAGE_NAME}" \
  bash -lc "ln -sfnT /workspace/storage /workspace/repo/storage && mkdir -p /workspace/storage/docker_logs && ${PYTHON_COMMAND} \"\$@\" > '${LOG_FILE}' 2>&1" -- "$@"