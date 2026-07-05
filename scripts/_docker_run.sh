#!/usr/bin/env bash
set -euo pipefail

GPU_ID="$1"
SCRIPT_PATH="$2"
LOG_BASENAME="$3"
shift 3

IMAGE_NAME="drone-rl-llm-curriculum"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STORAGE_DIR="$(cd "${PROJECT_DIR}/../storage" && pwd)"
DOCKER_HOME="${STORAGE_DIR}/.docker_home"
LOG_FILE="/workspace/storage/logs/${LOG_BASENAME}"

mkdir -p \
  "${STORAGE_DIR}/logs" \
  "${STORAGE_DIR}/results" \
  "${STORAGE_DIR}/models" \
  "${STORAGE_DIR}/videos" \
  "${STORAGE_DIR}/gifs" \
  "${STORAGE_DIR}/llm_logs" \
  "${STORAGE_DIR}/wandb" \
  "${STORAGE_DIR}/datasets" \
  "${STORAGE_DIR}/tmp" \
  "${DOCKER_HOME}"

SCRIPT_HOST_PATH="${PROJECT_DIR}/${SCRIPT_PATH}"
if [ ! -f "${SCRIPT_HOST_PATH}" ]; then
  echo "Script not found: ${SCRIPT_HOST_PATH}"
  exit 1
fi

# ----------------------------------------------------------------------
# Create runtime user mapping for container
# ----------------------------------------------------------------------
cat > "${DOCKER_HOME}/passwd" <<PASSWD
root:x:0:0:root:/root:/bin/bash
rino:x:$(id -u):$(id -g):Rino Albertin:/workspace/storage/.docker_home:/bin/bash
PASSWD

cat > "${DOCKER_HOME}/group" <<GROUP
root:x:0:
rino:x:$(id -g):
GROUP

chmod 644 "${DOCKER_HOME}/passwd" "${DOCKER_HOME}/group"

# ----------------------------------------------------------------------
# Load W&B key if available
# ----------------------------------------------------------------------
WANDB_API_KEY_VALUE="${WANDB_API_KEY:-}"
if [ -z "${WANDB_API_KEY_VALUE}" ] && [ -f "${HOME}/wandb_key.txt" ]; then
  WANDB_API_KEY_VALUE="$(cat "${HOME}/wandb_key.txt")"
fi

# ----------------------------------------------------------------------
# Optional SSH mount for Git operations
# ----------------------------------------------------------------------
SSH_ARGS=()
if [ -d "${HOME}/.ssh" ]; then
  SSH_ARGS=(-v "${HOME}/.ssh:/workspace/storage/.docker_home/.ssh:ro")
fi

# ----------------------------------------------------------------------
# Run job inside Docker
# ----------------------------------------------------------------------
docker run --rm \
  --gpus "\"device=${GPU_ID}\"" \
  --user "$(id -u):$(id -g)" \
  --shm-size=16G \
  --workdir /workspace/repo \
  -e HOME=/workspace/storage/.docker_home \
  -e PROJECT_ROOT=/workspace/repo \
  -e STORAGE_ROOT=/workspace/storage \
  -e LOGS_DIR=/workspace/storage/logs \
  -e RESULTS_DIR=/workspace/storage/results \
  -e MODELS_DIR=/workspace/storage/models \
  -e VIDEOS_DIR=/workspace/storage/videos \
  -e GIFS_DIR=/workspace/storage/gifs \
  -e LLM_LOGS_DIR=/workspace/storage/llm_logs \
  -e DATASETS_DIR=/workspace/storage/datasets \
  -e TMP_DIR=/workspace/storage/tmp \
  -e WANDB_DIR=/workspace/storage/wandb \
  -e WANDB_CACHE_DIR=/workspace/storage/wandb/cache \
  -e WANDB_CONFIG_DIR=/workspace/storage/wandb/config \
  -e WANDB_API_KEY="${WANDB_API_KEY_VALUE}" \
  -e PYTHONPATH=/workspace/repo \
  -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  -v "${DOCKER_HOME}/passwd:/etc/passwd:ro" \
  -v "${DOCKER_HOME}/group:/etc/group:ro" \
  -v "${PROJECT_DIR}:/workspace/repo:rw" \
  -v "${STORAGE_DIR}:/workspace/storage:rw" \
  "${SSH_ARGS[@]}" \
  "${IMAGE_NAME}" \
  bash -lc "ln -sfn /workspace/storage /workspace/repo/storage && mkdir -p /workspace/storage/logs && python '/workspace/repo/${SCRIPT_PATH}' \"\$@\" > '${LOG_FILE}' 2>&1" -- "$@"