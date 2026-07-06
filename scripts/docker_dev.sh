#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="drone-rl-llm-curriculum"
CONTAINER_NAME="drone-rl-llm-curriculum-dev"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
STORAGE_DIR="${PROJECT_DIR}/../storage"

mkdir -p "${STORAGE_DIR}"
STORAGE_DIR="$(cd "${STORAGE_DIR}" && pwd)"

DOCKER_HOME="${STORAGE_DIR}/.docker_home"

mkdir -p \
  "${STORAGE_DIR}/training_runs" \
  "${STORAGE_DIR}/evaluation_runs" \
  "${STORAGE_DIR}/comparison_reports" \
  "${STORAGE_DIR}/docker_logs" \
  "${STORAGE_DIR}/tmp" \
  "${DOCKER_HOME}"

cat > "${DOCKER_HOME}/passwd" <<PASSWD
root:x:0:0:root:/root:/bin/bash
rino:x:$(id -u):$(id -g):Rino Albertin:/workspace/storage/.docker_home:/bin/bash
PASSWD

cat > "${DOCKER_HOME}/group" <<GROUP
root:x:0:
rino:x:$(id -g):
GROUP

chmod 644 "${DOCKER_HOME}/passwd" "${DOCKER_HOME}/group"

SSH_ARGS=()
if [ -d "${HOME}/.ssh" ]; then
  SSH_ARGS=(-v "${HOME}/.ssh:/workspace/storage/.docker_home/.ssh:ro")
fi

if docker ps --format "{{.Names}}" | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' is already running."
  echo "Attach with VS Code or stop it with:"
  echo "  docker stop ${CONTAINER_NAME}"
  exit 0
fi

if docker ps -a --format "{{.Names}}" | grep -qx "${CONTAINER_NAME}"; then
  echo "Removing stopped container '${CONTAINER_NAME}'."
  docker rm "${CONTAINER_NAME}" >/dev/null
fi

WANDB_API_KEY_VALUE="${WANDB_API_KEY:-}"
if [ -z "${WANDB_API_KEY_VALUE}" ] && [ -f "${HOME}/wandb_key.txt" ]; then
  WANDB_API_KEY_VALUE="$(tr -d '\r\n' < "${HOME}/wandb_key.txt")"
fi

WANDB_ENV_ARGS=()
if [ -n "${WANDB_API_KEY_VALUE}" ]; then
  WANDB_ENV_ARGS+=("-e" "WANDB_API_KEY=${WANDB_API_KEY_VALUE}")
fi

docker run -d --rm \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --user "$(id -u):$(id -g)" \
  --shm-size=16G \
  --workdir /workspace/repo \
  -e HOME=/workspace/storage/.docker_home \
  -e PROJECT_ROOT=/workspace/repo \
  -e STORAGE_ROOT=/workspace/storage \
  -e TRAINING_RUNS_DIR=/workspace/storage/training_runs \
  -e EVALUATION_RUNS_DIR=/workspace/storage/evaluation_runs \
  -e COMPARISON_REPORTS_DIR=/workspace/storage/comparison_reports \
  -e DOCKER_LOGS_DIR=/workspace/storage/docker_logs \
  -e TMP_DIR=/workspace/storage/tmp \
  "${WANDB_ENV_ARGS[@]}" \
  -e PYTHONPATH=/workspace/repo \
  -e GIT_SSH_COMMAND="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null" \
  -v "${DOCKER_HOME}/passwd:/etc/passwd:ro" \
  -v "${DOCKER_HOME}/group:/etc/group:ro" \
  -v "${PROJECT_DIR}:/workspace/repo:rw" \
  -v "${STORAGE_DIR}:/workspace/storage:rw" \
  "${SSH_ARGS[@]}" \
  "${IMAGE_NAME}" \
  bash -lc "ln -sfnT /workspace/storage /workspace/repo/storage && sleep infinity"

echo "Container started: ${CONTAINER_NAME}"
echo "Attach with VS Code: Remote Explorer -> Containers -> ${CONTAINER_NAME}"
echo "Stop with: docker stop ${CONTAINER_NAME}"