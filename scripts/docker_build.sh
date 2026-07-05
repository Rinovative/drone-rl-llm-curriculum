#!/usr/bin/env bash
set -euo pipefail

IMAGE_NAME="drone-rl-llm-curriculum"

PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "${PROJECT_DIR}"

echo "Building Docker image: ${IMAGE_NAME}"
docker build \
  -t "${IMAGE_NAME}" \
  .
