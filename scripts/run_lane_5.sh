#!/usr/bin/env bash
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=scripts/experiment_runner_common.sh
source "${SCRIPT_DIR}/experiment_runner_common.sh"
run_lane 5
