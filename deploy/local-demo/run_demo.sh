#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"
load_local_demo_env

umask 077
mkdir -p -- "${PROJECT_ROOT}/var/local-demo"
log_file="${PROJECT_ROOT}/var/local-demo/api.log"

cd -- "${PROJECT_ROOT}"
"${PYTHON_BIN}" examples/live_api_server.py >"${log_file}" 2>&1 &
server_pid=$!

cleanup() {
  if kill -0 "${server_pid}" 2>/dev/null; then
    kill -TERM "${server_pid}" 2>/dev/null || true
  fi
  wait "${server_pid}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

"${PYTHON_BIN}" examples/live_api_demo.py --wait-seconds 120
