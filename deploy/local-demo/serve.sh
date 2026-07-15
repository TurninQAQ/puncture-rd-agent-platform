#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
. "${SCRIPT_DIR}/common.sh"
load_local_demo_env

cd -- "${PROJECT_ROOT}"
exec "${PYTHON_BIN}" examples/live_api_server.py
