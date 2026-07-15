#!/usr/bin/env bash
set -euo pipefail

LOCAL_DEMO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${LOCAL_DEMO_DIR}/../.." && pwd)"
PYTHON_BIN="${PUNCTURE_DEMO_PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"

load_local_demo_env() {
  local env_file="${PUNCTURE_DEMO_ENV_FILE:-${LOCAL_DEMO_DIR}/.env}"
  if [[ ! -f "${env_file}" || -L "${env_file}" ]]; then
    echo "LOCAL_DEMO_CONFIG_ERROR create a private regular ${LOCAL_DEMO_DIR}/.env" >&2
    return 1
  fi
  local mode
  mode="$(stat -c '%a' -- "${env_file}")"
  if [[ ! "${mode}" =~ ^[0-7]00$ ]]; then
    echo "LOCAL_DEMO_CONFIG_ERROR .env must not be group/world accessible" >&2
    return 1
  fi

  declare -A seen=()
  local line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="${line%$'\r'}"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    if [[ "${line}" != *=* ]]; then
      echo "LOCAL_DEMO_CONFIG_ERROR malformed .env line" >&2
      return 1
    fi
    key="${line%%=*}"
    value="${line#*=}"
    if [[ ! "${key}" =~ ^[A-Z][A-Z0-9_]*$ || -n "${seen[${key}]:-}" ]]; then
      echo "LOCAL_DEMO_CONFIG_ERROR invalid or duplicate .env key" >&2
      return 1
    fi
    case "${key}" in
      RUN_FULL_STACK_DEMO|PUNCTURE_API_POSTGRES_DSN|PUNCTURE_API_POSTGRES_SCHEMA|\
      PUNCTURE_DEMO_HOST|PUNCTURE_DEMO_PORT|PUNCTURE_DEMO_BASE_URL|\
      PUNCTURE_DEMO_TOKEN_FILE|PUNCTURE_DEMO_CASE_IDS|VLLM_BASE_URL|VLLM_MODEL|\
      VLLM_TIMEOUT_SECONDS|OPENSEARCH_ENDPOINT|OPENSEARCH_USERNAME|\
      OPENSEARCH_PASSWORD_FILE|OPENSEARCH_CA_FILE|OPENSEARCH_INSECURE|\
      RAG_READ_ALIAS|RAG_WRITE_ALIAS)
        ;;
      *)
        echo "LOCAL_DEMO_CONFIG_ERROR unsupported .env key" >&2
        return 1
        ;;
    esac
    seen["${key}"]=1
    export "${key}=${value}"
  done < "${env_file}"

  if [[ "${RUN_FULL_STACK_DEMO:-0}" != "1" ]]; then
    echo "LOCAL_DEMO_CONFIG_ERROR set RUN_FULL_STACK_DEMO=1 after reviewing the boundary" >&2
    return 1
  fi
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "LOCAL_DEMO_CONFIG_ERROR configured Python environment is unavailable" >&2
    return 1
  fi
}
