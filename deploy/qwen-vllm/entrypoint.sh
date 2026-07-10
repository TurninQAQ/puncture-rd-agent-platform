#!/usr/bin/env bash
set -Eeuo pipefail

require_value() {
  local name="$1"
  if [[ -z "${!name:-}" ]]; then
    echo "configuration error: ${name} must not be empty" >&2
    exit 64
  fi
}

require_integer() {
  local name="$1"
  local value="${!name:-}"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < 1 )); then
    echo "configuration error: ${name} must be a positive integer" >&2
    exit 64
  fi
}

is_true() {
  case "${1,,}" in
    true|1|yes|on) return 0 ;;
    false|0|no|off|"") return 1 ;;
    *) echo "configuration error: invalid boolean '${1}'" >&2; exit 64 ;;
  esac
}

VLLM_MODEL_ID="${VLLM_MODEL_ID:-Qwen/Qwen3-8B}"
VLLM_MODEL_REVISION="${VLLM_MODEL_REVISION:-main}"
VLLM_SERVED_MODEL_NAME="${VLLM_SERVED_MODEL_NAME:-qwen-enterprise-agent}"
VLLM_CONTAINER_PORT="${VLLM_CONTAINER_PORT:-8000}"
VLLM_TENSOR_PARALLEL_SIZE="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_TOOL_CALL_PARSER="${VLLM_TOOL_CALL_PARSER:-hermes}"

require_value VLLM_MODEL_ID
require_value VLLM_MODEL_REVISION
require_value VLLM_SERVED_MODEL_NAME
require_integer VLLM_CONTAINER_PORT
require_integer VLLM_TENSOR_PARALLEL_SIZE
require_integer VLLM_MAX_MODEL_LEN
require_integer VLLM_MAX_NUM_SEQS

args=(
  vllm serve "$VLLM_MODEL_ID"
  --host 0.0.0.0
  --port "$VLLM_CONTAINER_PORT"
  --revision "$VLLM_MODEL_REVISION"
  --served-model-name "$VLLM_SERVED_MODEL_NAME"
  --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
  --max-model-len "$VLLM_MAX_MODEL_LEN"
  --max-num-seqs "$VLLM_MAX_NUM_SEQS"
  --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.90}"
  --dtype "${VLLM_DTYPE:-auto}"
  --kv-cache-dtype "${VLLM_KV_CACHE_DTYPE:-auto}"
  --load-format "${VLLM_LOAD_FORMAT:-auto}"
  --generation-config "${VLLM_GENERATION_CONFIG:-vllm}"
  --swap-space "${VLLM_SWAP_SPACE_GB:-4}"
  --cpu-offload-gb "${VLLM_CPU_OFFLOAD_GB:-0}"
  --uvicorn-log-level "${VLLM_UVICORN_LOG_LEVEL:-info}"
)

if [[ -n "${VLLM_MAX_NUM_BATCHED_TOKENS:-}" ]]; then
  require_integer VLLM_MAX_NUM_BATCHED_TOKENS
  args+=(--max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS")
fi

if [[ -n "${VLLM_QUANTIZATION:-}" ]]; then
  args+=(--quantization "$VLLM_QUANTIZATION")
fi

if is_true "${VLLM_TRUST_REMOTE_CODE:-false}"; then
  args+=(--trust-remote-code)
fi
if is_true "${VLLM_ENFORCE_EAGER:-false}"; then
  args+=(--enforce-eager)
fi
if is_true "${VLLM_ENABLE_PREFIX_CACHING:-true}"; then
  args+=(--enable-prefix-caching)
fi
if is_true "${VLLM_ENABLE_CHUNKED_PREFILL:-true}"; then
  args+=(--enable-chunked-prefill)
fi
if is_true "${VLLM_DISABLE_LOG_REQUESTS:-true}"; then
  args+=(--disable-log-requests)
fi

if is_true "${VLLM_ENABLE_AUTO_TOOL_CHOICE:-true}"; then
  require_value VLLM_TOOL_CALL_PARSER
  args+=(--enable-auto-tool-choice --tool-call-parser "$VLLM_TOOL_CALL_PARSER")
fi
if [[ -n "${VLLM_TOOL_PARSER_PLUGIN:-}" ]]; then
  args+=(--tool-parser-plugin "$VLLM_TOOL_PARSER_PLUGIN")
fi
if [[ -n "${VLLM_CHAT_TEMPLATE_PATH:-}" ]]; then
  args+=(--chat-template "$VLLM_CHAT_TEMPLATE_PATH")
fi
if [[ -n "${VLLM_REASONING_PARSER:-}" ]]; then
  args+=(--reasoning-parser "$VLLM_REASONING_PARSER")
fi
if is_true "${VLLM_ENABLE_REASONING_LEGACY_FLAG:-false}"; then
  args+=(--enable-reasoning)
fi

case "${VLLM_STRUCTURED_OUTPUT_FLAG_STYLE:-none}" in
  none)
    ;;
  legacy)
    require_value VLLM_STRUCTURED_OUTPUT_BACKEND
    args+=(--guided-decoding-backend "$VLLM_STRUCTURED_OUTPUT_BACKEND")
    ;;
  config)
    require_value VLLM_STRUCTURED_OUTPUT_BACKEND
    args+=(--structured-outputs-config.backend "$VLLM_STRUCTURED_OUTPUT_BACKEND")
    ;;
  *)
    echo "configuration error: VLLM_STRUCTURED_OUTPUT_FLAG_STYLE must be none, legacy, or config" >&2
    exit 64
    ;;
esac

if is_true "${VLLM_STRUCTURED_OUTPUT_ENABLE_IN_REASONING:-false}"; then
  args+=(--structured-outputs-config.enable_in_reasoning=True)
fi

if [[ -n "${VLLM_API_KEY:-}" && -n "${VLLM_API_KEY_FILE:-}" ]]; then
  echo "configuration error: set only one of VLLM_API_KEY and VLLM_API_KEY_FILE" >&2
  exit 64
fi

api_key="${VLLM_API_KEY:-}"
if [[ -n "${VLLM_API_KEY_FILE:-}" ]]; then
  if [[ -c "$VLLM_API_KEY_FILE" ]]; then
    # Compose binds /dev/null for the unauthenticated bootstrap profile. Treat a
    # character-device sentinel as an intentionally absent secret and never read
    # from an arbitrary device.
    api_key=""
  elif [[ ! -r "$VLLM_API_KEY_FILE" || ! -f "$VLLM_API_KEY_FILE" ]]; then
    echo "configuration error: VLLM_API_KEY_FILE is not a readable file" >&2
    exit 64
  else
    IFS= read -r api_key < "$VLLM_API_KEY_FILE" || true
  fi
fi
if [[ -n "$api_key" ]]; then
  args+=(--api-key "$api_key")
fi

if [[ -n "${HF_TOKEN:-}" && -n "${HF_TOKEN_FILE:-}" ]]; then
  echo "configuration error: set only one of HF_TOKEN and HF_TOKEN_FILE" >&2
  exit 64
fi
if [[ -n "${HF_TOKEN_FILE:-}" ]]; then
  if [[ -c "$HF_TOKEN_FILE" ]]; then
    HF_TOKEN=""
  elif [[ ! -r "$HF_TOKEN_FILE" || ! -f "$HF_TOKEN_FILE" ]]; then
    echo "configuration error: HF_TOKEN_FILE is not a readable file" >&2
    exit 64
  else
    IFS= read -r HF_TOKEN < "$HF_TOKEN_FILE" || true
  fi
  export HF_TOKEN
fi

echo "Starting private model service: model=${VLLM_MODEL_ID} revision=${VLLM_MODEL_REVISION} served_name=${VLLM_SERVED_MODEL_NAME} tp=${VLLM_TENSOR_PARALLEL_SIZE} max_context=${VLLM_MAX_MODEL_LEN}" >&2
exec "${args[@]}"
