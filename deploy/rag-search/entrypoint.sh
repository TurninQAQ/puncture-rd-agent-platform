#!/usr/bin/env bash
set -Eeuo pipefail

password_file="${OPENSEARCH_ADMIN_PASSWORD_FILE:-/run/secrets/opensearch_admin_password}"
original_entrypoint="${OPENSEARCH_ORIGINAL_ENTRYPOINT:-/usr/share/opensearch/opensearch-docker-entrypoint.sh}"

if [[ -n "${OPENSEARCH_INITIAL_ADMIN_PASSWORD:-}" ]]; then
  echo "configuration error: provide the admin password through OPENSEARCH_ADMIN_PASSWORD_FILE only" >&2
  exit 64
fi
if [[ ! -f "$password_file" || ! -r "$password_file" ]]; then
  echo "configuration error: OPENSEARCH_ADMIN_PASSWORD_FILE must be a readable regular file" >&2
  exit 64
fi

IFS= read -r OPENSEARCH_INITIAL_ADMIN_PASSWORD < "$password_file" || true
if [[ -z "$OPENSEARCH_INITIAL_ADMIN_PASSWORD" ]]; then
  echo "configuration error: OpenSearch admin password file is empty" >&2
  exit 64
fi
if [[ "$OPENSEARCH_INITIAL_ADMIN_PASSWORD" == *$'\r'* || "$OPENSEARCH_INITIAL_ADMIN_PASSWORD" == *$'\n'* ]]; then
  echo "configuration error: OpenSearch admin password must be a single line" >&2
  exit 64
fi
if [[ ! -x "$original_entrypoint" ]]; then
  echo "configuration error: OpenSearch image entrypoint was not found" >&2
  exit 64
fi

export OPENSEARCH_INITIAL_ADMIN_PASSWORD
echo "Starting secured OpenSearch node cluster=${OPENSEARCH_CLUSTER_NAME:-unset} node=${OPENSEARCH_NODE_NAME:-unset}" >&2
exec "$original_entrypoint" "$@"
