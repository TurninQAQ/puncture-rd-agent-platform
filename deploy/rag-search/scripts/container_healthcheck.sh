#!/usr/bin/env bash
set -Eeuo pipefail

username="${OPENSEARCH_HEALTH_USERNAME:-admin}"
password_file="${OPENSEARCH_HEALTH_PASSWORD_FILE:-/run/secrets/opensearch_admin_password}"
endpoint="${OPENSEARCH_HEALTH_ENDPOINT:-https://127.0.0.1:9200}"

if [[ ! -f "$password_file" || ! -r "$password_file" ]]; then
  exit 1
fi
IFS= read -r password < "$password_file" || true
if [[ -z "$password" ]]; then
  exit 1
fi

authorization="$(printf '%s:%s' "$username" "$password" | base64 | tr -d '\r\n')"
curl_options=(--silent --show-error --fail --connect-timeout 3 --max-time 8)
case "${OPENSEARCH_HEALTH_INSECURE:-false}" in
  true|1|yes|on) curl_options+=(--insecure) ;;
  false|0|no|off|"") ;;
  *) exit 1 ;;
esac

# The credential is passed through curl's stdin configuration, not argv or logs.
{
  printf 'header = "Authorization: Basic %s"\n' "$authorization"
  printf 'url = "%s/_cluster/health?wait_for_status=yellow&timeout=2s"\n' "$endpoint"
} | curl "${curl_options[@]}" --config - --output /dev/null
