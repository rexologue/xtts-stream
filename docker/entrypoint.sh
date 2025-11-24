#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${XTTS_CONFIG_FILE:-}" ]]; then
  echo "XTTS_CONFIG_FILE environment variable must be set to the configuration file path" >&2
  exit 1
fi

if [[ ! -f "${XTTS_CONFIG_FILE}" ]]; then
  echo "Configured XTTS_CONFIG_FILE (${XTTS_CONFIG_FILE}) does not exist inside the container" >&2
  exit 1
fi

exec "$@"
