#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${XTTS_SETTINGS_FILE:-}" ]]; then
  echo "XTTS_SETTINGS_FILE environment variable must be set to the configuration file path" >&2
  exit 1
fi

if [[ ! -f "${XTTS_SETTINGS_FILE}" ]]; then
  echo "Configured XTTS_SETTINGS_FILE (${XTTS_SETTINGS_FILE}) does not exist inside the container" >&2
  exit 1
fi

exec "$@"
