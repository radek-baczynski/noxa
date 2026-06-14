#!/usr/bin/env bash
set -euo pipefail

export NOXA_RUNTIME_PROFILE="${NOXA_RUNTIME_PROFILE:-${NOXA_DOCKER_PROFILE:-cloud-cpu}}"

mkdir -p "${NOXA_MODEL_CACHE_DIR:-/data/models}"
mkdir -p "$(dirname "${NOXA_SQLITE_PATH:-/data/noxa.db}")"

cd /app
exec /app/.venv/bin/noxa "$@"
