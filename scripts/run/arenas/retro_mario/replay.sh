#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

if [[ $# -gt 0 && "${1}" != --* ]]; then
  RUN_ID_ARG="$1"
  shift
  set -- --run-id "${RUN_ID_ARG}" "$@"
elif [[ $# -eq 0 && -n "${RUN_ID:-}" ]]; then
  set -- --run-id "${RUN_ID}"
fi

exec bash "${ROOT}/scripts/run/arenas/replay/run_and_open.sh" "$@"
