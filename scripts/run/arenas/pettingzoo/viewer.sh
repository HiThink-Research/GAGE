#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MODE="${MODE:-dummy_visual}"
exec bash "${ROOT}/scripts/run/arenas/pettingzoo/run.sh" --mode "${MODE}" "$@"
