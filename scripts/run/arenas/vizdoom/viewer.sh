#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
MODE="${MODE:-llm_visual}"
exec bash "${ROOT}/scripts/run/arenas/vizdoom/run.sh" --mode "${MODE}" "$@"
