#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "[compat] scripts/oneclick wrappers are deprecated; use scripts/run entrypoints instead." >&2
exec bash "${ROOT}/scripts/run/arenas/vizdoom/viewer.sh" "$@"
