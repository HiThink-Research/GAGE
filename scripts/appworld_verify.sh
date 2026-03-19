#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper for the canonical `appworld verify tasks` entrypoint.
# The delegated script reads `APPWORLD_APIS_URL` and `APPWORLD_MCP_URL`.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "${ROOT}/scripts/run/appworld/verify.sh" "$@"
