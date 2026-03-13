#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
exec bash "${ROOT}/scripts/run/backends/providers/multi_provider_http/run_matrix.sh" "$@"
