#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
PYTHON_BIN="${PYTHON_BIN:-$(gage_default_python)}"
CFG="${CFG:-${ROOT}/config/custom/vizdoom_human_solo.yaml}"
RUN_ID="${RUN_ID:-vizdoom_human_solo_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)}"

mkdir -p "${OUTPUT_DIR}"

cat <<MSG
[vizdoom][human] Pygame input enabled.
Keys: A/Left=2, D/Right=3, Space/J=1
Python: ${PYTHON_BIN}
Config: ${CFG}
MSG

PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"
