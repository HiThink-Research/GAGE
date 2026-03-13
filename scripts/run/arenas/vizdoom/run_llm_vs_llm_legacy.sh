#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
PYTHON_BIN="${PYTHON_BIN:-$(gage_default_python)}"
CFG="${CFG:-${ROOT}/config/custom/vizdoom_llm_vs_llm.yaml}"
RUN_ID="${RUN_ID:-vizdoom_llm_vs_llm_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)}"

if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${LITELLM_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${LITELLM_API_KEY}"
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[oneclick][error] OPENAI_API_KEY or LITELLM_API_KEY is required." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cat <<MSG
[vizdoom][llm_vs_llm] backend vs backend.
Python: ${PYTHON_BIN}
Config: ${CFG}
MSG

PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"
