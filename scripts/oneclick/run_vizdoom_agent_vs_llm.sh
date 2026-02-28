#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-${ROOT}/config/custom/vizdoom_agent_vs_llm.yaml}"
RUN_ID="${RUN_ID:-vizdoom_human_vs_llm_incremental_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs}"

if [ -z "${OPENAI_API_KEY:-}" ] && [ -n "${LITELLM_API_KEY:-}" ]; then
  export OPENAI_API_KEY="${LITELLM_API_KEY}"
fi
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo "[oneclick][error] OPENAI_API_KEY or LITELLM_API_KEY is required." >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

cat <<MSG
[vizdoom][human_vs_llm] p0 uses type=human (pygame input).
[vizdoom][human_vs_llm] p1 uses type=backend (dut_model).
[vizdoom][human_vs_llm] Keys: A/Left=2, D/Right=3, Space/J=1
[vizdoom][human_vs_llm] Python: ${PYTHON_BIN}
[vizdoom][human_vs_llm] Config: ${CFG}
MSG

PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"
