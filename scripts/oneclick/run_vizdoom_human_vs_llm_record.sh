#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-${ROOT}/config/custom/vizdoom_human_vs_llm_record.yaml}"
RUN_ID="${RUN_ID:-vizdoom_human_vs_llm_record_$(date +%Y%m%d_%H%M%S)}"
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
[vizdoom][human] Pygame input enabled.
Keys: A/Left=2, D/Right=3, Space/J=1
Python: ${PYTHON_BIN}
Config: ${CFG}
MSG

PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"
