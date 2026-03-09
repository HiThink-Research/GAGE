#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
CFG="${CFG:-${ROOT}/config/custom/vizdoom_human_vs_llm.yaml}"
RUN_ID="${RUN_ID:-vizdoom_human_vs_llm_$(date +%Y%m%d_%H%M%S)}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs}"

export VIZDOOM_LITELLM_PROVIDER="${VIZDOOM_LITELLM_PROVIDER:-openai}"
export VIZDOOM_LITELLM_API_BASE="${VIZDOOM_LITELLM_API_BASE:-http://10.217.219.2:2722/v1}"
export VIZDOOM_LITELLM_MODEL="${VIZDOOM_LITELLM_MODEL:-/mnt/model/qwen3_omni_30b/}"
export VIZDOOM_LITELLM_API_KEY="${VIZDOOM_LITELLM_API_KEY:-${LITELLM_API_KEY:-${OPENAI_API_KEY:-empty}}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${VIZDOOM_LITELLM_API_KEY}}"
export LITELLM_API_KEY="${LITELLM_API_KEY:-${VIZDOOM_LITELLM_API_KEY}}"

mkdir -p "${OUTPUT_DIR}"

cat <<MSG
[vizdoom][human] Pygame input enabled.
Keys: A/Left=2, D/Right=3, Space/J=1
Python: ${PYTHON_BIN}
Config: ${CFG}
[vizdoom][human_vs_llm] Model base: ${VIZDOOM_LITELLM_API_BASE}
[vizdoom][human_vs_llm] Model name: ${VIZDOOM_LITELLM_MODEL}
MSG

PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"
