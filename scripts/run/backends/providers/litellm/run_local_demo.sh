#!/usr/bin/env bash
set -euo pipefail

# Start five LiteLLM mock providers and forward requests to an OpenAI-compatible
# chat endpoint, then run the PIQA template matrix.

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
SCRIPT_DIR="${ROOT}/scripts/run/backends/providers/litellm"
OUTPUT_ROOT="${OUTPUT_ROOT:-$(gage_default_runs_dir)/piqa_litellm_local}"
PIQA_SAMPLES="${PIQA_LITELLM_MAX_SAMPLES:-1}"
MODEL_MATRIX="${SCRIPT_DIR}/models.mock.yaml"
LITELLM_API_KEY="${LITELLM_API_KEY:-mock-key}"
OPENAI_COMPATIBLE_BASE="${OPENAI_API_BASE:-https://api.openai.com/v1}"
OPENAI_COMPATIBLE_API_KEY="${OPENAI_API_KEY:-${LITELLM_API_KEY}}"
OPENAI_COMPATIBLE_MODEL="${OPENAI_MODEL:-gpt-5.4}"

MOCK_OAI_PORT="${MOCK_OAI_PORT:-18080}"
MOCK_ANTH_PORT="${MOCK_ANTH_PORT:-18081}"
MOCK_GGL_PORT="${MOCK_GGL_PORT:-18082}"
MOCK_GROK_PORT="${MOCK_GROK_PORT:-18083}"
MOCK_KIMI_PORT="${MOCK_KIMI_PORT:-18084}"

PIDS=()
cleanup() {
  for pid in "${PIDS[@]:-}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" || true
    fi
  done
}
trap cleanup EXIT

start_mock() {
  local script=$1 port=$2
  PORT="${port}" MOCK_TARGET="${OPENAI_COMPATIBLE_BASE}" MOCK_API_KEY="${OPENAI_COMPATIBLE_API_KEY}" MOCK_MODEL="${OPENAI_COMPATIBLE_MODEL}" LITELLM_API_KEY="${LITELLM_API_KEY}" python "${SCRIPT_DIR}/${script}" >/dev/null 2>&1 &
  PIDS+=($!)
  echo "[piqa-litellm] start ${script} on ${port}, pid=${PIDS[-1]}"
}

start_mock "mocks/openai.py" "${MOCK_OAI_PORT}"
start_mock "mocks/anthropic.py" "${MOCK_ANTH_PORT}"
start_mock "mocks/google.py" "${MOCK_GGL_PORT}"
start_mock "mocks/grok.py" "${MOCK_GROK_PORT}"
start_mock "mocks/kimi.py" "${MOCK_KIMI_PORT}"

export PIQA_LITELLM_MAX_SAMPLES="${PIQA_SAMPLES}"
export MAX_SAMPLES="${PIQA_SAMPLES}"
export CONCURRENCY="${CONCURRENCY:-4}"
export LITELLM_API_KEY
export MODEL_MATRIX
export OUTPUT_ROOT

echo "[piqa-litellm] running PIQA mock matrix with max_samples=${PIQA_SAMPLES}, output=${OUTPUT_ROOT}"
bash "${SCRIPT_DIR}/run_matrix.sh"
