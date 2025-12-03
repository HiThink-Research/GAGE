#!/usr/bin/env bash
set -euo pipefail

# 一键启动本地 Flask Mock（OpenAI/Anthropic/Google），并用 mock model_matrix 生成并跑 5 个协议（含 Grok/Kimi）的 demo。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
SCRIPT_DIR="${ROOT}/scripts/oneclick/backends/litellm"
MOCK_MATRIX="${SCRIPT_DIR}/model_matrix.mock.yaml"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/litellm_mock_matrix}"
LITELLM_API_KEY="${LITELLM_API_KEY:-mock-key}"

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
  PORT="${port}" python "${SCRIPT_DIR}/${script}" >/dev/null 2>&1 &
  PIDS+=($!)
  echo "[litellm-mock] start ${script} on ${port}, pid=${PIDS[-1]}"
}

start_mock "mock_openai.py" "${MOCK_OAI_PORT}"
start_mock "mock_anthropic.py" "${MOCK_ANTH_PORT}"
start_mock "mock_google.py" "${MOCK_GGL_PORT}"
start_mock "mock_grok.py" "${MOCK_GROK_PORT}"
start_mock "mock_kimi.py" "${MOCK_KIMI_PORT}"

export MODEL_MATRIX="${MOCK_MATRIX}"
export OUTPUT_ROOT
export LITELLM_API_KEY

echo "[litellm-mock] using MODEL_MATRIX=${MODEL_MATRIX}, OUTPUT_ROOT=${OUTPUT_ROOT}"
bash "${SCRIPT_DIR}/run_all_models.sh"
