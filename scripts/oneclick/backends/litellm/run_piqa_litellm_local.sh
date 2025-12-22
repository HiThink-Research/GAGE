#!/usr/bin/env bash
set -euo pipefail

# 一键启动五个 LiteLLM Mock（OpenAI/Anthropic/Google/Grok/Kimi），并运行 PIQA 配置，生成汇总 summary。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
SCRIPT_DIR="${ROOT}/scripts/oneclick/backends/litellm"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/piqa_litellm_local}"
PIQA_SAMPLES="${PIQA_LITELLM_MAX_SAMPLES:-1}"
LITELLM_API_KEY="${LITELLM_API_KEY:-mock-key}"
QWEN_BASE="${QWEN_BASE:-http://127.0.0.1:1234/v1}"  # 本地 qwen OpenAI 兼容服务，用作统一底座
QWEN_MODEL="${QWEN_MODEL:-qwen2.5-0.5b-instruct-mlx}"

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
  PORT="${port}" MOCK_TARGET="${QWEN_BASE}" MOCK_API_KEY="${LITELLM_API_KEY}" MOCK_MODEL="${QWEN_MODEL}" LITELLM_API_KEY="${LITELLM_API_KEY}" python "${SCRIPT_DIR}/${script}" >/dev/null 2>&1 &
  PIDS+=($!)
  echo "[piqa-litellm] start ${script} on ${port}, pid=${PIDS[-1]}"
}

start_mock "mock_openai.py" "${MOCK_OAI_PORT}"
start_mock "mock_anthropic.py" "${MOCK_ANTH_PORT}"
start_mock "mock_google.py" "${MOCK_GGL_PORT}"
start_mock "mock_grok.py" "${MOCK_GROK_PORT}"
start_mock "mock_kimi.py" "${MOCK_KIMI_PORT}"

export PIQA_LITELLM_MAX_SAMPLES="${PIQA_SAMPLES}"
export LITELLM_API_KEY

echo "[piqa-litellm] running PIQA with max_samples=${PIQA_SAMPLES}, output=${OUTPUT_ROOT}"
python "${ROOT}/run.py" \
  --config "${ROOT}/config/custom/piqa_litellm.yaml" \
  --output-dir "${OUTPUT_ROOT}"

# python "${SCRIPT_DIR}/collect_summaries.py" --root "${OUTPUT_ROOT}" || true
# echo "[piqa-litellm] done. summary 位于 ${OUTPUT_ROOT}/*/summary.json"
