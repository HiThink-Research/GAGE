#!/usr/bin/env bash
set -euo pipefail

# 使用 Kimi (moonshot) OpenAI 兼容接口跑 demo_echo 数据

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${ROOT}/scripts/oneclick/configs/kimi_demo.yaml"
VENV_PATH="${VENV_PATH:-${ROOT}/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/kimi_oneclick}"
ENV_FILE="${ENV_FILE:-${ROOT}/scripts/oneclick/.env}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
CONCURRENCY="${CONCURRENCY:-1}"

# 自动加载 .env
if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

# 兼容多个常见变量名，最终导出 OPENAI_API_KEY
if [ -z "${OPENAI_API_KEY:-}" ]; then
  if [ -n "${MOONSHOT_API_KEY:-}" ]; then
    OPENAI_API_KEY="${MOONSHOT_API_KEY}"
  elif [ -n "${KIMI_API_KEY:-}" ]; then
    OPENAI_API_KEY="${KIMI_API_KEY}"
  fi
fi
if [ -n "${OPENAI_API_KEY:-}" ]; then
  export OPENAI_API_KEY
else
  echo "[oneclick][error] OPENAI_API_KEY/MOONSHOT_API_KEY/KIMI_API_KEY 未设置，Kimi 接口需要有效的 API key" >&2
  exit 1
fi

if [ -d "${VENV_PATH}" ]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

echo "[oneclick] running Kimi (openai_http) pipeline"
python "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --concurrency "${CONCURRENCY}"

echo "[oneclick] done. artifacts -> ${OUTPUT_DIR}"
