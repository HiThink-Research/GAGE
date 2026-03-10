#!/usr/bin/env bash
set -euo pipefail

# 使用 Kimi (moonshot) OpenAI 兼容接口跑 demo_echo 数据

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
CFG="${ROOT}/scripts/run/backends/demos/configs/kimi.pipeline.yaml"
VENV_PATH="${VENV_PATH:-$(gage_default_venv_path)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)/kimi_oneclick}"
ENV_FILE="${ENV_FILE:-$(gage_default_local_env_file)}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
CONCURRENCY="${CONCURRENCY:-1}"

# 自动加载本地 env
gage_load_local_env "${ENV_FILE}"

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

gage_activate_venv "${VENV_PATH}"

echo "[scripts/run] running Kimi (openai_http) pipeline"
python "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --concurrency "${CONCURRENCY}"

echo "[scripts/run] done. artifacts -> ${OUTPUT_DIR}"
