#!/usr/bin/env bash
set -euo pipefail

# 使用 multi_provider_http 后端跑 demo_echo 数据集

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TEMPLATE="${ROOT}/scripts/oneclick/configs/multi_provider_demo.template.yaml"
GEN_DIR="${ROOT}/scripts/oneclick/.generated"
CFG="${GEN_DIR}/multi_provider_demo.yaml"
VENV_PATH="${VENV_PATH:-${ROOT}/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/multi_provider_oneclick}"
ENV_FILE="${ENV_FILE:-${ROOT}/scripts/oneclick/.env}"

# 覆盖参数（可通过环境变量修改）
PROVIDER="${HF_PROVIDER:-together}"
MODEL_NAME="${HF_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${HF_TOKENIZER_NAME:-${MODEL_NAME}}"
TIMEOUT="${HF_TIMEOUT:-120}"
PARALLEL_CALLS="${HF_PARALLEL_CALLS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
CONCURRENCY="${CONCURRENCY:-1}"

# 如果提供 .env，自动加载（内容形如：export HF_API_TOKEN=xxx）
if [ -z "${HF_API_TOKEN:-}" ] && [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

# 兼容常见 token 变量名 & 确保子进程可见
if [ -z "${HF_API_TOKEN:-}" ] && [ -n "${HUGGINGFACEHUB_API_TOKEN:-}" ]; then
  HF_API_TOKEN="${HUGGINGFACEHUB_API_TOKEN}"
fi
if [ -n "${HF_API_TOKEN:-}" ]; then
  export HF_API_TOKEN
fi

if [ -z "${HF_API_TOKEN:-}" ]; then
  echo "[oneclick][warn] HF_API_TOKEN not set; AsyncInferenceClient will use default login context" >&2
fi

mkdir -p "${GEN_DIR}"

# 渲染模板为可运行的 PipelineConfig（避免引入额外依赖，直接字符串替换）
python - "$TEMPLATE" "$CFG" "$PROVIDER" "$MODEL_NAME" "$TOKENIZER_NAME" "$TIMEOUT" "$PARALLEL_CALLS" "$MAX_SAMPLES" "$CONCURRENCY" <<'PY'
import sys
from pathlib import Path

template, out = Path(sys.argv[1]), Path(sys.argv[2])
provider, model, tokenizer = sys.argv[3], sys.argv[4], sys.argv[5]
timeout, parallel_calls, max_samples, concurrency = sys.argv[6:]

text = template.read_text()
replacements = {
    "__PROVIDER__": provider,
    "__MODEL_NAME__": model,
    "__TOKENIZER_NAME__": tokenizer,
    "__TIMEOUT__": timeout,
    "__PARALLEL_CALLS__": parallel_calls,
    "__MAX_SAMPLES__": max_samples,
    "__CONCURRENCY__": concurrency,
}
for needle, value in replacements.items():
    text = text.replace(needle, str(value))
out.write_text(text)
print(f"[oneclick] rendered config -> {out}")
PY

if [ -d "${VENV_PATH}" ]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

echo "[oneclick] running multi_provider_http pipeline"
python "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --concurrency "${CONCURRENCY}"

echo "[oneclick] done. artifacts -> ${OUTPUT_DIR}"
