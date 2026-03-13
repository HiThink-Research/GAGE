#!/usr/bin/env bash
set -euo pipefail

# 使用 multi_provider_http 后端跑 demo_echo 数据集

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
TEMPLATE="${ROOT}/scripts/run/backends/demos/templates/multi_provider_http.demo_echo.template.yaml"
GEN_DIR="${GEN_DIR:-$(gage_default_state_dir)/demos}"
CFG="${GEN_DIR}/multi_provider_demo.yaml"
VENV_PATH="${VENV_PATH:-$(gage_default_venv_path)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)/multi_provider_oneclick}"
ENV_FILE="${ENV_FILE:-$(gage_default_local_env_file)}"

# 覆盖参数（可通过环境变量修改）
PROVIDER="${HF_PROVIDER:-together}"
MODEL_NAME="${HF_MODEL_NAME:-Qwen/Qwen2.5-7B-Instruct}"
TOKENIZER_NAME="${HF_TOKENIZER_NAME:-${MODEL_NAME}}"
TIMEOUT="${HF_TIMEOUT:-120}"
PARALLEL_CALLS="${HF_PARALLEL_CALLS:-4}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
CONCURRENCY="${CONCURRENCY:-1}"

# 如果提供本地 env，自动加载
if [ -z "${HF_API_TOKEN:-}" ]; then
  gage_load_local_env "${ENV_FILE}"
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

gage_activate_venv "${VENV_PATH}"

echo "[scripts/run] running multi_provider_http pipeline"
python "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --concurrency "${CONCURRENCY}"

echo "[scripts/run] done. artifacts -> ${OUTPUT_DIR}"
