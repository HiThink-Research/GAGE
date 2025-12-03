#!/usr/bin/env bash
set -euo pipefail

# 为 litellm 后端跑一组模型 demo（单样本，低 token 消耗）

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
TEMPLATE="${ROOT}/scripts/oneclick/backends/litellm/template.yaml"
MODEL_MATRIX="${MODEL_MATRIX:-${ROOT}/scripts/oneclick/backends/litellm/model_matrix.yaml}"
GEN_DIR="${ROOT}/scripts/oneclick/.generated/backends/litellm"
VENV_PATH="${VENV_PATH:-${ROOT}/.venv}"
ENV_FILE="${ENV_FILE:-${ROOT}/scripts/oneclick/.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/litellm_matrix}"

DEFAULT_MAX_SAMPLES="${MAX_SAMPLES:-1}"
DEFAULT_CONCURRENCY="${CONCURRENCY:-1}"

mkdir -p "${GEN_DIR}"

# 加载 .env 以获取 API key
if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

if [ ! -f "${MODEL_MATRIX}" ]; then
  echo "[litellm][error] 未找到模型列表 ${MODEL_MATRIX}，请复制 model_matrix.example.yaml 为 model_matrix.yaml 并填写" >&2
  exit 1
fi

# 优先启用虚拟环境，保证 python 依赖可用
if [ -d "${VENV_PATH}" ]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

rendered_paths=()
skipped=0
while IFS= read -r line; do
  rendered_paths+=("$line")
done < <(
  python - <<'PY' "${TEMPLATE}" "${GEN_DIR}" "${MODEL_MATRIX}" "${DEFAULT_MAX_SAMPLES}" "${DEFAULT_CONCURRENCY}"
import os, sys
from pathlib import Path
import yaml

template_path = Path(sys.argv[1])
gen_dir = Path(sys.argv[2])
matrix_path = Path(sys.argv[3])
default_max_samples = int(sys.argv[4])
default_concurrency = int(sys.argv[5])

template = template_path.read_text()
data = yaml.safe_load(matrix_path.read_text()) or {}
models = data.get("models") or []
if not models:
    sys.stderr.write("[litellm][error] model_matrix.yaml contains no models\n")
    sys.exit(1)

def repl(text: str, mapping: dict) -> str:
    for k, v in mapping.items():
        text = text.replace(k, str(v))
    return text

for entry in models:
    name = entry.get("name")
    provider = entry.get("provider")
    model_name = entry.get("model_name")
    api_base = entry.get("api_base", "")
    api_key_env = entry.get("api_key_env")
    api_key = None
    if api_key_env:
        api_key = os.getenv(api_key_env)
    # 回退常见变量
    if not api_key:
        for key in ("LITELLM_API_KEY", "OPENAI_API_KEY", "KIMI_API_KEY", "MOONSHOT_API_KEY"):
            api_key = os.getenv(key)
            if api_key:
                break

    max_new_tokens = entry.get("max_new_tokens", 16)
    timeout = entry.get("timeout", 60)
    parallel_calls = entry.get("parallel_calls", 1)
    max_samples = entry.get("max_samples", default_max_samples)
    concurrency = entry.get("concurrency", default_concurrency)
    if not name or not provider or not model_name:
        sys.stderr.write(f"[litellm][warn] skip entry missing name/provider/model_name: {entry}\n")
        continue
    if not api_key:
        sys.stderr.write(f"[litellm][warn] skip {name}: missing API key (env {api_key_env or 'OPENAI_API_KEY/...'} not set)\n")
        continue

    pipeline_name = f"litellm_{name}"
    backend_id = f"{name}_backend"
    adapter_id = f"{name}_dut"
    task_id = f"{name}_task"

    rendered = repl(template, {
        "__PIPELINE_NAME__": pipeline_name,
        "__BACKEND_ID__": backend_id,
        "__ADAPTER_ID__": adapter_id,
        "__TASK_ID__": task_id,
        "__PROVIDER__": provider,
        "__MODEL_NAME__": model_name,
        "__API_BASE__": api_base or "",
        "__API_KEY__": api_key,
        "__TIMEOUT__": timeout,
        "__MAX_NEW_TOKENS__": max_new_tokens,
        "__MAX_SAMPLES__": max_samples,
        "__CONCURRENCY__": concurrency,
    })
    out_path = gen_dir / f"{name}.yaml"
    out_path.write_text(rendered)
    print(out_path)
PY
)

if [ ${#rendered_paths[@]} -eq 0 ]; then
  echo "[litellm][error] 无可用配置生成（可能缺少 API key），请检查 model_matrix.yaml 与环境变量" >&2
  exit 1
fi

echo "[litellm] 即将顺序跑 ${#rendered_paths[@]} 个模型 demo"
for cfg in "${rendered_paths[@]}"; do
  base="$(basename "${cfg%.yaml}")"
  out_dir="${OUTPUT_ROOT}/${base}"
  echo "[litellm] running ${base} -> ${out_dir}"
  python "${ROOT}/run.py" \
    --config "${cfg}" \
    --output-dir "${out_dir}"
done

echo "[litellm] done. 全部结果保存在 ${OUTPUT_ROOT}"
