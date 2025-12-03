#!/usr/bin/env bash
set -euo pipefail

# 为 multi_provider_http 后端跑一组模型 demo（单样本，低 token 消耗）

# 注意：脚本位于 scripts/oneclick/backends/multi_provider_http，下跳四级回到仓库根
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
TEMPLATE="${ROOT}/scripts/oneclick/backends/multi_provider_http/template.yaml"
MODEL_MATRIX="${MODEL_MATRIX:-${ROOT}/scripts/oneclick/backends/multi_provider_http/model_matrix.yaml}"
GEN_DIR="${ROOT}/scripts/oneclick/.generated/backends/multi_provider_http"
# 优先使用显式指定的虚拟环境；否则若存在 .venv311 则使用更高版本；再退回 .venv
if [ -n "${VENV_PATH:-}" ]; then
  VENV_PATH="${VENV_PATH}"
elif [ -d "${ROOT}/.venv311" ]; then
  VENV_PATH="${ROOT}/.venv311"
else
  VENV_PATH="${ROOT}/.venv"
fi
ENV_FILE="${ENV_FILE:-${ROOT}/scripts/oneclick/.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/mph_matrix}"

# 默认最小化样本/并发，可在模型列表中覆盖
DEFAULT_MAX_SAMPLES="${MAX_SAMPLES:-1}"
DEFAULT_CONCURRENCY="${CONCURRENCY:-1}"

mkdir -p "${GEN_DIR}"

# 加载 .env 以获取 HF_API_TOKEN；本地冒烟（自建 OpenAI 兼容服务）也需要带上 token 通过接口校验
if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
if [ -z "${HF_API_TOKEN:-}" ] && [ -n "${HUGGINGFACEHUB_API_TOKEN:-}" ]; then
  HF_API_TOKEN="${HUGGINGFACEHUB_API_TOKEN}"
fi
if [ -z "${HF_API_TOKEN:-}" ]; then
  echo "[run_all_models][error] HF_API_TOKEN/HUGGINGFACEHUB_API_TOKEN 未设置，无法调用 provider API" >&2
  exit 1
fi
export HF_API_TOKEN

if [ ! -f "${MODEL_MATRIX}" ]; then
  echo "[run_all_models][error] 未找到模型列表 ${MODEL_MATRIX}，请复制 model_matrix.example.yaml 为 model_matrix.yaml 并填写" >&2
  exit 1
fi

# 优先启用虚拟环境，保证 python 依赖可用
if [ -d "${VENV_PATH}" ]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

# 渲染多个模型的配置，返回路径列表
rendered_paths=()
while IFS= read -r line; do
  rendered_paths+=("$line")
done < <(
  python - <<'PY' "${TEMPLATE}" "${GEN_DIR}" "${MODEL_MATRIX}" "${DEFAULT_MAX_SAMPLES}" "${DEFAULT_CONCURRENCY}"
import sys, uuid
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
    sys.stderr.write("[run_all_models][error] model_matrix.yaml contains no models\n")
    sys.exit(1)

def repl(text: str, mapping: dict) -> str:
    for k, v in mapping.items():
        text = text.replace(k, str(v))
    return text

for entry in models:
    name = entry.get("name")
    provider = entry.get("provider")
    model_name = entry.get("model_name")
    tokenizer_name = entry.get("tokenizer_name") or model_name
    max_new_tokens = entry.get("max_new_tokens", 16)
    timeout = entry.get("timeout", 120)
    parallel_calls = entry.get("parallel_calls", 2)
    max_samples = entry.get("max_samples", default_max_samples)
    concurrency = entry.get("concurrency", default_concurrency)
    if not name or not provider or not model_name:
        sys.stderr.write(f"[run_all_models][warn] skip entry missing name/provider/model_name: {entry}\n")
        continue

    pipeline_name = f"mph_{name}"
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
        "__TOKENIZER_NAME__": tokenizer_name,
        "__TIMEOUT__": timeout,
        "__PARALLEL_CALLS__": parallel_calls,
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
  echo "[run_all_models][error] 无可用配置生成，检查模型列表" >&2
  exit 1
fi

echo "[run_all_models] 即将顺序跑 ${#rendered_paths[@]} 个模型 demo"
for cfg in "${rendered_paths[@]}"; do
  base="$(basename "${cfg%.yaml}")"
  out_dir="${OUTPUT_ROOT}/${base}"
  echo "[run_all_models] running ${base} -> ${out_dir}"
  python "${ROOT}/run.py" \
    --config "${cfg}" \
    --output-dir "${out_dir}"
done

echo "[run_all_models] done. 全部结果保存在 ${OUTPUT_ROOT}"
