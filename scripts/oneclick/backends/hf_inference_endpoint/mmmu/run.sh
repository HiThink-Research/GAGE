#!/usr/bin/env bash
set -euo pipefail

# 一键跑 MMMU（hf_inference_endpoint），可复用现有 endpoint 或按模板创建。

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../../" && pwd)"
BASE_DIR="${ROOT}/scripts/oneclick/backends/hf_inference_endpoint/mmmu"
TEMPLATE="${BASE_DIR}/template.yaml"
MODEL_MATRIX="${MODEL_MATRIX:-${BASE_DIR}/model_matrix.yaml}"
GEN_DIR="${GEN_DIR:-${ROOT}/scripts/oneclick/.generated/backends/hf_inference_endpoint/mmmu}"
ENV_FILE="${ENV_FILE:-${ROOT}/scripts/oneclick/.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/hf_endpoint_mmmu}"
DRY_RUN="${DRY_RUN:-0}"
# 自动启用虚拟环境
if [ -n "${VENV_PATH:-}" ]; then
  VENV_PATH="${VENV_PATH}"
elif [ -d "${ROOT}/.venv311" ]; then
  VENV_PATH="${ROOT}/.venv311"
else
  VENV_PATH="${ROOT}/.venv"
fi
if [ -d "${VENV_PATH}" ]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

DEFAULT_MAX_SAMPLES="${MAX_SAMPLES:-1}"
DEFAULT_CONCURRENCY="${CONCURRENCY:-1}"
DEFAULT_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-512}"
DEFAULT_DATA_LIMIT="${DATA_LIMIT:-1}"
DEFAULT_WAIT_TIMEOUT="${WAIT_TIMEOUT:-1800}"
DEFAULT_POLL_INTERVAL="${POLL_INTERVAL:-60}"

mkdir -p "${GEN_DIR}"

if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi
if [ -z "${HUGGINGFACEHUB_API_TOKEN:-}" ] && [ -n "${HF_API_TOKEN:-}" ]; then
  HUGGINGFACEHUB_API_TOKEN="${HF_API_TOKEN}"
fi
if [ -z "${HUGGINGFACEHUB_API_TOKEN:-}" ]; then
  echo "[mmmu_endpoint] HUGGINGFACEHUB_API_TOKEN/HF_API_TOKEN 未设置" >&2
  exit 1
fi
export HUGGINGFACEHUB_API_TOKEN

if [ ! -f "${MODEL_MATRIX}" ]; then
  echo "[mmmu_endpoint][error] 未找到模型列表 ${MODEL_MATRIX}，请填充或使用示例 ${BASE_DIR}/model_matrix.example.yaml" >&2
  exit 1
fi

rendered_paths=()
while IFS= read -r line; do
  rendered_paths+=("$line")
done < <(
  python - <<'PY' "${TEMPLATE}" "${GEN_DIR}" "${MODEL_MATRIX}" "${DEFAULT_MAX_SAMPLES}" "${DEFAULT_CONCURRENCY}" "${DEFAULT_MAX_NEW_TOKENS}" "${DEFAULT_WAIT_TIMEOUT}" "${DEFAULT_POLL_INTERVAL}" "${DEFAULT_DATA_LIMIT}"
import re, sys
from pathlib import Path
import yaml

template_path = Path(sys.argv[1])
gen_dir = Path(sys.argv[2])
matrix_path = Path(sys.argv[3])
default_max_samples = int(sys.argv[4])
default_concurrency = int(sys.argv[5])
default_max_new_tokens = int(sys.argv[6])
default_wait_timeout = int(sys.argv[7])
default_poll_interval = int(sys.argv[8])
default_data_limit = int(sys.argv[9])

template = template_path.read_text()
data = yaml.safe_load(matrix_path.read_text()) or {}
models = data.get("models") or []
if not models:
    sys.stderr.write("[mmmu_endpoint][error] model_matrix.yaml contains no models\n")
    sys.exit(1)

def repl(text: str, mapping: dict) -> str:
    for k, v in mapping.items():
        text = text.replace(k, str(v))
    return text

def dump_scalar(val):
    txt = yaml.safe_dump(val, default_flow_style=True).strip()
    if txt == "...":
        return "null"
    txt = txt.replace("\n...", "").replace("...", "")
    return txt

def normalize_endpoint_name(model_name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9-]", "-", f"{model_name}-gage").lower()

for entry in models:
    name = entry.get("name")
    endpoint_name = entry.get("endpoint_name")
    model_name = entry.get("model_name")
    if not endpoint_name and model_name:
        endpoint_name = normalize_endpoint_name(model_name)

    if not name or (not endpoint_name and not model_name):
        sys.stderr.write(f"[mmmu_endpoint][warn] skip entry missing name and/or endpoint/model: {entry}\n")
        continue

    pipeline_name = f"hfi_{name}"
    backend_id = f"{name}_backend"
    adapter_id = f"{name}_dut"
    task_id = f"{name}_task"

    reuse_existing = entry.get("reuse_existing", True)
    auto_start = entry.get("auto_start", True)
    delete_on_exit = entry.get("delete_on_exit", False)
    vendor = entry.get("vendor", "aws")
    region = entry.get("region", "us-east-1")
    instance_type = entry.get("instance_type", "nvidia-a10g")
    instance_size = entry.get("instance_size", "x1")
    dtype = entry.get("dtype", None)
    wait_timeout = int(entry.get("wait_timeout", default_wait_timeout))
    poll_interval = int(entry.get("poll_interval", default_poll_interval))
    enable_async = bool(entry.get("enable_async", False))
    async_max_concurrency = int(entry.get("async_max_concurrency", 0))
    env_vars = entry.get("env_vars") or {}
    max_new_tokens = int(entry.get("max_new_tokens", default_max_new_tokens))
    max_samples = int(entry.get("max_samples", default_max_samples))
    concurrency = int(entry.get("concurrency", default_concurrency))
    data_limit = int(entry.get("data_limit", default_data_limit))

    rendered = repl(template, {
        "__PIPELINE_NAME__": pipeline_name,
        "__BACKEND_ID__": backend_id,
        "__ADAPTER_ID__": adapter_id,
        "__TASK_ID__": task_id,
        "__ENDPOINT_NAME__": dump_scalar(endpoint_name),
        "__NAMESPACE__": dump_scalar(entry.get("namespace", None)),
        "__MODEL_NAME__": dump_scalar(model_name),
        "__REVISION__": dump_scalar(entry.get("revision", "main")),
        "__REUSE_EXISTING__": dump_scalar(reuse_existing),
        "__AUTO_START__": dump_scalar(auto_start),
        "__DELETE_ON_EXIT__": dump_scalar(delete_on_exit),
        "__ACCELERATOR__": dump_scalar(entry.get("accelerator", "gpu")),
        "__VENDOR__": dump_scalar(vendor),
        "__REGION__": dump_scalar(region),
        "__INSTANCE_TYPE__": dump_scalar(instance_type),
        "__INSTANCE_SIZE__": dump_scalar(instance_size),
        "__ENDPOINT_TYPE__": dump_scalar(entry.get("endpoint_type", "protected")),
        "__FRAMEWORK__": dump_scalar(entry.get("framework", "pytorch")),
        "__DTYPE__": dump_scalar(dtype),
        "__WAIT_TIMEOUT__": dump_scalar(wait_timeout),
        "__POLL_INTERVAL__": dump_scalar(poll_interval),
        "__ENABLE_ASYNC__": dump_scalar(enable_async),
        "__ASYNC_MAX_CONCURRENCY__": dump_scalar(async_max_concurrency),
        "__ENV_VARS__": dump_scalar(env_vars),
        "__MAX_NEW_TOKENS__": dump_scalar(max_new_tokens),
        "__MAX_SAMPLES__": dump_scalar(max_samples),
        "__CONCURRENCY__": dump_scalar(concurrency),
        "__DATA_LIMIT__": dump_scalar(data_limit),
    })
    out_path = gen_dir / f"{name}.yaml"
    out_path.write_text(rendered)
    print(out_path)
PY
)

if [ ${#rendered_paths[@]} -eq 0 ]; then
  echo "[mmmu_endpoint][error] 无可用配置生成，检查模型列表" >&2
  exit 1
fi

echo "[mmmu_endpoint] 即将顺序跑 ${#rendered_paths[@]} 个模型 demo"
dry_flag="$(echo "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')"
for cfg in "${rendered_paths[@]}"; do
  base="$(basename "${cfg%.yaml}")"
  out_dir="${OUTPUT_ROOT}/${base}"
  echo "[mmmu_endpoint] running ${base} -> ${out_dir}"
  if [ "${dry_flag}" = "1" ] || [ "${dry_flag}" = "true" ]; then
    continue
  fi
  python "${ROOT}/run.py" \
    --config "${cfg}" \
    --output-dir "${out_dir}"
done

if [ "${dry_flag}" = "1" ] || [ "${dry_flag}" = "true" ]; then
  echo "[mmmu_endpoint] DRY_RUN 启用，已跳过实际 run.py 执行"
fi

echo "[mmmu_endpoint] done. 全部结果保存在 ${OUTPUT_ROOT}"
