#!/usr/bin/env bash
set -euo pipefail

# 为 TGI 后端生成 PIQA 测试配置，可选择 dry-run 跳过实际 run.py

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../../" && pwd)"
TEMPLATE="${TEMPLATE:-${ROOT}/scripts/oneclick/backends/tgi/template.yaml}"
MODEL_MATRIX="${MODEL_MATRIX:-${ROOT}/scripts/oneclick/backends/tgi/model_matrix.yaml}"
GEN_DIR="${GEN_DIR:-${ROOT}/config/custom/tgi_generated}"
# 优先使用显式指定的虚拟环境；否则若存在 .venv311 则使用更高版本；再退回 .venv
if [ -n "${VENV_PATH:-}" ]; then
  VENV_PATH="${VENV_PATH}"
elif [ -d "${ROOT}/.venv311" ]; then
  VENV_PATH="${ROOT}/.venv311"
else
  VENV_PATH="${ROOT}/.venv"
fi
ENV_FILE="${ENV_FILE:-${ROOT}/scripts/oneclick/.env}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${ROOT}/runs/tgi_matrix}"
DRY_RUN="${DRY_RUN:-0}"

DEFAULT_MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-64}"
DEFAULT_TEMPERATURE="${TEMPERATURE:-0.2}"
DEFAULT_TOP_P="${TOP_P:-0.9}"
DEFAULT_STOP="${STOP:-[]}"
DEFAULT_MAX_SAMPLES="${MAX_SAMPLES:-1}"
DEFAULT_CONCURRENCY="${CONCURRENCY:-1}"
DEFAULT_DATA_LIMIT="${DATA_LIMIT:-200}"
DEFAULT_TIMEOUT="${TIMEOUT:-120}"

mkdir -p "${GEN_DIR}"

# 可选加载 .env，便于统一注入代理或凭据
if [ -f "${ENV_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${ENV_FILE}"
fi

if [ ! -f "${MODEL_MATRIX}" ]; then
  echo "[tgi][error] 未找到模型列表 ${MODEL_MATRIX}，请复制 model_matrix.yaml.example 或指定 MODEL_MATRIX" >&2
  exit 1
fi

# 优先启用虚拟环境，保证 python 依赖可用
if [ -d "${VENV_PATH}" ]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

rendered_paths=()
while IFS= read -r line; do
  rendered_paths+=("$line")
done < <(
  python - <<'PY' "${TEMPLATE}" "${GEN_DIR}" "${MODEL_MATRIX}" "${DEFAULT_MAX_NEW_TOKENS}" "${DEFAULT_TEMPERATURE}" "${DEFAULT_TOP_P}" "${DEFAULT_STOP}" "${DEFAULT_MAX_SAMPLES}" "${DEFAULT_CONCURRENCY}" "${DEFAULT_DATA_LIMIT}" "${DEFAULT_TIMEOUT}"
import sys
from pathlib import Path
import yaml

template_path = Path(sys.argv[1])
gen_dir = Path(sys.argv[2])
matrix_path = Path(sys.argv[3])
default_max_new_tokens = int(sys.argv[4])
default_temperature = float(sys.argv[5])
default_top_p = float(sys.argv[6])
default_stop = yaml.safe_load(sys.argv[7])
default_max_samples = int(sys.argv[8])
default_concurrency = int(sys.argv[9])
default_data_limit = int(sys.argv[10])
default_timeout = int(sys.argv[11])

template = template_path.read_text()
data = yaml.safe_load(matrix_path.read_text()) or {}
models = data.get("models") or []
if not models:
    sys.stderr.write("[tgi][error] model_matrix.yaml contains no models\n")
    sys.exit(1)

def repl(text: str, mapping: dict) -> str:
    for k, v in mapping.items():
        text = text.replace(k, str(v))
    return text

def dump_scalar(val):
    txt = yaml.safe_dump(val, default_flow_style=True)
    txt = txt.strip()
    if txt == "...":
        return "null"
    txt = txt.replace("\n...", "").replace("...", "")
    return txt

for entry in models:
    name = entry.get("name")
    if not name:
        sys.stderr.write(f"[tgi][warn] skip entry missing name: {entry}\n")
        continue

    pipeline_name = f"tgi_{name}"
    backend_id = f"{name}_backend"
    adapter_id = f"{name}_dut"
    task_id = f"{name}_task"

    base_url = entry.get("base_url", "http://127.0.0.1:8080")
    max_new_tokens = int(entry.get("max_new_tokens", default_max_new_tokens))
    temperature = float(entry.get("temperature", default_temperature))
    top_p = float(entry.get("top_p", default_top_p))
    stop = entry.get("stop", default_stop)
    max_samples = int(entry.get("max_samples", default_max_samples))
    concurrency = int(entry.get("concurrency", default_concurrency))
    data_limit = int(entry.get("data_limit", default_data_limit))
    timeout = int(entry.get("timeout", default_timeout))

    rendered = repl(template, {
        "__PIPELINE_NAME__": pipeline_name,
        "__BACKEND_ID__": backend_id,
        "__ADAPTER_ID__": adapter_id,
        "__TASK_ID__": task_id,
        "__BASE_URL__": dump_scalar(base_url),
        "__MAX_NEW_TOKENS__": dump_scalar(max_new_tokens),
        "__TEMPERATURE__": dump_scalar(temperature),
        "__TOP_P__": dump_scalar(top_p),
        "__STOP__": dump_scalar(stop),
        "__MAX_SAMPLES__": dump_scalar(max_samples),
        "__CONCURRENCY__": dump_scalar(concurrency),
        "__DATA_LIMIT__": dump_scalar(data_limit),
        "__TIMEOUT__": dump_scalar(timeout),
    })
    out_path = gen_dir / f"{name}.yaml"
    out_path.write_text(rendered)
    print(out_path)
PY
)

if [ ${#rendered_paths[@]} -eq 0 ]; then
  echo "[tgi][error] 无可用配置生成，检查模型列表" >&2
  exit 1
fi

echo "[tgi] 即将顺序跑 ${#rendered_paths[@]} 个模型 demo"
dry_flag="$(echo "${DRY_RUN}" | tr '[:upper:]' '[:lower:]')"
for cfg in "${rendered_paths[@]}"; do
  base="$(basename "${cfg%.yaml}")"
  out_dir="${OUTPUT_ROOT}/${base}"
  echo "[tgi] running ${base} -> ${out_dir}"
  if [ "${dry_flag}" = "1" ] || [ "${dry_flag}" = "true" ]; then
    continue
  fi
  python "${ROOT}/run.py" \
    --config "${cfg}" \
    --output-dir "${out_dir}"
done

if [ "${dry_flag}" = "1" ] || [ "${dry_flag}" = "true" ]; then
  echo "[tgi] DRY_RUN 启用，已跳过实际 run.py 执行"
fi

echo "[tgi] done. 全部结果保存在 ${OUTPUT_ROOT}"
