#!/usr/bin/env bash
set -euo pipefail

# 运行最小 dummy echo 冒烟，验证端到端链路

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${ROOT}/scripts/oneclick/configs/demo_echo_pipeline.yaml"
VENV_PATH="${VENV_PATH:-${ROOT}/.venv}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs/demo_echo_oneclick}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
CONCURRENCY="${CONCURRENCY:-1}"

if [ -d "${VENV_PATH}" ]; then
  # shellcheck disable=SC1091
  source "${VENV_PATH}/bin/activate"
fi

echo "[oneclick] running dummy echo pipeline"
python "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --concurrency "${CONCURRENCY}"

echo "[oneclick] done. artifacts -> ${OUTPUT_DIR}"
