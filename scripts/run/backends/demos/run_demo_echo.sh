#!/usr/bin/env bash
set -euo pipefail

# 运行最小 dummy echo 冒烟，验证端到端链路

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
CFG="${ROOT}/scripts/run/backends/demos/configs/demo_echo.pipeline.yaml"
VENV_PATH="${VENV_PATH:-$(gage_default_venv_path)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)/demo_echo_oneclick}"
MAX_SAMPLES="${MAX_SAMPLES:-1}"
CONCURRENCY="${CONCURRENCY:-1}"

gage_activate_venv "${VENV_PATH}"
PYTHON_BIN="${PYTHON_BIN:-$(gage_default_python)}"

echo "[scripts/run] running dummy echo pipeline"
"${PYTHON_BIN}" "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --max-samples "${MAX_SAMPLES}" \
  --concurrency "${CONCURRENCY}"

echo "[scripts/run] done. artifacts -> ${OUTPUT_DIR}"
