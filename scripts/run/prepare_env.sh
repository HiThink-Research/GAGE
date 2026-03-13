#!/usr/bin/env bash
set -euo pipefail

# 创建并填充虚拟环境，安装项目依赖

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
VENV_PATH="${VENV_PATH:-$(gage_default_venv_path)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [ ! -d "${VENV_PATH}" ]; then
  echo "[oneclick] creating venv at ${VENV_PATH}"
  "${PYTHON_BIN}" -m venv "${VENV_PATH}"
else
  echo "[oneclick] venv already exists at ${VENV_PATH}"
fi

source "${VENV_PATH}/bin/activate"

echo "[oneclick] upgrading pip"
pip install --upgrade pip >/dev/null

echo "[oneclick] installing requirements"
pip install -r "${ROOT}/requirements.txt"

echo "[oneclick] env ready -> source ${VENV_PATH}/bin/activate to use it"
