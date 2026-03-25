#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
DEFAULT_TEMPLATE_DIR="${ROOT_DIR}/config/builtin_templates"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

resolve_python_bin() {
  if [[ -n "${PYTHON_BIN:-}" && -x "${PYTHON_BIN}" ]]; then
    printf '%s\n' "${PYTHON_BIN}"
    return 0
  fi
  if [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
    printf '%s\n' "${VIRTUAL_ENV}/bin/python"
    return 0
  fi
  if [[ -x "${ROOT_DIR}/.venv/bin/python" ]]; then
    printf '%s\n' "${ROOT_DIR}/.venv/bin/python"
    return 0
  fi
  if [[ -x "${ROOT_DIR}/.venv311/bin/python" ]]; then
    printf '%s\n' "${ROOT_DIR}/.venv311/bin/python"
    return 0
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    command -v python3
    return 0
  fi
  printf '%s\n' "python"
}

PYTHON_BIN="$(resolve_python_bin)"

CONFIGS=()
if [[ $# -gt 0 ]]; then
  CONFIGS=("$@")
else
  if [[ -d "${DEFAULT_TEMPLATE_DIR}" ]]; then
    while IFS= read -r line; do
      CONFIGS+=("$line")
    done < <(find "${DEFAULT_TEMPLATE_DIR}" -type f -name "*.yaml" | sort)
  fi
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "[gage-eval] no config files provided or discovered" >&2
  exit 1
fi

for cfg in "${CONFIGS[@]}"; do
  echo "[gage-eval] validating ${cfg}"
  "${PYTHON_BIN}" -m gage_eval.tools.config_checker --config "${cfg}"
done
