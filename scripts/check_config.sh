#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEFAULT_TEMPLATE_DIR="${ROOT_DIR}/config/templates"
export PYTHONPATH="${ROOT_DIR}/src:${PYTHONPATH:-}"

CONFIGS=()
if [[ $# -gt 0 ]]; then
  CONFIGS=("$@")
else
  for dir in "${DEFAULT_TEMPLATE_DIR}/builtin" "${DEFAULT_TEMPLATE_DIR}/custom"; do
    [[ -d "${dir}" ]] || continue
    while IFS= read -r line; do
      CONFIGS+=("$line")
    done < <(find "${dir}" -type f -name "*.yaml" | sort)
  done
fi

if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "[gage-eval] no config files provided or discovered" >&2
  exit 1
fi

for cfg in "${CONFIGS[@]}"; do
  echo "[gage-eval] validating ${cfg}"
  python -m gage_eval.tools.config_checker --config "${cfg}"
done
