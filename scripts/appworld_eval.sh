#!/usr/bin/env bash
set -euo pipefail

# Run the AppWorld official JSONL evaluation pipeline.

usage() {
  cat <<'USAGE'
Usage:
  appworld_eval.sh [--config PATH] [--output-dir DIR] [--run-id ID]

Defaults:
  --config     config/custom/appworld_official_jsonl.yaml
  --output-dir ${GAGE_EVAL_SAVE_DIR:-./runs}
  --run-id     appworld_official_jsonl
USAGE
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${ROOT_DIR}/config/custom/appworld_official_jsonl.yaml"
OUTPUT_DIR="${GAGE_EVAL_SAVE_DIR:-${ROOT_DIR}/runs}"
RUN_ID="${APPWORLD_RUN_ID:-appworld_official_jsonl}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --run-id)
      RUN_ID="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "Config not found: ${CONFIG_PATH}" >&2
  exit 1
fi

PYTHONPATH="${ROOT_DIR}/src" python "${ROOT_DIR}/run.py" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"
