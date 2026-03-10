#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-showdown}"
ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run/arenas/doudizhu/run.sh --mode <mode> [legacy-script-args...]

Modes:
  showdown
  human-vs-ai

Examples:
  bash scripts/run/arenas/doudizhu/run.sh --mode showdown
  bash scripts/run/arenas/doudizhu/run.sh --mode human-vs-ai
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --mode=*)
      MODE="${1#*=}"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      ARGS+=("$1")
      shift
      ;;
  esac
done

case "${MODE}" in
  showdown) TARGET="${SCRIPT_DIR}/run_showdown_legacy.sh" ;;
  human-vs-ai) TARGET="${SCRIPT_DIR}/run_human_vs_ai_legacy.sh" ;;
  *)
    echo "[doudizhu][error] Unsupported mode: ${MODE}" >&2
    usage >&2
    exit 1
    ;;
esac

exec bash "${TARGET}" "${ARGS[@]}"
