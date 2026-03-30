#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-human-vs-ai}"
ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run/arenas/doudizhu/run.sh --mode <mode> [legacy-script-args...]

Modes:
  human-vs-ai

Examples:
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
  human-vs-ai) TARGET="${SCRIPT_DIR}/run_human_vs_ai_legacy.sh" ;;
  *)
    echo "[doudizhu][error] Unsupported mode: ${MODE}" >&2
    usage >&2
    exit 1
    ;;
esac

# NOTE: Bash 3.2 with nounset treats "${ARGS[@]}" as unbound when the array is empty.
if (( ${#ARGS[@]} > 0 )); then
  exec bash "${TARGET}" "${ARGS[@]}"
fi

exec bash "${TARGET}"
