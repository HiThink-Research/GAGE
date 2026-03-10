#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODE="${MODE:-human-vs-dummy}"
ARGS=()

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run/arenas/vizdoom/run.sh --mode <mode> [legacy-script-args...]

Modes:
  human-vs-dummy
  human-solo
  human-vs-llm
  human-vs-llm-record
  llm-vs-llm
  ai-vs-ai
  agent-vs-llm
  human-vs-strategies

Examples:
  bash scripts/run/arenas/vizdoom/run.sh --mode human-vs-llm
  bash scripts/run/arenas/vizdoom/run.sh --mode llm-vs-llm
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
  human-vs-dummy) TARGET="${SCRIPT_DIR}/run_human_vs_dummy_legacy.sh" ;;
  human-solo) TARGET="${SCRIPT_DIR}/run_human_solo_legacy.sh" ;;
  human-vs-llm) TARGET="${SCRIPT_DIR}/run_human_vs_llm_legacy.sh" ;;
  human-vs-llm-record) TARGET="${SCRIPT_DIR}/run_human_vs_llm_record_legacy.sh" ;;
  llm-vs-llm) TARGET="${SCRIPT_DIR}/run_llm_vs_llm_legacy.sh" ;;
  ai-vs-ai) TARGET="${SCRIPT_DIR}/run_ai_vs_ai_legacy.sh" ;;
  agent-vs-llm) TARGET="${SCRIPT_DIR}/run_agent_vs_llm_legacy.sh" ;;
  human-vs-strategies) TARGET="${SCRIPT_DIR}/run_human_vs_strategies_legacy.sh" ;;
  *)
    echo "[vizdoom][error] Unsupported mode: ${MODE}" >&2
    usage >&2
    exit 1
    ;;
esac

exec bash "${TARGET}" "${ARGS[@]}"
