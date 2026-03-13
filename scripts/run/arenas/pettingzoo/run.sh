#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
GAME="${GAME:-pong}"
MODE="${MODE:-dummy}"
CONFIG="${CONFIG:-}"
RUN_ID="${RUN_ID:-}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)}"
WS_RGB_PORT="${WS_RGB_PORT:-5800}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run/arenas/pettingzoo/run.sh [options]

Options:
  --game <game>         PettingZoo game id. Default: pong
  --mode <mode>         dummy | ai | ws_dummy | human_record. Default: dummy
  --config <path>       Explicit config path. Overrides --game/--mode mapping.
  --run-id <run_id>     Run id. Default is auto-generated from game/mode/timestamp.
  --output-dir <dir>    Output directory. Default: runs
  --python-bin <path>   Python interpreter path
  -h, --help            Show this help

Examples:
  bash scripts/run/arenas/pettingzoo/run.sh --game boxing --mode dummy
  bash scripts/run/arenas/pettingzoo/run.sh --game space_invaders --mode ai
  bash scripts/run/arenas/pettingzoo/run.sh --mode ws_dummy
  bash scripts/run/arenas/pettingzoo/run.sh --mode human_record
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --game)
      GAME="${2:-}"
      shift 2
      ;;
    --mode)
      MODE="${2:-}"
      shift 2
      ;;
    --config)
      CONFIG="${2:-}"
      shift 2
      ;;
    --run-id)
      RUN_ID="${2:-}"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[pettingzoo][error] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXEC="${PYTHON_BIN}"
else
  PYTHON_EXEC="$(gage_default_python)"
fi

resolve_config() {
  if [[ -n "${CONFIG}" ]]; then
    printf '%s\n' "${CONFIG}"
    return 0
  fi

  case "${MODE}" in
    dummy)
      printf '%s\n' "${ROOT}/config/custom/pettingzoo/${GAME}_dummy.yaml"
      ;;
    ai)
      printf '%s\n' "${ROOT}/config/custom/pettingzoo/${GAME}_ai.yaml"
      ;;
    ws_dummy)
      printf '%s\n' "${ROOT}/config/custom/pettingzoo/${GAME}_dummy_ws_rgb.yaml"
      ;;
    human_record)
      if [[ "${GAME}" != "space_invaders" ]]; then
        echo "[pettingzoo][error] human_record currently supports only game=space_invaders." >&2
        exit 1
      fi
      printf '%s\n' "${ROOT}/config/custom/pettingzoo/space_invaders_human_vs_human_record.yaml"
      ;;
    *)
      echo "[pettingzoo][error] Unsupported mode: ${MODE}" >&2
      exit 1
      ;;
  esac
}

CFG="$(resolve_config)"
if [[ ! -f "${CFG}" ]]; then
  echo "[pettingzoo][error] Config not found: ${CFG}" >&2
  exit 1
fi

if [[ "${MODE}" == "ai" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" && -n "${LITELLM_API_KEY:-}" ]]; then
    export OPENAI_API_KEY="${LITELLM_API_KEY}"
  fi
  if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "[pettingzoo][error] OPENAI_API_KEY or LITELLM_API_KEY is required for ai mode." >&2
    exit 1
  fi
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="pettingzoo_${GAME}_${MODE}_$(date +%Y%m%d_%H%M%S)"
fi

mkdir -p "${OUTPUT_DIR}"

# Keep game_log inside sample JSON so replay scripts work without extra reruns.
export GAGE_EVAL_GAME_LOG_INLINE_LIMIT="${GAGE_EVAL_GAME_LOG_INLINE_LIMIT:--1}"
export GAGE_EVAL_GAME_LOG_INLINE_BYTES="${GAGE_EVAL_GAME_LOG_INLINE_BYTES:-0}"

cat <<MSG
[pettingzoo] Python: ${PYTHON_EXEC}
[pettingzoo] Game: ${GAME}
[pettingzoo] Mode: ${MODE}
[pettingzoo] Config: ${CFG}
[pettingzoo] Output: ${OUTPUT_DIR}
[pettingzoo] Run ID: ${RUN_ID}
MSG

if [[ "${MODE}" == "ws_dummy" ]]; then
  cat <<MSG
[pettingzoo] Viewer: http://127.0.0.1:${WS_RGB_PORT}/ws_rgb/viewer
MSG
fi

if [[ "${MODE}" == "human_record" ]]; then
  cat <<'MSG'
[pettingzoo] Viewer: http://127.0.0.1:5800/ws_rgb/viewer
[pettingzoo] Input queue: http://127.0.0.1:8001
[pettingzoo] Keys:
  player_0 -> Q/W/E/A/S/D
  player_1 -> U/I/O/J/K/L
MSG
fi

PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
WS_RGB_PORT="${WS_RGB_PORT}" \
"${PYTHON_EXEC}" "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"
