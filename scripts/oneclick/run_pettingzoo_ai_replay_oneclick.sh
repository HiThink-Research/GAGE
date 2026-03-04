#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GAME="${GAME:-space_invaders}"
MODE="${MODE:-ai}"
CONFIG="${CONFIG:-}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs}"
RUN_ID="${RUN_ID:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5800}"
FPS="${FPS:-12}"
AUTO_OPEN="${AUTO_OPEN:-1}"
DRY_RUN="${DRY_RUN:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/oneclick/run_pettingzoo_ai_replay_oneclick.sh [options]

Options:
  --game <game>                 PettingZoo game key (default: space_invaders)
  --mode <mode>                 Config mode suffix (default: ai)
  --config <yaml>               Explicit config path override
  --run-id <run_id>             Run id (default: pz_<game>_<mode>_<epoch>)
  --output-dir <dir>            Output directory (default: runs)
  --host <host>                 Replay host (default: 127.0.0.1)
  --port <port>                 Replay port (default: 5800)
  --fps <fps>                   Replay fps (default: 12)
  --auto-open <0|1>             Auto open viewer (default: 1)
  --python-bin <path>           Python executable
  --dry-run <0|1>               Print commands only, do not execute (default: 0)
  -h, --help                    Show this help

Environment notes:
  GAGE_EVAL_GAME_LOG_INLINE_LIMIT defaults to -1 when unset.
  GAGE_EVAL_GAME_LOG_INLINE_BYTES defaults to 0 when unset.
  OPENAI_API_KEY / OPENAI_BASE_URL are passed through as-is.

Example:
  bash scripts/oneclick/run_pettingzoo_ai_replay_oneclick.sh \
    --game space_invaders --mode ai --port 5800 --fps 12
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
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --port)
      PORT="${2:-}"
      shift 2
      ;;
    --fps)
      FPS="${2:-}"
      shift 2
      ;;
    --auto-open)
      AUTO_OPEN="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[pz-replay][error] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXEC="${PYTHON_BIN}"
elif [[ -x "${ROOT}/.venv/bin/python" ]]; then
  PYTHON_EXEC="${ROOT}/.venv/bin/python"
elif [[ -x "/Users/shuo/mamba/envs/gage/bin/python" ]]; then
  PYTHON_EXEC="/Users/shuo/mamba/envs/gage/bin/python"
else
  PYTHON_EXEC="python3"
fi

if ! command -v "${PYTHON_EXEC}" >/dev/null 2>&1; then
  echo "[pz-replay][error] python not found: ${PYTHON_EXEC}" >&2
  exit 1
fi

if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="pz_${GAME}_${MODE}_$(date +%s)"
fi

if [[ -z "${CONFIG}" ]]; then
  CONFIG="${ROOT}/config/custom/pettingzoo/${GAME}_${MODE}.yaml"
fi

export GAGE_EVAL_GAME_LOG_INLINE_LIMIT="${GAGE_EVAL_GAME_LOG_INLINE_LIMIT:--1}"
export GAGE_EVAL_GAME_LOG_INLINE_BYTES="${GAGE_EVAL_GAME_LOG_INLINE_BYTES:-0}"

PYTHONPATH_VALUE="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
RUN_OUTPUT_DIR="${OUTPUT_DIR}/${RUN_ID}"
SAMPLES_DIR="${RUN_OUTPUT_DIR}/samples"
VIEWER_URL="http://${HOST}:${PORT}/ws_rgb/viewer"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  "$@"
}

if [[ "${DRY_RUN}" == "1" ]]; then
  cat <<EOF
[pz-replay] dry-run mode enabled.
[pz-replay] Config: ${CONFIG}
[pz-replay] Run ID: ${RUN_ID}
[pz-replay] Output: ${RUN_OUTPUT_DIR}
[pz-replay] Viewer: ${VIEWER_URL}
EOF
  printf '+ export GAGE_EVAL_GAME_LOG_INLINE_LIMIT=%q\n' "${GAGE_EVAL_GAME_LOG_INLINE_LIMIT}"
  printf '+ export GAGE_EVAL_GAME_LOG_INLINE_BYTES=%q\n' "${GAGE_EVAL_GAME_LOG_INLINE_BYTES}"
  printf '+ env PYTHONPATH=%q %q %q --config %q --output-dir %q --run-id %q\n' \
    "${PYTHONPATH_VALUE}" "${PYTHON_EXEC}" "${ROOT}/run.py" "${CONFIG}" "${OUTPUT_DIR}" "${RUN_ID}"
  printf '+ SAMPLE_JSON="$(find %q -type f -name \"*.json\" | sort | head -n 1)"\n' "${SAMPLES_DIR}"
  printf '+ env PYTHONPATH=%q %q -m gage_eval.tools.ws_rgb_replay --sample-json \"$SAMPLE_JSON\" --game pettingzoo --host %q --port %q --fps %q --auto-open %q\n' \
    "${PYTHONPATH_VALUE}" "${PYTHON_EXEC}" "${HOST}" "${PORT}" "${FPS}" "${AUTO_OPEN}"
  printf '+ echo %q\n' "${VIEWER_URL}"
  exit 0
fi

if [[ ! -f "${CONFIG}" ]]; then
  echo "[pz-replay][error] Config not found: ${CONFIG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[pz-replay] python: ${PYTHON_EXEC}"
echo "[pz-replay] game: ${GAME}"
echo "[pz-replay] mode: ${MODE}"
echo "[pz-replay] config: ${CONFIG}"
echo "[pz-replay] run_id: ${RUN_ID}"
echo "[pz-replay] output_dir: ${OUTPUT_DIR}"
echo "[pz-replay] replay: ${VIEWER_URL}"
echo "[pz-replay] GAGE_EVAL_GAME_LOG_INLINE_LIMIT=${GAGE_EVAL_GAME_LOG_INLINE_LIMIT}"
echo "[pz-replay] GAGE_EVAL_GAME_LOG_INLINE_BYTES=${GAGE_EVAL_GAME_LOG_INLINE_BYTES}"

run_cmd env "PYTHONPATH=${PYTHONPATH_VALUE}" "${PYTHON_EXEC}" "${ROOT}/run.py" \
  --config "${CONFIG}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"

if [[ ! -d "${SAMPLES_DIR}" ]]; then
  echo "[pz-replay][error] samples directory not found: ${SAMPLES_DIR}" >&2
  exit 1
fi

SAMPLE_JSON="$(find "${SAMPLES_DIR}" -type f -name "*.json" | sort | head -n 1)"
if [[ -z "${SAMPLE_JSON}" ]]; then
  echo "[pz-replay][error] no sample json found under: ${SAMPLES_DIR}" >&2
  exit 1
fi

echo "[pz-replay] sample_json: ${SAMPLE_JSON}"
echo "[pz-replay] viewer: ${VIEWER_URL}"

run_cmd env "PYTHONPATH=${PYTHONPATH_VALUE}" "${PYTHON_EXEC}" -m gage_eval.tools.ws_rgb_replay \
  --sample-json "${SAMPLE_JSON}" \
  --game pettingzoo \
  --host "${HOST}" \
  --port "${PORT}" \
  --fps "${FPS}" \
  --auto-open "${AUTO_OPEN}"

