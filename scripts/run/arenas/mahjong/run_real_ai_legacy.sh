#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
PYTHON_BIN="${PYTHON_BIN:-$(gage_default_python)}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)}"
VISUAL_PORT="${VISUAL_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
SAMPLE_ID="${SAMPLE_ID:-mahjong_real_ai_0001}"
FRONTEND_DIR="${ROOT}/frontend/arena-visual"

mkdir -p "${OUTPUT_DIR}"
export GAGE_EVAL_RUN_ID="${RUN_ID}"
export GAGE_EVAL_SAMPLE_ID="${SAMPLE_ID}"

# Port picking utilities
is_port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
  return 1
}

pick_port() {
  local base_port="$1"
  local max_tries="${2:-50}"
  local port="${base_port}"
  local idx=0
  while [ "${idx}" -lt "${max_tries}" ]; do
    if ! is_port_in_use "${port}"; then
      echo "${port}"
      return 0
    fi
    port=$((port + 1))
    idx=$((idx + 1))
  done
  return 1
}

wait_for_port() {
  local port="$1"
  local timeout="${2:-60}"
  local elapsed=0
  while [ "${elapsed}" -lt "${timeout}" ]; do
    if is_port_in_use "${port}"; then
      return 0
    fi
    sleep 1
    elapsed=$((elapsed + 1))
  done
  return 1
}

cleanup_old_processes() {
  echo "[mahjong] cleaning up old processes on ports ${VISUAL_PORT} and ${FRONTEND_PORT}..."
  for port in "${VISUAL_PORT}" "${FRONTEND_PORT}"; do
    if is_port_in_use "${port}"; then
      if command -v lsof >/dev/null 2>&1; then
        lsof -tiTCP:"${port}" -sTCP:LISTEN | xargs kill -9 2>/dev/null || true
      fi
    fi
  done
  # Clean up any orphaned game loop scripts
  pkill -f run.py || true
}

# Run cleanup before picking new ports
cleanup_old_processes

# Select Ports
VISUAL_PORT_SELECTED="$(pick_port "${VISUAL_PORT}" 50)"
if [ -z "${VISUAL_PORT_SELECTED}" ]; then
  echo "[mahjong][error] Unable to find a free visual port starting at ${VISUAL_PORT}." >&2
  exit 1
fi
VISUAL_PORT="${VISUAL_PORT_SELECTED}"

FRONTEND_PORT_SELECTED="$(pick_port "${FRONTEND_PORT}" 50)"
if [ -z "${FRONTEND_PORT_SELECTED}" ]; then
  echo "[mahjong][error] Unable to find a free frontend port starting at ${FRONTEND_PORT}." >&2
  exit 1
fi
FRONTEND_PORT="${FRONTEND_PORT_SELECTED}"

echo "[mahjong] starting visual server on port ${VISUAL_PORT}..."
PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -m gage_eval.tools.arena_visual_server \
  --port "${VISUAL_PORT}" \
  --arena-visual-dir "${OUTPUT_DIR}" &
VISUAL_PID=$!

echo "[mahjong] starting arena visual frontend on port ${FRONTEND_PORT}..."
(
  cd "${FRONTEND_DIR}"
  VITE_ARENA_GATEWAY_BASE_URL="http://127.0.0.1:${VISUAL_PORT}" \
    NODE_OPTIONS="--openssl-legacy-provider" \
    BROWSER=none \
    PORT="${FRONTEND_PORT}" \
    npm run start
) > /dev/null 2>&1 &
FRONTEND_PID=$!

# Wait for frontend to be ready
echo "[mahjong] waiting for frontend to be ready..."
if wait_for_port "${FRONTEND_PORT}" 60; then
    echo "[mahjong] Frontend is ready."
else
    echo "[mahjong][warn] Frontend did not start within 60s." >&2
fi

# Start Game Generation
echo "[mahjong] starting Real AI (Pipeline) Game Generation..."
if [ -z "${OPENAI_API_KEY:-}" ]; then
    echo "=========================================================="
    echo "WARNING: OPENAI_API_KEY is not set."
    echo "The pipeline may fail if you are using OpenAI models."
    echo "Please export OPENAI_API_KEY='sk-...' before running."
    echo "=========================================================="
fi

# Run the pipeline once. 
# Note: unlike the demo loop, this runs once and finishes. 
# We run it in the background so the script stays alive for the server.
"${PYTHON_BIN}" "${ROOT}/run.py" \
  --config "${ROOT}/config/custom/mahjong/mahjong_4_ai.yaml" \
  --output-dir "${OUTPUT_DIR}" &
GAME_PID=$!

REPLAY_URL="http://localhost:${FRONTEND_PORT}/sessions/${SAMPLE_ID}?run_id=${RUN_ID}"

echo "=================================="
echo "Mahjong Real AI Demo Running at: ${REPLAY_URL}"
echo "=================================="

# Auto-open browser
if command -v open >/dev/null 2>&1; then
    open "${REPLAY_URL}"
elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${REPLAY_URL}"
fi

# Wait for user interrupt with a more robust trap
cleanup() {
    echo ""
    echo "[mahjong] shutting down components..."
    # Kill the entire process group if possible or target specifically
    kill ${VISUAL_PID} ${FRONTEND_PID} ${GAME_PID} 2>/dev/null || true
    # Extra insurance for Node/React children
    pkill -P ${FRONTEND_PID} 2>/dev/null || true
    exit 0
}

trap cleanup SIGINT SIGTERM
wait
