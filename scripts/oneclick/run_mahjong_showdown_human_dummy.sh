#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${ROOT}/.venv/bin/python"
CFG="${CFG:-${ROOT}/config/custom/mahjong_human_vs_3_dummy.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs}"
RUN_ID="${RUN_ID:-mahjong_showdown_human_dummy_$(date +%Y%m%d_%H%M%S)}"
REPLAY_PORT="${REPLAY_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"
ACTION_PORT="${ACTION_PORT:-8001}"
FRONTEND_DIR="${FRONTEND_DIR:-${ROOT}/frontend/rlcard-showdown}"
AUTO_OPEN="${AUTO_OPEN:-1}"

export GAGE_EVAL_RUN_ID="${RUN_ID}"
export GAGE_EVAL_SAVE_DIR="${OUTPUT_DIR}"

if [ ! -x "${PYTHON_BIN}" ]; then
  echo "[oneclick][error] Missing python executable: ${PYTHON_BIN}." >&2
  exit 1
fi

if ! command -v node >/dev/null 2>&1; then
  echo "[oneclick][error] node is required for the Showdown frontend." >&2
  exit 1
fi
if ! command -v npm >/dev/null 2>&1; then
  echo "[oneclick][error] npm is required for the Showdown frontend." >&2
  exit 1
fi
if [ ! -d "${FRONTEND_DIR}/node_modules" ]; then
  echo "[oneclick][error] ${FRONTEND_DIR}/node_modules not found." >&2
  echo "[oneclick][hint] Run: (cd ${FRONTEND_DIR} && npm install)" >&2
  exit 1
fi

FRONTEND_NODE_OPTIONS="${NODE_OPTIONS:-}"
if [ -z "${FRONTEND_NODE_OPTIONS}" ]; then
  FRONTEND_NODE_OPTIONS="--openssl-legacy-provider"
elif [[ "${FRONTEND_NODE_OPTIONS}" != *"--openssl-legacy-provider"* ]]; then
  FRONTEND_NODE_OPTIONS="${FRONTEND_NODE_OPTIONS} --openssl-legacy-provider"
fi

is_port_in_use() {
  local port="$1"
  if command -v lsof >/dev/null 2>&1; then
    lsof -nP -iTCP:"${port}" -sTCP:LISTEN >/dev/null 2>&1
    return $?
  fi
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

open_url() {
  local url="$1"
  if [ "${AUTO_OPEN}" = "0" ]; then
    return 0
  fi
  if command -v open >/dev/null 2>&1; then
    open "${url}"
    return 0
  fi
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${url}"
    return 0
  fi
  echo "[oneclick][warn] Unable to auto-open browser. Open manually: ${url}" >&2
  return 1
}

pick_port() {
  local base_port="$1"
  local max_tries="${2:-50}"
  local reserved="${3:-}"
  local port="${base_port}"
  local idx=0
  while [ "${idx}" -lt "${max_tries}" ]; do
    if [ -n "${reserved}" ] && [ "${port}" = "${reserved}" ]; then
      port=$((port + 1))
      idx=$((idx + 1))
      continue
    fi
    if ! is_port_in_use "${port}"; then
      echo "${port}"
      return 0
    fi
    port=$((port + 1))
    idx=$((idx + 1))
  done
  return 1
}

REPLAY_PORT_SELECTED="$(pick_port "${REPLAY_PORT}" 50)"
if [ -z "${REPLAY_PORT_SELECTED}" ]; then
  echo "[oneclick][error] Unable to find a free replay port starting at ${REPLAY_PORT}." >&2
  exit 1
fi
if [ "${REPLAY_PORT_SELECTED}" != "${REPLAY_PORT}" ]; then
  echo "[oneclick][warn] Replay port ${REPLAY_PORT} in use. Using ${REPLAY_PORT_SELECTED}." >&2
fi
REPLAY_PORT="${REPLAY_PORT_SELECTED}"

FRONTEND_PORT_SELECTED="$(pick_port "${FRONTEND_PORT}" 50)"
if [ -z "${FRONTEND_PORT_SELECTED}" ]; then
  echo "[oneclick][error] Unable to find a free frontend port starting at ${FRONTEND_PORT}." >&2
  exit 1
fi
if [ "${FRONTEND_PORT_SELECTED}" != "${FRONTEND_PORT}" ]; then
  echo "[oneclick][warn] Frontend port ${FRONTEND_PORT} in use. Using ${FRONTEND_PORT_SELECTED}." >&2
fi
FRONTEND_PORT="${FRONTEND_PORT_SELECTED}"

ACTION_PORT_SELECTED="$(pick_port "${ACTION_PORT}" 50 "${REPLAY_PORT}")"
if [ -z "${ACTION_PORT_SELECTED}" ]; then
  echo "[oneclick][error] Unable to find a free action port starting at ${ACTION_PORT}." >&2
  exit 1
fi
if [ "${ACTION_PORT_SELECTED}" != "${ACTION_PORT}" ]; then
  echo "[oneclick][warn] Action port ${ACTION_PORT} in use. Using ${ACTION_PORT_SELECTED}." >&2
fi
ACTION_PORT="${ACTION_PORT_SELECTED}"

mkdir -p "${OUTPUT_DIR}"

export MAHJONG_ACTION_PORT="${ACTION_PORT}"

ACTION_URL="http://127.0.0.1:${ACTION_PORT}"
ACTION_URL_ENC="$(
  ACTION_URL="${ACTION_URL}" "${PYTHON_BIN}" - <<'PY'
from urllib.parse import quote
import os
print(quote(os.environ["ACTION_URL"], safe=""))
PY
)"
REPLAY_URL="http://127.0.0.1:${FRONTEND_PORT}/replay/mahjong?replay_path=mahjong_replay.json&mode=human&play=1&action_url=${ACTION_URL_ENC}"
echo "[oneclick] replay url: ${REPLAY_URL}"
echo "[oneclick] action url: ${ACTION_URL}/tournament/action"

echo "[oneclick] starting replay server..."
PYTHONPATH="${ROOT}/src" "${PYTHON_BIN}" -m gage_eval.tools.replay_server \
  --port "${REPLAY_PORT}" \
  --replay-dir "${OUTPUT_DIR}" &
REPLAY_PID=$!

echo "[oneclick] starting showdown frontend..."
(
  cd "${FRONTEND_DIR}"
  REACT_APP_GAGE_API_URL="http://127.0.0.1:${REPLAY_PORT}" \
    REACT_APP_GAGE_ACTION_URL="${ACTION_URL}" \
    NODE_OPTIONS="${FRONTEND_NODE_OPTIONS}" \
    BROWSER=none \
    PORT="${FRONTEND_PORT}" \
    npm run start
) &
FRONTEND_PID=$!

if wait_for_port "${FRONTEND_PORT}" 60; then
  open_url "${REPLAY_URL}"
else
  echo "[oneclick][warn] Frontend did not start within 60s. Open manually: ${REPLAY_URL}" >&2
fi

echo "[oneclick] running mahjong dummy pipeline..."
echo "[oneclick] submit moves to the action url when it is your turn."
"${PYTHON_BIN}" "${ROOT}/run.py" \
  --config "${CFG}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}"

echo "[oneclick] replay ready:"
echo "${REPLAY_URL}"

wait "${FRONTEND_PID}"
