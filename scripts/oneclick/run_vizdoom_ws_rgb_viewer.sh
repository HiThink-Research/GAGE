#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CFG="${CFG:-${ROOT}/config/custom/vizdoom_dummy_vs_dummy_ws_rgb.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/runs}"
RUN_ID="${RUN_ID:-vizdoom_dummy_ws_rgb_$(date +%Y%m%d_%H%M%S)}"
WS_RGB_HOST="${WS_RGB_HOST:-127.0.0.1}"
WS_RGB_PORT="${WS_RGB_PORT:-5800}"
AUTO_OPEN="${AUTO_OPEN:-1}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-45}"

if [ -n "${PYTHON_BIN:-}" ]; then
  PYTHON_EXEC="${PYTHON_BIN}"
elif [ -x "${ROOT}/.venv/bin/python" ]; then
  PYTHON_EXEC="${ROOT}/.venv/bin/python"
else
  PYTHON_EXEC="python3"
fi

export VIZDOOM_LITELLM_PROVIDER="${VIZDOOM_LITELLM_PROVIDER:-openai}"
export VIZDOOM_LITELLM_API_BASE="${VIZDOOM_LITELLM_API_BASE:-http://10.217.219.2:2722/v1}"
export VIZDOOM_LITELLM_MODEL="${VIZDOOM_LITELLM_MODEL:-/mnt/model/qwen3_omni_30b/}"
export VIZDOOM_LITELLM_API_KEY="${VIZDOOM_LITELLM_API_KEY:-${LITELLM_API_KEY:-${OPENAI_API_KEY:-empty}}}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${VIZDOOM_LITELLM_API_KEY}}"
export LITELLM_API_KEY="${LITELLM_API_KEY:-${VIZDOOM_LITELLM_API_KEY}}"

if [ ! -f "${CFG}" ]; then
  echo "[ws_rgb][error] Config not found: ${CFG}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

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

open_url() {
  local url="$1"
  if [ "${AUTO_OPEN}" = "0" ]; then
    return 0
  fi
  if command -v open >/dev/null 2>&1; then
    open "${url}" || true
    return 0
  fi
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${url}" || true
    return 0
  fi
  echo "[ws_rgb][warn] Auto-open unavailable. Open manually: ${url}" >&2
  return 1
}

wait_for_viewer() {
  local url="$1"
  local timeout_s="$2"
  local waited=0
  while [ "${waited}" -lt "${timeout_s}" ]; do
    if curl -sf "${url}" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

SELECTED_PORT="$(pick_port "${WS_RGB_PORT}" 50 || true)"
if [ -z "${SELECTED_PORT}" ]; then
  echo "[ws_rgb][error] Unable to find a free ws_rgb port from ${WS_RGB_PORT}." >&2
  exit 1
fi
if [ "${SELECTED_PORT}" != "${WS_RGB_PORT}" ]; then
  echo "[ws_rgb][warn] WS_RGB_PORT ${WS_RGB_PORT} in use. Using ${SELECTED_PORT}." >&2
fi
WS_RGB_PORT="${SELECTED_PORT}"

VIEWER_URL="http://${WS_RGB_HOST}:${WS_RGB_PORT}/ws_rgb/viewer"
LOG_FILE="${OUTPUT_DIR}/${RUN_ID}.ws_rgb.log"

echo "[ws_rgb] python: ${PYTHON_EXEC}"
echo "[ws_rgb] config: ${CFG}"
echo "[ws_rgb] run_id: ${RUN_ID}"
echo "[ws_rgb] ws_rgb_host: ${WS_RGB_HOST}"
echo "[ws_rgb] ws_rgb_port: ${WS_RGB_PORT}"
echo "[ws_rgb] log: ${LOG_FILE}"
echo "[ws_rgb] viewer: ${VIEWER_URL}"
echo "[ws_rgb] model_base: ${VIZDOOM_LITELLM_API_BASE}"
echo "[ws_rgb] model_name: ${VIZDOOM_LITELLM_MODEL}"

(
  PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  WS_RGB_HOST="${WS_RGB_HOST}" \
  WS_RGB_PORT="${WS_RGB_PORT}" \
  "${PYTHON_EXEC}" "${ROOT}/run.py" \
    --config "${CFG}" \
    --output-dir "${OUTPUT_DIR}" \
    --run-id "${RUN_ID}" 2>&1 | tee "${LOG_FILE}"
) &
RUN_PID=$!

cleanup() {
  if kill -0 "${RUN_PID}" >/dev/null 2>&1; then
    kill "${RUN_PID}" >/dev/null 2>&1 || true
    wait "${RUN_PID}" || true
  fi
}

trap cleanup SIGINT SIGTERM

if wait_for_viewer "${VIEWER_URL}" "${WAIT_TIMEOUT_S}"; then
  echo "[ws_rgb] viewer ready: ${VIEWER_URL}"
  open_url "${VIEWER_URL}" || true
else
  echo "[ws_rgb][warn] Viewer not ready within ${WAIT_TIMEOUT_S}s." >&2
  echo "[ws_rgb][hint] Check log: ${LOG_FILE}" >&2
fi

wait "${RUN_PID}"
EXIT_CODE=$?
echo "[ws_rgb] run exited with code ${EXIT_CODE}"
exit "${EXIT_CODE}"
