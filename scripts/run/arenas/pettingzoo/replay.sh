#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
PYTHON_BIN="${PYTHON_BIN:-$(gage_default_python)}"
RUN_ID="${1:-${RUN_ID:-}}"
RUNS_DIR="${RUNS_DIR:-${OUTPUT_DIR:-$(gage_default_runs_dir)}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5800}"
FPS="${FPS:-12}"
MAX_FRAMES="${MAX_FRAMES:-0}"
AUTO_OPEN="${AUTO_OPEN:-0}"

if [[ -z "${RUN_ID}" ]]; then
  cat <<'MSG' >&2
Usage:
  bash scripts/run/arenas/pettingzoo/replay.sh <run_id>

Optional environment variables:
  PYTHON_BIN   Python executable used to launch replay server
  HOST         Bind host (default: 127.0.0.1)
  PORT         Bind port (default: 5800)
  FPS          Replay playback fps (default: 12)
  MAX_FRAMES   Replay frame limit, 0 means unlimited
  AUTO_OPEN    Auto open browser viewer (0/1, default: 0)
MSG
  exit 1
fi

SAMPLES_DIR="${RUNS_DIR}/${RUN_ID}/samples"
if [[ ! -d "${SAMPLES_DIR}" ]]; then
  echo "[pettingzoo][replay][error] samples directory not found: ${SAMPLES_DIR}" >&2
  exit 1
fi

SAMPLE_JSON="$(find "${SAMPLES_DIR}" -name '*.json' | sort | head -n 1)"
if [[ -z "${SAMPLE_JSON}" ]]; then
  echo "[pettingzoo][replay][error] no sample json found under: ${SAMPLES_DIR}" >&2
  exit 1
fi

cat <<MSG
[pettingzoo][replay] Starting replay server.
Run ID: ${RUN_ID}
Sample: ${SAMPLE_JSON}
Viewer: http://${HOST}:${PORT}/ws_rgb/viewer
Python: ${PYTHON_BIN}
MSG

PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m gage_eval.tools.ws_rgb_replay \
  --sample-json "${SAMPLE_JSON}" \
  --host "${HOST}" \
  --port "${PORT}" \
  --fps "${FPS}" \
  --max-frames "${MAX_FRAMES}" \
  --game pettingzoo \
  --auto-open "${AUTO_OPEN}"
