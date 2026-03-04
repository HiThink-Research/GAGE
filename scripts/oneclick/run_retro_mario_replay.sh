#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${1:-${RUN_ID:-}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5800}"
FPS="${FPS:-12}"
MAX_FRAMES="${MAX_FRAMES:-0}"
AUTO_OPEN="${AUTO_OPEN:-0}"

if [[ -z "${RUN_ID}" ]]; then
  cat <<'MSG' >&2
Usage:
  bash scripts/oneclick/run_retro_mario_replay.sh <run_id>

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

SAMPLES_DIR="${ROOT}/runs/${RUN_ID}/samples"
if [[ ! -d "${SAMPLES_DIR}" ]]; then
  echo "[retro_mario][replay][error] samples directory not found: ${SAMPLES_DIR}" >&2
  exit 1
fi

SAMPLE_JSON="$(find "${SAMPLES_DIR}" -name '*.json' | sort | head -n 1)"
if [[ -z "${SAMPLE_JSON}" ]]; then
  echo "[retro_mario][replay][error] no sample json found under: ${SAMPLES_DIR}" >&2
  exit 1
fi

cat <<MSG
[retro_mario][replay] Starting replay server.
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
  --auto-open "${AUTO_OPEN}"
