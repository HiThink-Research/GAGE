#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"
GAME="${GAME:-}"
MODE="${MODE:-dummy}"
HOST="${HOST:-127.0.0.1}"
WS_RGB_PORT="${WS_RGB_PORT:-5800}"
FPS="${FPS:-12}"
AUTO_OPEN="${AUTO_OPEN:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$(gage_default_runs_dir)}"
RUN_ID="${RUN_ID:-}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-45}"
KEEP_TMP="${KEEP_TMP:-0}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run/arenas/replay/run_and_open.sh --game <game> [--mode dummy|ai] [options]

Required:
  --game <game>                One of: gomoku, tictactoe, doudizhu, mahjong, pettingzoo

Optional:
  --mode <mode>                dummy (default) | ai
  --run-id <run_id>            Run id (default auto-generated)
  --output-dir <dir>           Output directory (default: runs)
  --host <host>                Replay host (default: 127.0.0.1)
  --port <port>                Replay ws_rgb port (default: 5800)
  --fps <fps>                  Replay fps (default: 12)
  --auto-open <0|1>            Auto open browser (default: 1)
  --wait-timeout-s <seconds>   Viewer readiness timeout (default: 45)
  --keep-tmp <0|1>             Keep temporary config directory (default: 0)
  --python-bin <path>          Python interpreter path
  -h, --help                   Show this help

Examples:
  bash scripts/run/arenas/replay/run_and_open.sh --game gomoku --mode dummy
  OPENAI_API_KEY=... bash scripts/run/arenas/replay/run_and_open.sh --game mahjong --mode ai
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
      WS_RGB_PORT="${2:-}"
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
    --wait-timeout-s)
      WAIT_TIMEOUT_S="${2:-}"
      shift 2
      ;;
    --keep-tmp)
      KEEP_TMP="${2:-}"
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
      echo "[oneclick][error] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${GAME}" ]]; then
  echo "[oneclick][error] --game is required." >&2
  usage
  exit 1
fi

case "${GAME}" in
  gomoku|tictactoe|doudizhu|mahjong|pettingzoo) ;;
  *)
    echo "[oneclick][error] Unsupported game: ${GAME}" >&2
    exit 1
    ;;
esac

case "${MODE}" in
  dummy|ai) ;;
  *)
    echo "[oneclick][error] Unsupported mode: ${MODE}" >&2
    exit 1
    ;;
esac

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_EXEC="${PYTHON_BIN}"
else
  PYTHON_EXEC="$(gage_default_python)"
fi

if ! command -v "${PYTHON_EXEC}" >/dev/null 2>&1; then
  echo "[oneclick][error] python not found: ${PYTHON_EXEC}" >&2
  exit 1
fi

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
  while [[ "${idx}" -lt "${max_tries}" ]]; do
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
  if [[ "${AUTO_OPEN}" = "0" ]]; then
    return 0
  fi
  if command -v open >/dev/null 2>&1; then
    open "${url}" >/dev/null 2>&1 || true
    return 0
  fi
  if command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${url}" >/dev/null 2>&1 || true
    return 0
  fi
  echo "[oneclick][warn] Auto-open unavailable. Open manually: ${url}" >&2
  return 1
}

wait_for_viewer() {
  local base_url="$1"
  local timeout_s="$2"
  local waited=0
  while [[ "${waited}" -lt "${timeout_s}" ]]; do
    if curl -sf "${base_url}/ws_rgb/displays" >/dev/null 2>&1; then
      return 0
    fi
    sleep 1
    waited=$((waited + 1))
  done
  return 1
}

prepare_config() {
  local game="$1"
  local mode="$2"
  local output_path="$3"

  "${PYTHON_EXEC}" - <<'PY' "${game}" "${mode}" "${output_path}" "${ROOT}"
from __future__ import annotations

from pathlib import Path
import sys
import yaml

game = str(sys.argv[1]).strip().lower()
mode = str(sys.argv[2]).strip().lower()
output_path = Path(sys.argv[3]).expanduser().resolve()
repo_root = Path(sys.argv[4]).expanduser().resolve()
output_path.parent.mkdir(parents=True, exist_ok=True)


def _replay_block(max_frames: int) -> dict[str, object]:
    return {
        "enabled": True,
        "primary_mode": True,
        "frame_capture": {
            "enabled": True,
            "frame_stride": 1,
            "max_frames": int(max_frames),
        },
    }


def _load_yaml(path: Path) -> dict[str, object]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"invalid_config_root:{path}")
    return payload


def _patch_arena_for_replay(
    payload: dict[str, object],
    *,
    target_game: str,
    max_frames: int,
    disable_visualizer: bool,
) -> dict[str, object]:
    adapters = payload.get("role_adapters")
    if not isinstance(adapters, list):
        return payload

    for adapter in adapters:
        if not isinstance(adapter, dict):
            continue
        if adapter.get("role_type") != "arena":
            continue
        params = adapter.setdefault("params", {})
        if not isinstance(params, dict):
            continue
        env = params.setdefault("environment", {})
        if not isinstance(env, dict):
            continue
        env["replay"] = _replay_block(max_frames=max_frames)
        if target_game == "pettingzoo":
            env_kwargs = env.setdefault("env_kwargs", {})
            if isinstance(env_kwargs, dict):
                env_kwargs["render_mode"] = "rgb_array"
        if disable_visualizer:
            visualizer = params.get("visualizer")
            if isinstance(visualizer, dict):
                visualizer["enabled"] = False
                visualizer["launch_browser"] = False
                visualizer["wait_for_finish"] = False
    return payload


dummy_source = {
    "gomoku": "config/custom/oneclick/replay_dummy/gomoku_dummy_replay.yaml",
    "tictactoe": "config/custom/oneclick/replay_dummy/tictactoe_dummy_replay.yaml",
    "doudizhu": "config/custom/oneclick/replay_dummy/doudizhu_dummy_replay.yaml",
    "mahjong": "config/custom/oneclick/replay_dummy/mahjong_dummy_replay.yaml",
    "pettingzoo": "config/custom/oneclick/replay_dummy/pettingzoo_dummy_replay.yaml",
}

dummy_max_frames = {
    "gomoku": 120,
    "tictactoe": 60,
    "doudizhu": 180,
    "mahjong": 180,
    "pettingzoo": 90,
}

ai_source = {
    "gomoku": "config/custom/gomoku/gomoku_litellm_local.yaml",
    "tictactoe": "config/custom/tictactoe/tictactoe_litellm_local.yaml",
    "doudizhu": "config/custom/doudizhu/doudizhu_litellm_local.yaml",
    "mahjong": "config/custom/mahjong/mahjong_4_ai.yaml",
    "pettingzoo": "config/custom/pettingzoo/pong_ai.yaml",
}

if mode == "dummy":
    source_path = (repo_root / dummy_source[game]).resolve()
    config = _load_yaml(source_path)
    config = _patch_arena_for_replay(
        config,
        target_game=game,
        max_frames=int(dummy_max_frames[game]),
        disable_visualizer=False,
    )
else:
    source_path = (repo_root / ai_source[game]).resolve()
    config = _load_yaml(source_path)
    config = _patch_arena_for_replay(
        config,
        target_game=game,
        max_frames=300,
        disable_visualizer=True,
    )

output_path.write_text(
    yaml.safe_dump(config, allow_unicode=False, sort_keys=False),
    encoding="utf-8",
)
print(output_path)
PY
}

STAMP="$(date +%Y%m%d_%H%M%S)"
if [[ -z "${RUN_ID}" ]]; then
  RUN_ID="oneclick_${GAME}_${MODE}_${STAMP}"
fi

SELECTED_PORT="$(pick_port "${WS_RGB_PORT}" 50 || true)"
if [[ -z "${SELECTED_PORT}" ]]; then
  echo "[oneclick][error] Unable to pick a free ws_rgb port from ${WS_RGB_PORT}" >&2
  exit 1
fi
if [[ "${SELECTED_PORT}" != "${WS_RGB_PORT}" ]]; then
  echo "[oneclick][warn] Port ${WS_RGB_PORT} is in use; switched to ${SELECTED_PORT}"
fi
WS_RGB_PORT="${SELECTED_PORT}"

WORK_DIR="$(mktemp -d "/tmp/gage_replay_${GAME}_${MODE}_XXXXXX")"
CONFIG_PATH="${WORK_DIR}/${GAME}_${MODE}.yaml"

cleanup() {
  if [[ -n "${REPLAY_PID:-}" ]] && kill -0 "${REPLAY_PID}" >/dev/null 2>&1; then
    kill "${REPLAY_PID}" >/dev/null 2>&1 || true
    wait "${REPLAY_PID}" >/dev/null 2>&1 || true
  fi
  if [[ "${KEEP_TMP}" != "1" ]]; then
    rm -rf "${WORK_DIR}" >/dev/null 2>&1 || true
  else
    echo "[oneclick] keep temp dir: ${WORK_DIR}"
  fi
}
trap cleanup EXIT SIGINT SIGTERM

echo "[oneclick] game=${GAME} mode=${MODE}"
echo "[oneclick] python=${PYTHON_EXEC}"
echo "[oneclick] work_dir=${WORK_DIR}"

prepare_config "${GAME}" "${MODE}" "${CONFIG_PATH}" >/dev/null

mkdir -p "${OUTPUT_DIR}"
RUN_LOG="${OUTPUT_DIR}/${RUN_ID}.log"
echo "[oneclick] config=${CONFIG_PATH}"
echo "[oneclick] run_id=${RUN_ID}"
echo "[oneclick] output_dir=${OUTPUT_DIR}"
echo "[oneclick] run_log=${RUN_LOG}"

"${PYTHON_EXEC}" "${ROOT}/run.py" \
  --config "${CONFIG_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --run-id "${RUN_ID}" 2>&1 | tee "${RUN_LOG}"

SAMPLE_JSON="$(find "${OUTPUT_DIR}/${RUN_ID}/samples" -name '*.json' | head -n 1)"
if [[ -z "${SAMPLE_JSON}" ]]; then
  REPLAY_JSON="$(find "${OUTPUT_DIR}/${RUN_ID}/replays" -name 'replay.json' | head -n 1)"
  if [[ -z "${REPLAY_JSON}" ]]; then
    echo "[oneclick][error] sample json and replay json are both missing for run_id=${RUN_ID}" >&2
    exit 1
  fi
  SAMPLE_JSON="${WORK_DIR}/synthetic_sample_for_replay.json"
  "${PYTHON_EXEC}" - <<'PY' "${CONFIG_PATH}" "${REPLAY_JSON}" "${SAMPLE_JSON}" "${GAME}"
from __future__ import annotations

import json
from pathlib import Path
import sys
import yaml

config_path = Path(sys.argv[1]).resolve()
replay_json = Path(sys.argv[2]).resolve()
sample_json = Path(sys.argv[3]).resolve()
fallback_game = str(sys.argv[4] or "game").strip()

task_id = f"{fallback_game}_replay"
sample_id = replay_json.parent.name
try:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
except Exception:
    cfg = {}
if isinstance(cfg, dict):
    tasks = cfg.get("tasks")
    if isinstance(tasks, list) and tasks:
        first = tasks[0]
        if isinstance(first, dict) and first.get("task_id"):
            task_id = str(first.get("task_id"))

payload = {
    "task_id": task_id,
    "sample": {
        "id": sample_id,
        "task_id": task_id,
        "metadata": {"player_ids": ["player_0"]},
        "predict_result": [
            {
                "index": 0,
                "replay_path": str(replay_json),
            }
        ],
    },
}
sample_json.write_text(
    json.dumps(payload, ensure_ascii=False, indent=2),
    encoding="utf-8",
)
print(sample_json)
PY
  echo "[oneclick] sample json not found; synthesized replay sample at ${SAMPLE_JSON}"
fi

VIEWER_URL="http://${HOST}:${WS_RGB_PORT}/ws_rgb/viewer"
echo "[oneclick] sample_json=${SAMPLE_JSON}"
echo "[oneclick] launching replay viewer at ${VIEWER_URL}"

(
  PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_EXEC}" -m gage_eval.tools.ws_rgb_replay \
    --sample-json "${SAMPLE_JSON}" \
    --host "${HOST}" \
    --port "${WS_RGB_PORT}" \
    --fps "${FPS}"
) &
REPLAY_PID=$!

if wait_for_viewer "http://${HOST}:${WS_RGB_PORT}" "${WAIT_TIMEOUT_S}"; then
  echo "[oneclick] viewer ready: ${VIEWER_URL}"
  open_url "${VIEWER_URL}" || true
else
  echo "[oneclick][warn] viewer did not become ready within ${WAIT_TIMEOUT_S}s"
fi

echo "[oneclick] press Ctrl+C to stop replay server"
wait "${REPLAY_PID}"
