#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
# shellcheck disable=SC1091
source "${ROOT}/scripts/run/common/env.sh"

RUN_ID="${RUN_ID:-}"
SAMPLE_ID="${SAMPLE_ID:-}"
RUNS_DIR="${RUNS_DIR:-${OUTPUT_DIR:-$(gage_default_runs_dir)}}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8010}"
AUTO_OPEN="${AUTO_OPEN:-1}"
WAIT_TIMEOUT_S="${WAIT_TIMEOUT_S:-10}"
PYTHON_EXEC="${PYTHON_BIN:-$(gage_default_python)}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/run/arenas/replay/run_and_open.sh --run-id <run_id> [options]

Options:
  --run-id <run_id>       Completed run id under the runs directory.
  --sample-id <sample_id> Arena sample id. Defaults to the first visual session in the run.
  --output-dir <dir>      Runs directory. Default: GAGE_RUNS_DIR or workspace runs.
  --host <host>           Bind host for the Arena Visual server. Default: 127.0.0.1.
  --port <port>           Bind port for the Arena Visual server. Default: 8010.
  --python-bin <path>     Python interpreter path. Default: PYTHON_BIN or detected Python.
  --no-open               Do not open the browser automatically.
  -h, --help              Show this help.

Examples:
  bash scripts/run/arenas/replay/run_and_open.sh --run-id doudizhu_dummy_visual_20260413_120000
  bash scripts/run/arenas/replay/run_and_open.sh --run-id run_a --sample-id sample_0 --port 8011
EOF
}

require_value() {
  local flag="$1"
  local value="${2:-}"
  if [[ -z "${value}" || "${value}" == --* ]]; then
    echo "[arena-visual][error] ${flag} requires a value." >&2
    usage >&2
    exit 1
  fi
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-id)
      require_value "$1" "${2:-}"
      RUN_ID="${2:-}"
      shift 2
      ;;
    --run-id=*)
      VALUE="${1#*=}"
      require_value "--run-id" "${VALUE}"
      RUN_ID="${VALUE}"
      shift
      ;;
    --sample-id)
      require_value "$1" "${2:-}"
      SAMPLE_ID="${2:-}"
      shift 2
      ;;
    --sample-id=*)
      VALUE="${1#*=}"
      require_value "--sample-id" "${VALUE}"
      SAMPLE_ID="${VALUE}"
      shift
      ;;
    --output-dir)
      require_value "$1" "${2:-}"
      RUNS_DIR="${2:-}"
      shift 2
      ;;
    --output-dir=*)
      VALUE="${1#*=}"
      require_value "--output-dir" "${VALUE}"
      RUNS_DIR="${VALUE}"
      shift
      ;;
    --host)
      require_value "$1" "${2:-}"
      HOST="${2:-}"
      shift 2
      ;;
    --host=*)
      VALUE="${1#*=}"
      require_value "--host" "${VALUE}"
      HOST="${VALUE}"
      shift
      ;;
    --port)
      require_value "$1" "${2:-}"
      PORT="${2:-}"
      shift 2
      ;;
    --port=*)
      VALUE="${1#*=}"
      require_value "--port" "${VALUE}"
      PORT="${VALUE}"
      shift
      ;;
    --python-bin)
      require_value "$1" "${2:-}"
      PYTHON_EXEC="${2:-}"
      shift 2
      ;;
    --python-bin=*)
      VALUE="${1#*=}"
      require_value "--python-bin" "${VALUE}"
      PYTHON_EXEC="${VALUE}"
      shift
      ;;
    --no-open)
      AUTO_OPEN=0
      shift
      ;;
    --open)
      AUTO_OPEN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      if [[ -z "${RUN_ID}" ]]; then
        RUN_ID="$1"
        shift
      else
        echo "[arena-visual][error] Unexpected argument: $1" >&2
        usage >&2
        exit 1
      fi
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
  echo "[arena-visual][error] --run-id is required." >&2
  usage >&2
  exit 1
fi

if [[ ! -d "${RUNS_DIR}" ]]; then
  echo "[arena-visual][error] Runs directory not found: ${RUNS_DIR}" >&2
  exit 1
fi
RUNS_DIR="$(cd "${RUNS_DIR}" && pwd)"
RUN_DIR="${RUNS_DIR}/${RUN_ID}"
if [[ ! -d "${RUN_DIR}" ]]; then
  echo "[arena-visual][error] Run directory not found: ${RUN_DIR}" >&2
  exit 1
fi

find_first_sample_id() {
  if [[ ! -d "${RUN_DIR}/replays" ]]; then
    return 0
  fi
  find "${RUN_DIR}/replays" -path "*/arena_visual_session/v1/manifest.json" -type f 2>/dev/null \
    | sort \
    | head -n 1 \
    | sed -E 's#^.*/replays/([^/]+)/arena_visual_session/v1/manifest\.json$#\1#'
}

if [[ -z "${SAMPLE_ID}" ]]; then
  SAMPLE_ID="$(find_first_sample_id)"
fi

if [[ -z "${SAMPLE_ID}" ]]; then
  echo "[arena-visual][error] No Arena Visual session artifact found under: ${RUN_DIR}/replays" >&2
  exit 1
fi

MANIFEST="${RUN_DIR}/replays/${SAMPLE_ID}/arena_visual_session/v1/manifest.json"
if [[ ! -f "${MANIFEST}" ]]; then
  echo "[arena-visual][error] Manifest not found: ${MANIFEST}" >&2
  exit 1
fi

VIEW_URL="http://${HOST}:${PORT}/sessions/${SAMPLE_ID}?run_id=${RUN_ID}"
API_URL="http://${HOST}:${PORT}/arena_visual/sessions/${SAMPLE_ID}?run_id=${RUN_ID}"

cat <<MSG
[arena-visual] Python: ${PYTHON_EXEC}
[arena-visual] Runs: ${RUNS_DIR}
[arena-visual] Run ID: ${RUN_ID}
[arena-visual] Sample ID: ${SAMPLE_ID}
[arena-visual] Manifest: ${MANIFEST}
[arena-visual] URL: ${VIEW_URL}
MSG

PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}" \
"${PYTHON_EXEC}" -m gage_eval.tools.arena_visual_server \
  --arena-visual-dir "${RUNS_DIR}" \
  --host "${HOST}" \
  --port "${PORT}" &
SERVER_PID=$!

cleanup() {
  if kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    kill "${SERVER_PID}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT INT TERM

deadline=$((SECONDS + WAIT_TIMEOUT_S))
while (( SECONDS < deadline )); do
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "[arena-visual][error] Arena Visual server exited before becoming ready." >&2
    wait "${SERVER_PID}" || true
    exit 1
  fi
  if command -v curl >/dev/null 2>&1; then
    if curl -sf "${API_URL}" >/dev/null 2>&1; then
      break
    fi
  else
    sleep 1
    break
  fi
  sleep 0.2
done

if (( AUTO_OPEN == 1 )); then
  if command -v open >/dev/null 2>&1; then
    open "${VIEW_URL}" >/dev/null 2>&1 || true
  elif command -v xdg-open >/dev/null 2>&1; then
    xdg-open "${VIEW_URL}" >/dev/null 2>&1 || true
  fi
fi

echo "[arena-visual] Press Ctrl+C to stop the server."
wait "${SERVER_PID}"
