#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

REPLAY_PATH="${REPLAY_PATH:-}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-5800}"
FPS="${FPS:-12}"
AUTO_OPEN="${AUTO_OPEN:-1}"
ALL_FRAMES="${ALL_FRAMES:-1}"
KEEP_TMP="${KEEP_TMP:-0}"
TASK_ID="${TASK_ID:-replay_from_path}"
SAMPLE_ID="${SAMPLE_ID:-sample_from_path}"
GAME="${GAME:-}"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/oneclick/run_ws_rgb_replay_from_path.sh <replay_path_or_dir> [options]

Arguments:
  replay_path_or_dir            Path to replay.json or replay directory.

Options:
  --host <host>                 Bind host (default: 127.0.0.1)
  --port <port>                 Bind port (default: 5800)
  --fps <fps>                   Replay fps (default: 12)
  --auto-open <0|1>             Auto open viewer in browser (default: 1)
  --all-frames <0|1>            Rebuild frame events from frames/* when possible (default: 1)
  --python-bin <path>           Python executable
  --task-id <task_id>           Synthetic task id in temporary sample
  --sample-id <sample_id>       Synthetic sample id in temporary sample
  --game <game>                 Optional fallback game key (e.g. pettingzoo)
  --keep-tmp <0|1>              Keep generated temporary files (default: 0)
  -h, --help                    Show this help

Examples:
  bash scripts/oneclick/run_ws_rgb_replay_from_path.sh runs/retro_mario_openai_rgb/replays/retro_mario_demo
  bash scripts/oneclick/run_ws_rgb_replay_from_path.sh runs/retro_mario_openai_rgb/replays/retro_mario_demo/replay.json --port 5810
EOF
}

if [[ $# -lt 1 ]]; then
  usage
  exit 1
fi

if [[ "$1" != -* ]]; then
  REPLAY_PATH="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
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
    --all-frames)
      ALL_FRAMES="${2:-}"
      shift 2
      ;;
    --python-bin)
      PYTHON_BIN="${2:-}"
      shift 2
      ;;
    --task-id)
      TASK_ID="${2:-}"
      shift 2
      ;;
    --sample-id)
      SAMPLE_ID="${2:-}"
      shift 2
      ;;
    --game)
      GAME="${2:-}"
      shift 2
      ;;
    --keep-tmp)
      KEEP_TMP="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[replay-from-path][error] Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${REPLAY_PATH}" ]]; then
  echo "[replay-from-path][error] replay_path_or_dir is required." >&2
  usage
  exit 1
fi

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
  echo "[replay-from-path][error] python not found: ${PYTHON_EXEC}" >&2
  exit 1
fi

resolve_replay_json() {
  local input_path="$1"
  if [[ -f "${input_path}" ]]; then
    echo "${input_path}"
    return 0
  fi
  if [[ -d "${input_path}" ]]; then
    if [[ -f "${input_path}/replay.json" ]]; then
      echo "${input_path}/replay.json"
      return 0
    fi
    local nested
    nested="$(find "${input_path}" -name 'replay.json' | head -n 1)"
    if [[ -n "${nested}" ]]; then
      echo "${nested}"
      return 0
    fi
  fi
  return 1
}

REPLAY_JSON="$(resolve_replay_json "${REPLAY_PATH}" || true)"
if [[ -z "${REPLAY_JSON}" || ! -f "${REPLAY_JSON}" ]]; then
  echo "[replay-from-path][error] replay.json not found from: ${REPLAY_PATH}" >&2
  exit 1
fi
REPLAY_JSON="$(cd "$(dirname "${REPLAY_JSON}")" && pwd)/$(basename "${REPLAY_JSON}")"

WORK_DIR="$(mktemp -d "/tmp/gage_replay_from_path_XXXXXX")"
SAMPLE_JSON="${WORK_DIR}/synthetic_sample_for_replay.json"

cleanup() {
  if [[ "${KEEP_TMP}" != "1" ]]; then
    rm -rf "${WORK_DIR}" >/dev/null 2>&1 || true
  else
    echo "[replay-from-path] keep temp dir: ${WORK_DIR}" >&2
  fi
}
trap cleanup EXIT

EFFECTIVE_REPLAY_JSON="$(
"${PYTHON_EXEC}" - <<'PY' "${REPLAY_JSON}" "${WORK_DIR}" "${ALL_FRAMES}"
from __future__ import annotations

import copy
import json
import re
import sys
import time
from pathlib import Path


def _stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _load_events(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    for line in path.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            payload = json.loads(text)
        except Exception:
            continue
        if isinstance(payload, dict):
            events.append(payload)
    return events


def _format_from_suffix(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".jpg", ".jpeg"}:
        return "jpeg"
    if ext == ".png":
        return "png"
    if ext == ".webp":
        return "webp"
    if ext == ".bmp":
        return "bmp"
    return "jpeg"


def _frame_step_from_name(path: Path, index: int) -> int:
    matched = re.search(r"(\d+)", path.stem)
    if matched is None:
        return index
    try:
        return int(matched.group(1))
    except Exception:
        return index


replay_json = Path(sys.argv[1]).resolve()
work_dir = Path(sys.argv[2]).resolve()
all_frames_enabled = str(sys.argv[3] or "1").strip().lower() in {"1", "true", "yes", "on"}

effective_replay_json = replay_json

if not all_frames_enabled:
    _stderr("[replay-from-path] all-frames mode disabled")
    print(str(effective_replay_json))
    raise SystemExit(0)

try:
    replay_payload = json.loads(replay_json.read_text(encoding="utf-8"))
except Exception as exc:
    _stderr(f"[replay-from-path] keep original replay: read replay.json failed: {exc}")
    print(str(effective_replay_json))
    raise SystemExit(0)

if not isinstance(replay_payload, dict):
    _stderr("[replay-from-path] keep original replay: replay.json root is not object")
    print(str(effective_replay_json))
    raise SystemExit(0)

if replay_payload.get("schema") != "gage_replay/v1":
    _stderr("[replay-from-path] keep original replay: schema is not gage_replay/v1")
    print(str(effective_replay_json))
    raise SystemExit(0)

recording = replay_payload.get("recording")
if not isinstance(recording, dict):
    _stderr("[replay-from-path] keep original replay: recording block missing")
    print(str(effective_replay_json))
    raise SystemExit(0)

events_rel = str(recording.get("events_path") or "events.jsonl")
events_path = (replay_json.parent / events_rel).resolve()
events = _load_events(events_path)
frame_events = [e for e in events if str(e.get("type") or "").strip().lower() == "frame"]
non_frame_events = [dict(e) for e in events if str(e.get("type") or "").strip().lower() != "frame"]

frames_dir = replay_json.parent / "frames"
image_files: list[Path] = []
if frames_dir.exists() and frames_dir.is_dir():
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp"):
        image_files.extend(frames_dir.glob(pattern))
image_files = sorted(set(path.resolve() for path in image_files), key=lambda path: path.name)

if len(image_files) == 0:
    _stderr("[replay-from-path] keep original replay: frames directory has no images")
    print(str(effective_replay_json))
    raise SystemExit(0)

if len(image_files) <= len(frame_events):
    _stderr(
        f"[replay-from-path] keep original replay: events already have enough frames "
        f"(events={len(frame_events)} images={len(image_files)})"
    )
    print(str(effective_replay_json))
    raise SystemExit(0)

seq_candidates: list[int] = []
for event in events:
    seq_value = event.get("seq")
    try:
        seq_candidates.append(int(seq_value))
    except Exception:
        continue
seq_base = max(seq_candidates) if seq_candidates else 0
timestamp_base_ms = int(time.time() * 1000)

rebuilt_frame_events: list[dict] = []
for index, image_file in enumerate(image_files):
    step = _frame_step_from_name(image_file, index)
    abs_image_path = str(image_file)
    event = {
        "type": "frame",
        "seq": int(seq_base + index + 1),
        "step": int(step),
        "timestamp_ms": int(timestamp_base_ms + index * 16),
        "actor": "replay_rebuild",
        "frame": {
            "board_text": "",
            "move_count": int(step),
            "metadata": {
                "rebuild_source": "frames_dir",
                "frame_file": image_file.name,
                "replay_index": int(index),
            },
        },
        "image": {
            "path": abs_image_path,
            "format": _format_from_suffix(image_file),
        },
    }
    rebuilt_frame_events.append(event)

rebuilt_events = non_frame_events + rebuilt_frame_events
events_out = work_dir / "events.rebuilt.jsonl"
events_out.write_text(
    "\n".join(json.dumps(item, ensure_ascii=False) for item in rebuilt_events) + "\n",
    encoding="utf-8",
)

replay_out_payload = copy.deepcopy(replay_payload)
recording_out = dict(replay_out_payload.get("recording") or {})
recording_out["events_path"] = events_out.name
counts_out = dict(recording_out.get("counts") or {})
counts_out["frame"] = len(rebuilt_frame_events)
recording_out["counts"] = counts_out
replay_out_payload["recording"] = recording_out

replay_out = work_dir / "replay.rebuilt.json"
replay_out.write_text(json.dumps(replay_out_payload, ensure_ascii=False, indent=2), encoding="utf-8")
_stderr(
    f"[replay-from-path] rebuilt frame events: events={len(frame_events)} -> {len(rebuilt_frame_events)}; "
    f"replay={replay_out}"
)
effective_replay_json = replay_out

print(str(effective_replay_json))
PY
)"

if [[ -z "${EFFECTIVE_REPLAY_JSON}" ]]; then
  EFFECTIVE_REPLAY_JSON="${REPLAY_JSON}"
fi

"${PYTHON_EXEC}" - <<'PY' "${EFFECTIVE_REPLAY_JSON}" "${SAMPLE_JSON}" "${TASK_ID}" "${SAMPLE_ID}"
from __future__ import annotations

import json
import sys
from pathlib import Path

replay_json = str(Path(sys.argv[1]).expanduser().resolve())
sample_json = Path(sys.argv[2]).expanduser().resolve()
task_id = str(sys.argv[3] or "replay_from_path").strip() or "replay_from_path"
sample_id = str(sys.argv[4] or "sample_from_path").strip() or "sample_from_path"

payload = {
    "task_id": task_id,
    "sample": {
        "id": sample_id,
        "task_id": task_id,
        "metadata": {
            "player_ids": ["player_0"],
            "source": "run_ws_rgb_replay_from_path.sh",
        },
        "predict_result": [
            {
                "replay_v1_path": replay_json,
            }
        ],
    },
}
sample_json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo "[replay-from-path] replay_json=${REPLAY_JSON}"
echo "[replay-from-path] effective_replay_json=${EFFECTIVE_REPLAY_JSON}"
echo "[replay-from-path] sample_json=${SAMPLE_JSON}"

CMD=(
  "${PYTHON_EXEC}" -m gage_eval.tools.ws_rgb_replay
  --sample-json "${SAMPLE_JSON}"
  --host "${HOST}"
  --port "${PORT}"
  --fps "${FPS}"
  --auto-open "${AUTO_OPEN}"
)

if [[ -n "${GAME}" ]]; then
  CMD+=(--game "${GAME}")
fi

PYTHONPATH="${ROOT}/src" "${CMD[@]}"
