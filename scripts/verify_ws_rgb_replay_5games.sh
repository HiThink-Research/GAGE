#!/usr/bin/env bash
set -euo pipefail

# Verify ws_rgb replay integration across 5 games:
# gomoku / tictactoe / doudizhu / mahjong / pettingzoo.
#
# This script is offline-first:
# - Uses dummy backends only.
# - Uses local datasets under tests/data.
# - Validates replay artifacts and replay server endpoints.
#
# Usage:
#   bash scripts/verify_ws_rgb_replay_5games.sh
#
# Optional env:
#   PYTHON_BIN   Python interpreter path (default tries current venv first).
#   VERIFY_PORT  replay_server port for endpoint checks (default: 18080).
#   RUN_PREFIX   run id prefix (default: verify5g).
#   OUTPUT_DIR   run output dir (default: runs).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -n "${PYTHON_BIN:-}" ]]; then
  PY="${PYTHON_BIN}"
elif [[ -x "${REPO_ROOT}/.venv/bin/python" ]]; then
  PY="${REPO_ROOT}/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PY="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PY="$(command -v python)"
else
  echo "[verify5g] python interpreter not found" >&2
  exit 1
fi

VERIFY_PORT="${VERIFY_PORT:-18080}"
RUN_PREFIX="${RUN_PREFIX:-verify5g}"
OUTPUT_DIR="${OUTPUT_DIR:-runs}"
STAMP="$(date +%Y%m%d_%H%M%S)"
WORK_DIR="/tmp/gage_ws_rgb_verify_${STAMP}"
mkdir -p "${WORK_DIR}"

log() {
  printf '[verify5g] %s\n' "$*"
}

write_configs() {
  # gomoku: align player ids with Test_Gomoku_LiteLLM metadata (Black/White).
  cat > "${WORK_DIR}/gomoku.yaml" <<'YAML'
api_version: gage/v1alpha1
kind: PipelineConfig
metadata:
  name: gomoku_replay_verify
custom:
  steps:
    - step: arena
      adapter_id: gomoku_arena
datasets:
  - dataset_id: gomoku_local
    loader: jsonl
    params:
      path: tests/data/Test_Gomoku_LiteLLM.jsonl
      preprocess: grid_game_preprocessor
      streaming: false
backends:
  - backend_id: gomoku_dummy_black
    type: dummy
    config:
      responses: ["A1", "B1", "C1", "D1"]
      cycle: true
  - backend_id: gomoku_dummy_white
    type: dummy
    config:
      responses: ["A2", "B2", "C2", "D2"]
      cycle: true
role_adapters:
  - adapter_id: gomoku_dummy_black
    role_type: dut_model
    backend_id: gomoku_dummy_black
    capabilities: [chat_completion]
  - adapter_id: gomoku_dummy_white
    role_type: dut_model
    backend_id: gomoku_dummy_white
    capabilities: [chat_completion]
  - adapter_id: gomoku_arena
    role_type: arena
    params:
      environment:
        impl: gomoku_local_v1
        board_size: 9
        coord_scheme: A1
        replay:
          enabled: true
          primary_mode: true
          frame_capture:
            enabled: true
            frame_stride: 1
            max_frames: 120
      rules:
        win_len: 5
        illegal_policy:
          retry: 0
          on_fail: loss
      scheduler:
        type: turn
        max_turns: 60
      parser:
        impl: grid_parser_v1
        coord_scheme: A1
      players:
        - name: Black
          player_id: Black
          type: backend
          ref: gomoku_dummy_black
          max_retries: 0
          fallback_policy: first_legal
        - name: White
          player_id: White
          type: backend
          ref: gomoku_dummy_white
          max_retries: 0
          fallback_policy: first_legal
tasks:
  - task_id: gomoku_replay_verify
    dataset_id: gomoku_local
    max_samples: 1
    concurrency: 1
YAML

  # tictactoe: align player ids with Test_TicTacToe metadata (X/O).
  cat > "${WORK_DIR}/tictactoe.yaml" <<'YAML'
api_version: gage/v1alpha1
kind: PipelineConfig
metadata:
  name: tictactoe_replay_verify
custom:
  steps:
    - step: arena
      adapter_id: tictactoe_arena
datasets:
  - dataset_id: tictactoe_local
    loader: jsonl
    params:
      path: tests/data/Test_TicTacToe.jsonl
      preprocess: grid_game_preprocessor
      streaming: false
backends:
  - backend_id: ttt_dummy_x
    type: dummy
    config:
      responses: ["0", "1", "2"]
      cycle: true
  - backend_id: ttt_dummy_o
    type: dummy
    config:
      responses: ["2", "1", "0"]
      cycle: true
role_adapters:
  - adapter_id: ttt_dummy_x
    role_type: dut_model
    backend_id: ttt_dummy_x
    capabilities: [chat_completion]
  - adapter_id: ttt_dummy_o
    role_type: dut_model
    backend_id: ttt_dummy_o
    capabilities: [chat_completion]
  - adapter_id: tictactoe_arena
    role_type: arena
    params:
      environment:
        impl: tictactoe_v1
        board_size: 3
        coord_scheme: ROW_COL
        replay:
          enabled: true
          primary_mode: true
          frame_capture:
            enabled: true
            frame_stride: 1
            max_frames: 40
      rules:
        win_len: 3
        illegal_policy:
          retry: 0
          on_fail: loss
      scheduler:
        type: turn
        max_turns: 16
      parser:
        impl: grid_parser_v1
        board_size: 3
        coord_scheme: ROW_COL
      players:
        - name: X
          player_id: X
          type: backend
          ref: ttt_dummy_x
          max_retries: 0
          fallback_policy: first_legal
        - name: O
          player_id: O
          type: backend
          ref: ttt_dummy_o
          max_retries: 0
          fallback_policy: first_legal
tasks:
  - task_id: tictactoe_replay_verify
    dataset_id: tictactoe_local
    max_samples: 1
    concurrency: 1
YAML

  cat > "${WORK_DIR}/doudizhu.yaml" <<'YAML'
api_version: gage/v1alpha1
kind: PipelineConfig
metadata:
  name: doudizhu_replay_verify
custom:
  steps:
    - step: arena
      adapter_id: doudizhu_arena
datasets:
  - dataset_id: doudizhu_local
    loader: jsonl
    params:
      path: tests/data/Test_Doudizhu_LiteLLM.jsonl
      preprocess: card_game_preprocessor
      streaming: false
backends:
  - backend_id: ddz_dummy_0
    type: dummy
    config:
      responses: ["pass", "333", "444"]
      cycle: true
  - backend_id: ddz_dummy_1
    type: dummy
    config:
      responses: ["pass", "555", "666"]
      cycle: true
  - backend_id: ddz_dummy_2
    type: dummy
    config:
      responses: ["pass", "777", "888"]
      cycle: true
role_adapters:
  - adapter_id: ddz_dummy_0
    role_type: dut_model
    backend_id: ddz_dummy_0
    capabilities: [chat_completion]
  - adapter_id: ddz_dummy_1
    role_type: dut_model
    backend_id: ddz_dummy_1
    capabilities: [chat_completion]
  - adapter_id: ddz_dummy_2
    role_type: dut_model
    backend_id: ddz_dummy_2
    capabilities: [chat_completion]
  - adapter_id: doudizhu_arena
    role_type: arena
    params:
      environment:
        impl: doudizhu_arena_v1
        chat_mode: off
        replay_live: false
        replay:
          enabled: true
          primary_mode: true
          frame_capture:
            enabled: true
            frame_stride: 1
            max_frames: 150
      rules:
        illegal_policy:
          retry: 0
          on_fail: random
      scheduler:
        type: turn
        max_turns: 100
      parser:
        impl: doudizhu_arena_parser_v1
      players:
        - name: p0
          player_id: player_0
          type: backend
          ref: ddz_dummy_0
          max_retries: 0
          fallback_policy: first_legal
        - name: p1
          player_id: player_1
          type: backend
          ref: ddz_dummy_1
          max_retries: 0
          fallback_policy: first_legal
        - name: p2
          player_id: player_2
          type: backend
          ref: ddz_dummy_2
          max_retries: 0
          fallback_policy: first_legal
tasks:
  - task_id: doudizhu_replay_verify
    dataset_id: doudizhu_local
    max_samples: 1
    concurrency: 1
YAML

  cat > "${WORK_DIR}/mahjong.yaml" <<'YAML'
api_version: gage/v1alpha1
kind: PipelineConfig
metadata:
  name: mahjong_replay_verify
custom:
  steps:
    - step: arena
      adapter_id: mahjong_arena
datasets:
  - dataset_id: mahjong_local
    loader: jsonl
    params:
      path: tests/data/Test_Mahjong_Dummy.jsonl
      preprocess: card_game_preprocessor
      streaming: false
backends:
  - backend_id: mj_dummy_0
    type: dummy
    config:
      responses: ["pass", "stand", "chow"]
      cycle: true
  - backend_id: mj_dummy_1
    type: dummy
    config:
      responses: ["pass", "stand", "pong"]
      cycle: true
  - backend_id: mj_dummy_2
    type: dummy
    config:
      responses: ["pass", "stand", "kong"]
      cycle: true
  - backend_id: mj_dummy_3
    type: dummy
    config:
      responses: ["pass", "stand", "hu"]
      cycle: true
role_adapters:
  - adapter_id: mj_dummy_0
    role_type: dut_model
    backend_id: mj_dummy_0
    capabilities: [chat_completion]
  - adapter_id: mj_dummy_1
    role_type: dut_model
    backend_id: mj_dummy_1
    capabilities: [chat_completion]
  - adapter_id: mj_dummy_2
    role_type: dut_model
    backend_id: mj_dummy_2
    capabilities: [chat_completion]
  - adapter_id: mj_dummy_3
    role_type: dut_model
    backend_id: mj_dummy_3
    capabilities: [chat_completion]
  - adapter_id: mahjong_arena
    role_type: arena
    params:
      environment:
        impl: mahjong_rlcard_v1
        game_type: mahjong
        chat_mode: off
        replay_live: false
        replay:
          enabled: true
          primary_mode: true
          frame_capture:
            enabled: true
            frame_stride: 1
            max_frames: 180
      rules:
        illegal_policy:
          retry: 0
          on_fail: random
      scheduler:
        type: turn
        max_turns: 120
      parser:
        impl: mahjong_v1
      players:
        - name: p0
          player_id: player_0
          type: backend
          ref: mj_dummy_0
          max_retries: 0
          fallback_policy: first_legal
        - name: p1
          player_id: player_1
          type: backend
          ref: mj_dummy_1
          max_retries: 0
          fallback_policy: first_legal
        - name: p2
          player_id: player_2
          type: backend
          ref: mj_dummy_2
          max_retries: 0
          fallback_policy: first_legal
        - name: p3
          player_id: player_3
          type: backend
          ref: mj_dummy_3
          max_retries: 0
          fallback_policy: first_legal
tasks:
  - task_id: mahjong_replay_verify
    dataset_id: mahjong_local
    max_samples: 1
    concurrency: 1
YAML

  cat > "${WORK_DIR}/pettingzoo.yaml" <<'YAML'
api_version: gage/v1alpha1
kind: PipelineConfig
metadata:
  name: pettingzoo_replay_verify
custom:
  steps:
    - step: arena
      adapter_id: pettingzoo_arena
datasets:
  - dataset_id: pz_local
    loader: jsonl
    params:
      path: tests/data/pettingzoo_atari_demo.jsonl
      streaming: false
backends:
  - backend_id: pz_dummy_0
    type: dummy
    config:
      responses: ["0", "1", "2"]
      cycle: true
  - backend_id: pz_dummy_1
    type: dummy
    config:
      responses: ["2", "1", "0"]
      cycle: true
role_adapters:
  - adapter_id: pz_dummy_0
    role_type: dut_model
    backend_id: pz_dummy_0
    capabilities: [chat_completion]
  - adapter_id: pz_dummy_1
    role_type: dut_model
    backend_id: pz_dummy_1
    capabilities: [chat_completion]
  - adapter_id: pettingzoo_arena
    role_type: arena
    params:
      environment:
        impl: pettingzoo_aec_v1
        env_id: pettingzoo.classic.rps_v2
        include_raw_obs: false
        use_action_meanings: false
        replay:
          enabled: true
          primary_mode: true
          frame_capture:
            enabled: true
            frame_stride: 1
            max_frames: 80
      rules:
        illegal_policy:
          retry: 0
          on_fail: loss
      scheduler:
        type: turn
        max_turns: 30
      parser:
        impl: discrete_action_parser_v1
      players:
        - name: p0
          player_id: player_0
          type: backend
          ref: pz_dummy_0
          max_retries: 0
          fallback_policy: first_legal
        - name: p1
          player_id: player_1
          type: backend
          ref: pz_dummy_1
          max_retries: 0
          fallback_policy: first_legal
tasks:
  - task_id: pettingzoo_replay_verify
    dataset_id: pz_local
    max_samples: 1
    concurrency: 1
YAML
}

run_games() {
  declare -gA RUN_IDS=()
  for game in gomoku mahjong doudizhu pettingzoo tictactoe; do
    local cfg="${WORK_DIR}/${game}.yaml"
    local run_id="${RUN_PREFIX}_${game}_${STAMP}"
    local log_file="${WORK_DIR}/${run_id}.log"
    log "running ${game} (run_id=${run_id})"
    if "${PY}" run.py --config "${cfg}" --output-dir "${OUTPUT_DIR}" --run-id "${run_id}" >"${log_file}" 2>&1; then
      RUN_IDS["${game}"]="${run_id}"
    else
      log "run failed for ${game}; see ${log_file}"
      tail -n 80 "${log_file}" || true
      exit 1
    fi
  done
}

validate_replays() {
  local summary_json="${WORK_DIR}/summary.json"
  PYTHONPATH=src "${PY}" - "${VERIFY_PORT}" "${OUTPUT_DIR}" "${summary_json}" \
    "${RUN_IDS[gomoku]}" \
    "${RUN_IDS[mahjong]}" \
    "${RUN_IDS[doudizhu]}" \
    "${RUN_IDS[pettingzoo]}" \
    "${RUN_IDS[tictactoe]}" <<'PY'
import json
import os
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

port = int(sys.argv[1])
output_dir = Path(sys.argv[2]).resolve()
summary_path = Path(sys.argv[3]).resolve()
runs = {
    "gomoku": sys.argv[4],
    "mahjong": sys.argv[5],
    "doudizhu": sys.argv[6],
    "pettingzoo": sys.argv[7],
    "tictactoe": sys.argv[8],
}

summary: dict[str, dict] = {}
for game, run_id in runs.items():
    replay_files = sorted((output_dir / run_id / "replays").glob("*/replay.json"))
    if not replay_files:
        summary[game] = {
            "ok": False,
            "run_id": run_id,
            "error": "replay_json_missing",
        }
        continue
    replay_file = replay_files[0].resolve()
    sample_id = replay_file.parent.name
    replay_payload = json.loads(replay_file.read_text(encoding="utf-8"))
    events_rel = ((replay_payload.get("recording") or {}).get("events_path") or "events.jsonl")
    events_path = (replay_file.parent / events_rel).resolve()
    counts = {"action": 0, "frame": 0, "result": 0}
    if events_path.exists():
        for line in events_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except Exception:
                continue
            event_type = str(event.get("type") or "").lower()
            if event_type in counts:
                counts[event_type] += 1
    summary[game] = {
        "ok": True,
        "run_id": run_id,
        "sample_id": sample_id,
        "replay_path": str(replay_file),
        "schema": replay_payload.get("schema"),
        "counts": counts,
    }

env = dict(os.environ)
env["PYTHONPATH"] = "src"
proc = subprocess.Popen(
    [
        sys.executable,
        "-m",
        "gage_eval.tools.replay_server",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--replay-dir",
        str(output_dir),
    ],
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    env=env,
)

try:
    for _ in range(60):
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{port}/tournament/replay?run_id=__none__&sample_id=__none__",
                timeout=0.4,
            )
            break
        except Exception:
            time.sleep(0.2)

    for game, row in summary.items():
        if not row.get("ok"):
            continue
        run_id = row["run_id"]
        sample_id = row["sample_id"]
        q_events = urllib.parse.urlencode({"run_id": run_id, "sample_id": sample_id})
        q_frame = urllib.parse.urlencode(
            {"run_id": run_id, "sample_id": sample_id, "index": 0, "format": "json"}
        )

        events_status = -1
        frame_status = -1
        has_frame_event = False
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/tournament/replay/events?{q_events}",
                timeout=2.0,
            ) as response:
                events_status = int(response.status)
                payload = json.loads(response.read().decode("utf-8"))
                events = payload.get("events") if isinstance(payload, dict) else []
                if isinstance(events, list):
                    has_frame_event = any(
                        isinstance(item, dict)
                        and str(item.get("type") or "").lower() == "frame"
                        for item in events
                    )
        except Exception:
            pass

        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/tournament/replay/frame?{q_frame}",
                timeout=2.0,
            ) as response:
                frame_status = int(response.status)
        except Exception:
            pass

        row["replay_events_http"] = events_status
        row["replay_events_has_frame"] = has_frame_event
        row["replay_frame_http"] = frame_status
finally:
    proc.terminate()
    try:
        proc.wait(timeout=2)
    except Exception:
        proc.kill()

summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
print(summary_path)
print(json.dumps(summary, ensure_ascii=False, indent=2))

failed = []
for game, row in summary.items():
    if not row.get("ok"):
        failed.append(f"{game}:setup")
        continue
    if row.get("schema") != "gage_replay/v1":
        failed.append(f"{game}:schema")
    counts = row.get("counts") or {}
    if int(counts.get("frame", 0)) <= 0:
        failed.append(f"{game}:frame_count")
    if int(row.get("replay_events_http", -1)) != 200:
        failed.append(f"{game}:events_endpoint")
    if not bool(row.get("replay_events_has_frame", False)):
        failed.append(f"{game}:events_no_frame")
    if int(row.get("replay_frame_http", -1)) != 200:
        failed.append(f"{game}:frame_endpoint")

if failed:
    print("FAILED: " + ",".join(failed))
    raise SystemExit(1)
PY
}

write_configs
run_games
validate_replays

log "all 5 games replay verification passed"
log "work dir: ${WORK_DIR}"
