# Game Arena PettingZoo Guide

English | [中文](game_arena_pettingzoo_zh.md)

This is the canonical Game Arena guide for PettingZoo Atari in this repository. It consolidates installation, demo startup, replay, and parameter-tuning notes into one place.

## 1. Overview

PettingZoo Atari currently covers three common paths:

- Dummy run for environment and viewer verification
- LLM vs LLM run for API-backed evaluation
- Human vs Human record run for browser-based interaction

Use this guide as the standard entry. Older PettingZoo docs are kept for reference only.

## 2. Canonical Files

| Type | Path | Purpose |
| --- | --- | --- |
| One-click script | `scripts/oneclick/run_pettingzoo_ws_rgb_viewer.sh` | Start a ws_rgb demo and auto-open the viewer when possible |
| Config directory | `config/custom/pettingzoo/` | All PettingZoo game configs live here |
| Recommended dummy config | `config/custom/pettingzoo/pong_dummy_ws_rgb.yaml` | Fastest ws_rgb smoke test |
| Recommended AI config | `config/custom/pettingzoo/space_invaders_ai.yaml` | Standard LLM demo |
| Human record config | `config/custom/pettingzoo/space_invaders_human_vs_human_record.yaml` | Browser-based human input |
| Supplemental command index | `docs/guide/pettingzoo_atari_run_commands.md` | Full per-game AI and Dummy command list |
| Replay tool | `src/gage_eval/tools/ws_rgb_replay.py` | Start post-run replay from sample artifacts |

## 3. Prerequisites

Use the same Python interpreter for installation and `run.py`.

```bash
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -m pip install -U \
  "pettingzoo[atari]>=1.24.3" \
  "shimmy[atari]>=1.0.0" \
  "AutoROM[accept-rom-license]>=0.6.1"
"$PYTHON_BIN" -m AutoROM.AutoROM --accept-license
```

Recommended ROM check:

```bash
"$PYTHON_BIN" - <<'PY'
from pettingzoo.atari import pong_v3

env = pong_v3.env(render_mode="rgb_array")
env.reset(seed=0)
print("PettingZoo Atari ROM check: OK")
env.close()
PY
```

## 4. Startup Paths

### 4.1 Recommended smoke test: one-click ws_rgb dummy run

```bash
CONFIG=config/custom/pettingzoo/pong_dummy_ws_rgb.yaml \
RUN_ID="pz_pong_dummy_ws_rgb_$(date +%Y%m%d_%H%M%S)" \
bash scripts/oneclick/run_pettingzoo_ws_rgb_viewer.sh
```

What the script does:

1. Picks a Python executable.
2. Validates the config path.
3. Picks a free `WS_RGB_PORT`.
4. Runs `python run.py --config ...`.
5. Waits for `/ws_rgb/viewer` and auto-opens it when desktop support is available.

Useful script environment variables:

- `CONFIG`: Config file to run
- `PYTHON_BIN`: Python executable
- `RUN_ID`: Run id under `runs/`
- `OUTPUT_DIR`: Output directory
- `WS_RGB_PORT`: Preferred viewer port, auto-incremented if occupied
- `AUTO_OPEN`: Set `0` to disable browser auto-open
- `WAIT_TIMEOUT_S`: Viewer readiness timeout

### 4.2 LLM run

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
export GAME="space_invaders"
export RUN_ID="pz_${GAME}_ai_$(date +%Y%m%d_%H%M%S)"

python run.py \
  --config "config/custom/pettingzoo/${GAME}_ai.yaml" \
  --output-dir runs \
  --run-id "$RUN_ID"
```

If you plan to replay from sample artifacts later, set these before the run:

```bash
export GAGE_EVAL_GAME_LOG_INLINE_LIMIT=-1
export GAGE_EVAL_GAME_LOG_INLINE_BYTES=0
```

### 4.3 Human vs Human record run

```bash
PYTHONPATH=src python run.py \
  --config config/custom/pettingzoo/space_invaders_human_vs_human_record.yaml \
  --output-dir runs \
  --run-id pz_space_invaders_h2h_record
```

Default viewer URL:

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

Default key map in `space_invaders_human_vs_human_record.yaml`:

- `player_0`: `Q/W/E/A/S/D`
- `player_1`: `U/I/O/J/K/L`

## 5. Execution Order

The repository currently follows this order for PettingZoo demos:

1. Install PettingZoo Atari dependencies and ROMs.
2. Choose one config under `config/custom/pettingzoo/`.
3. Set API keys only when the selected config uses an LLM backend.
4. Start the run either with `run_pettingzoo_ws_rgb_viewer.sh` or with `python run.py`.
5. Open the viewer during the run for `display_mode: websocket` configs, or start `ws_rgb_replay` after the run.

## 6. Key Parameters and Where to Change Them

| Item | Where to change | Meaning |
| --- | --- | --- |
| API key | Shell env: `OPENAI_API_KEY` | Required by `*_ai.yaml` configs that use OpenAI backends |
| Game selection | Config filename and `environment.env_id` | Switch the Atari game and runtime env id |
| Runtime length | `environment.env_kwargs.max_cycles` | Environment frame limit |
| Arena turn cap | `scheduler.max_turns` | Arena-side maximum turn count |
| Viewer mode | `environment.display_mode` | Use `websocket` for ws_rgb live view |
| Viewer host/port | `human_input.ws_host` / `human_input.ws_port` | Bind address for the live viewer |
| Replay playback FPS | `python -m gage_eval.tools.ws_rgb_replay --fps` | Replay speed in the post-run viewer |
| Action labels | `environment.use_action_meanings` | `true` shows names such as `FIRE`, `LEFT`; `false` uses numeric ids |
| Human key mapping | `environment.action_schema.key_map` | Browser key-to-action mapping for record-mode human control |
| Human input endpoint | `human_input.host` / `human_input.port` | Queue input server for record-mode human control |

Notes:

- Most game configs use `max_cycles: 300`; `space_invaders_*` uses `3000`.
- The one-click script controls only the preferred viewer port. The actual game behavior still comes from the selected YAML.

## 7. Outputs and Replay

Run artifacts are written under:

```text
runs/<run_id>/
```

For post-run replay:

```bash
export RUN_ID="<your_run_id>"
export GAGE_EVAL_GAME_LOG_INLINE_LIMIT=-1
export GAGE_EVAL_GAME_LOG_INLINE_BYTES=0
SAMPLE_JSON="$(find "runs/${RUN_ID}/samples" -type f -name '*.json' | head -n 1)"

PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo
```

Default replay URL:

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

## 8. Other Docs

The following files are still kept in the repository:

- `docs/guide/pettingzoo_user_guide.md`
- `docs/guide/pettingzoo_user_guide_zh.md`
- `docs/guide/pettingzoo_atari_run_commands.md`
- `docs/guide/pettingzoo_atari_run_commands_zh.md`
