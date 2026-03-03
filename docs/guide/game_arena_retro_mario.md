# Game Arena Retro Mario Guide

English | [中文](game_arena_retro_mario_zh.md)

This is the canonical Game Arena guide for the stable-retro Mario demos in this repository. It unifies setup, run commands, replay, and parameter locations.

## 1. Overview

Retro Mario in this repository is currently configuration-driven. There is no dedicated one-click Mario launcher yet, so the standard path is:

1. Import the ROM into stable-retro.
2. Choose a YAML config under `config/custom/`.
3. Run `python run.py --config ...`.
4. Open ws_rgb live view or replay when the selected config enables it.

## 2. Canonical Files

| Type | Path | Purpose |
| --- | --- | --- |
| Dummy ws_rgb config | `config/custom/retro_mario_phase1_dummy_ws.yaml` | Fastest live-view smoke test |
| Human ws_rgb config | `config/custom/retro_mario_phase1_human_ws.yaml` | Human-controlled Mario session |
| OpenAI ws_rgb config | `config/custom/retro_mario_openai_ws_rgb_auto_eval.yaml` | API-backed live-view demo |
| OpenAI headless config | `config/custom/retro_mario_openai_headless_auto_eval.yaml` | API-backed headless demo |
| Dummy headless config | `config/custom/retro_mario_phase1_dummy_headless_auto_eval.yaml` | Offline smoke test without viewer |
| Dataset | `config/custom/retro_mario_phase1.jsonl` | Default sample input |
| Environment implementation | `src/gage_eval/role/arena/games/retro/retro_env.py` | Runtime behavior and replay writing |
| Parser | `src/gage_eval/role/arena/parsers/retro_action_parser.py` | Parse `{"move": "...", "hold_ticks": ...}` actions |

## 3. Prerequisites

Install `stable-retro` and import the ROM before running any Mario config.

```bash
python -m pip install stable-retro
python -m retro.import "<rom_save_path>"
```

Current game id used by the configs:

```text
SuperMarioBros3-Nes-v0
```

For OpenAI configs, set the key before running:

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
```

Optional model override:

```bash
export RETRO_OPENAI_MODEL="gpt-4o-mini"
```

## 4. Startup Paths

### 4.1 Recommended smoke test: dummy + ws_rgb

```bash
export RETRO_WS_RGB_PORT=5800

env PYTHONPATH=src python run.py \
  --config config/custom/retro_mario_phase1_dummy_ws.yaml \
  --output-dir runs \
  --run-id retro_mario_dummy_ws
```

Default live-view URL:

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

### 4.2 OpenAI + ws_rgb

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
export RETRO_WS_RGB_PORT=5800

env PYTHONPATH=src python run.py \
  --config config/custom/retro_mario_openai_ws_rgb_auto_eval.yaml \
  --output-dir runs \
  --run-id retro_mario_openai_ws
```

### 4.3 Human + ws_rgb

```bash
env PYTHONPATH=src python run.py \
  --config config/custom/retro_mario_phase1_human_ws.yaml \
  --output-dir runs \
  --run-id retro_mario_human_ws
```

Current human input defaults:

- Viewer URL: `http://127.0.0.1:5800/ws_rgb/viewer`
- Queue input endpoint: `human_input.host: 0.0.0.0`, `human_input.port: 8001`
- Input FPS: `human_input.fps: 30`

Default browser key aliases come from the retro input mapper:

- Movement: `W/A/S/D` or arrow keys
- Jump: `J`, `Space`, `Z`, or `C`
- Run: `K` or `X`
- Select: `L` or `Shift`
- Start: `Enter`

### 4.4 Headless runs

```bash
env PYTHONPATH=src python run.py \
  --config config/custom/retro_mario_phase1_dummy_headless_auto_eval.yaml \
  --output-dir runs \
  --run-id retro_mario_dummy_headless
```

Use the headless configs when you want an offline verification path without ws_rgb.

## 5. Execution Order

Retro Mario startup in the current repository is:

1. Install `stable-retro`.
2. Import the ROM with `python -m retro.import`.
3. Choose one YAML config.
4. Set API keys only for the OpenAI configs.
5. Run `python run.py --config ...`.
6. Open `ws_rgb/viewer` during the run for websocket configs, or start replay after the run.

## 6. Key Parameters and Where to Change Them

| Item | Where to change | Meaning |
| --- | --- | --- |
| API key | Shell env: `OPENAI_API_KEY` | Required by `retro_mario_openai_*` configs |
| Model name | `RETRO_OPENAI_MODEL` or `backends[].config.model` | Select the OpenAI model |
| Live-view port | `human_input.ws_port` or env `RETRO_WS_RGB_PORT` | ws_rgb bind port for websocket configs |
| Display mode | `environment.display_mode` | `websocket` for live view, `headless` for offline runs |
| Legal moves | `environment.legal_moves` | Action space exposed to the player |
| Hold duration | `environment.action_schema.hold_ticks_*` and `parser.hold_ticks_*` | Clamp and default hold duration for macro moves |
| Tick pacing | `scheduler.tick_ms` | Main scheduler interval in milliseconds |
| Human input FPS | `human_input.fps` | Browser input sampling cadence in human mode |
| Replay mode | `environment.replay.mode` | `action`, `frame`, or `both` replay content |
| Replay frame capture | `environment.replay.frame_capture.*` | Frame stride, image format, and frame cap |

Notes:

- Mario actions are JSON objects such as `{"move":"right_run_jump","hold_ticks":6}`.
- The parser defaults `hold_ticks` when the value is omitted and clamps it into the configured range.
- `display_mode: websocket` is treated as headless runtime plus ws_rgb streaming by the arena adapter.

## 7. Outputs and Replay

Run artifacts are written under:

```text
runs/<run_id>/
```

Replay can be started from one sample artifact:

```bash
RUN_ID="<your_run_id>"
SAMPLE_JSON="$(find "runs/${RUN_ID}/samples" -type f -name '*.json' | head -n 1)"
REPLAY_PORT="${RETRO_REPLAY_PORT:-5800}"

env PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 0.0.0.0 \
  --port "${REPLAY_PORT}" \
  --fps 12
```

Replay URL:

```text
http://127.0.0.1:<REPLAY_PORT>/ws_rgb/viewer
```

Before launching replay, confirm the sample JSON has a `replay_path` field.

## 8. Legacy Docs

Older Mario docs are still in the repository but are no longer the standard entry:

- `docs/guide/stable_retro_mario_demo_guide.md`
- `docs/guide/stable_retro_mario_demo_guide_zh.md`
