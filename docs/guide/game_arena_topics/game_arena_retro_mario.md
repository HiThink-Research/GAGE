# Game Arena Retro Mario Guide

English | [中文](game_arena_retro_mario_zh.md)

This is the canonical Game Arena guide for the stable-retro Mario demos in this repository. It unifies setup, script-based startup, replay, and parameter locations.

## 1. Overview

Retro Mario now has a shared script entry for the common demo paths:

- Dummy + websocketRGB smoke test
- OpenAI + websocketRGB live view
- Human + websocketRGB control
- Dummy or OpenAI headless runs

The actual runtime behavior still comes from the YAML config, but startup and replay are now standardized through scripts under `scripts/run/arenas/retro_mario/`.

## 2. Canonical Files

| Type | Path | Purpose |
| --- | --- | --- |
| Standard startup script | `scripts/run/arenas/retro_mario/run.sh` | Main entry for common Mario startup modes |
| Replay script | `scripts/run/arenas/retro_mario/replay.sh` | Replay one finished run by `run_id` |
| Dummy websocketRGB config | `config/custom/retro_mario_phase1_dummy_ws.yaml` | Fastest live-view smoke test |
| Human websocketRGB config | `config/custom/retro_mario_phase1_human_ws.yaml` | Human-controlled Mario session |
| OpenAI websocketRGB config | `config/custom/retro_mario_openai_ws_rgb_auto_eval.yaml` | API-backed live-view demo |
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

### 4.1 Recommended smoke test: dummy + websocketRGB

```bash
bash scripts/run/arenas/retro_mario/run.sh \
  --mode dummy_ws \
  --run-id "retro_mario_dummy_ws_$(date +%Y%m%d_%H%M%S)"
```

Default live-view URL:

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

What the startup script does:

1. Picks a Python executable.
2. Resolves the YAML config from `--mode`, unless `--config` is provided.
3. Validates the config path.
4. Checks API-key requirements for OpenAI modes.
5. Runs `python run.py --config ...`.
6. Prints the viewer URL or input endpoint hints for websocket modes.

Supported startup modes in `scripts/run/arenas/retro_mario/run.sh`:

- `dummy_ws`
- `openai_ws`
- `human_ws`
- `dummy_headless`
- `openai_headless`

Useful startup options:

- `--mode`: Startup mode from the list above
- `--config`: Explicit YAML path; overrides the built-in mode mapping
- `--run-id`: Output run id under `runs/`
- `--output-dir`: Output directory, default `runs`
- `--python-bin`: Explicit Python interpreter path

Useful environment variables:

- `RETRO_WS_RGB_PORT`: Preferred websocketRGB port for websocket modes
- `OPENAI_API_KEY`: Required for `openai_ws` and `openai_headless`
- `LITELLM_API_KEY`: Accepted fallback for OpenAI modes
- `RETRO_OPENAI_MODEL`: Optional model override if your config reads it

### 4.2 OpenAI + websocketRGB

```bash
export OPENAI_API_KEY="<YOUR_KEY>"

bash scripts/run/arenas/retro_mario/run.sh \
  --mode openai_ws \
  --run-id "retro_mario_openai_ws_$(date +%Y%m%d_%H%M%S)"
```

Where to change the model/API for this command:

- API key: export `OPENAI_API_KEY` before launch. `scripts/run/arenas/retro_mario/run.sh` also accepts `LITELLM_API_KEY` and copies it into `OPENAI_API_KEY`.
- `openai_ws` reads `config/custom/retro_mario_openai_ws_rgb_auto_eval.yaml`. `openai_headless` reads `config/custom/retro_mario_openai_headless_auto_eval.yaml`.
- Hosted OpenAI-compatible API: edit `backends[0].config.base_url` to switch the endpoint. Edit `backends[0].config.model`, or set `RETRO_OPENAI_MODEL`, to switch the deployed model name.
- Local OpenAI-compatible service: point `backends[0].config.base_url` to the local server and set `backends[0].config.model` to the served model id. These configs currently set `require_api_key: true`, so keep `OPENAI_API_KEY` non-empty or change that flag if your local service does not require auth.

### 4.3 Human + websocketRGB

```bash
bash scripts/run/arenas/retro_mario/run.sh \
  --mode human_ws \
  --run-id "retro_mario_human_ws_$(date +%Y%m%d_%H%M%S)"
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
bash scripts/run/arenas/retro_mario/run.sh \
  --mode dummy_headless \
  --run-id "retro_mario_dummy_headless_$(date +%Y%m%d_%H%M%S)"
```

Use the headless configs when you want an offline verification path without websocketRGB.

For OpenAI headless:

```bash
export OPENAI_API_KEY="<YOUR_KEY>"

bash scripts/run/arenas/retro_mario/run.sh \
  --mode openai_headless \
  --run-id "retro_mario_openai_headless_$(date +%Y%m%d_%H%M%S)"
```

### 4.5 Explicit config override

Use `--config` if you want the script wrapper but need a custom YAML path:

```bash
bash scripts/run/arenas/retro_mario/run.sh \
  --config config/custom/retro_mario_phase1_dummy_ws.yaml \
  --run-id "retro_mario_custom_$(date +%Y%m%d_%H%M%S)"
```

## 5. Execution Order

Retro Mario startup in the current repository is:

1. Install `stable-retro`.
2. Import the ROM with `python -m retro.import`.
3. Choose a standard `--mode` or a custom `--config`.
4. Set API keys only for the OpenAI modes.
5. Run `scripts/run/arenas/retro_mario/run.sh`.
6. Open `ws_rgb/viewer` during the run for websocket modes, or start `scripts/run/arenas/retro_mario/replay.sh <run_id>` after the run.

## 6. Key Parameters and Where to Change Them

| Item | Where to change | Meaning |
| --- | --- | --- |
| API key | Shell env: `OPENAI_API_KEY` or `LITELLM_API_KEY` | Required by OpenAI startup modes; the script accepts either |
| Startup mode | `scripts/run/arenas/retro_mario/run.sh --mode` | Select `dummy_ws`, `openai_ws`, `human_ws`, `dummy_headless`, or `openai_headless` |
| Custom config | `scripts/run/arenas/retro_mario/run.sh --config` | Bypass the built-in mode mapping |
| Model endpoint | `backends[].config.base_url` in the selected `retro_mario_openai_*.yaml` | Switch between hosted and local OpenAI-compatible endpoints |
| Model name | `RETRO_OPENAI_MODEL` or `backends[].config.model` | Select the OpenAI model |
| API-key enforcement | `backends[].config.require_api_key` | Keep it `true` for hosted APIs; disable only for trusted local gateways |
| Live-view port | `human_input.ws_port` or env `RETRO_WS_RGB_PORT` | websocketRGB bind port for websocket configs; the script passes the env through |
| Display mode | `environment.display_mode` | `websocket` for live view, `headless` for offline runs |
| Legal moves | `environment.legal_moves` | Action space exposed to the player |
| Hold duration | `environment.action_schema.hold_ticks_*` and `parser.hold_ticks_*` | Clamp and default hold duration for macro moves |
| Tick pacing | `scheduler.tick_ms` | Main scheduler interval in milliseconds |
| Human input FPS | `human_input.fps` | Browser input sampling cadence in human mode |
| Replay mode | `environment.replay.mode` | `action`, `frame`, or `both` replay content |
| Replay frame capture | `environment.replay.frame_capture.*` | Frame stride, image format, and frame cap |
| Replay playback FPS | Env `FPS` for `scripts/run/arenas/retro_mario/replay.sh` | Replay speed in the post-run viewer |
| Replay host/port | Env `HOST` and `PORT` for `scripts/run/arenas/retro_mario/replay.sh` | Replay server bind address |
| Replay frame cap | Env `MAX_FRAMES` for `scripts/run/arenas/retro_mario/replay.sh` | Stop replay after a fixed number of frames |

Notes:

- Mario actions are JSON objects such as `{"move":"right_run_jump","hold_ticks":6}`.
- The parser defaults `hold_ticks` when the value is omitted and clamps it into the configured range.
- `display_mode: websocket` is treated as headless runtime plus websocketRGB streaming by the arena adapter.

## 7. Outputs and Replay

Run artifacts are written under:

```text
runs/<run_id>/
```

Replay can be started from one sample artifact:

```bash
bash scripts/run/arenas/retro_mario/replay.sh <run_id>
```

Useful replay environment variables:

- `PYTHON_BIN`: Python executable used by the replay script
- `HOST`: Bind host, default `127.0.0.1`
- `PORT`: Bind port, default `5800`
- `FPS`: Playback FPS, default `12`
- `MAX_FRAMES`: Frame cap, default `0` for unlimited
- `AUTO_OPEN`: Set `1` to auto-open the browser viewer

Replay URL:

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

Before launching replay, confirm the sample JSON has a replay payload such as `replay_path`.


