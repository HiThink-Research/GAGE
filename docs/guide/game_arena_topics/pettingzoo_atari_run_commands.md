# PettingZoo Atari Run Commands

English | [中文](pettingzoo_atari_run_commands_zh.md)

This document lists the script-based startup commands for all integrated PettingZoo Atari games.
Run these commands from the project root.

## 1. Standard Scripts

| Type | Script | Purpose |
| --- | --- | --- |
| Start game | `scripts/run/arenas/pettingzoo/run.sh` | Standard startup entry for AI, Dummy, ws dummy, and human record modes |
| Replay | `scripts/run/arenas/pettingzoo/replay.sh` | Replay one finished run by `run_id` |
| Live-view helper | `scripts/run/arenas/pettingzoo/viewer.sh` | Specialized websocketRGB helper for the `space_invaders_dummy_ws_rgb` smoke test |
| Run + replay helper | `scripts/run/arenas/replay/run_and_open.sh` | Convenience helper that runs one representative PettingZoo config and then opens replay |

## 2. Replay Script

Replay one finished run:

```bash
bash scripts/run/arenas/pettingzoo/replay.sh <run_id>
```

Example:

```bash
bash scripts/run/arenas/pettingzoo/replay.sh pettingzoo_space_invaders_ai_20260303_120000
```

## 3. AI Startup Commands

Set API key first:

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
```

All integrated AI games:

| Game ID | Script Command |
| --- | --- |
| `basketball_pong` | `bash scripts/run/arenas/pettingzoo/run.sh --game basketball_pong --mode ai` |
| `boxing` | `bash scripts/run/arenas/pettingzoo/run.sh --game boxing --mode ai` |
| `combat_plane` | `bash scripts/run/arenas/pettingzoo/run.sh --game combat_plane --mode ai` |
| `combat_tank` | `bash scripts/run/arenas/pettingzoo/run.sh --game combat_tank --mode ai` |
| `double_dunk` | `bash scripts/run/arenas/pettingzoo/run.sh --game double_dunk --mode ai` |
| `entombed_competitive` | `bash scripts/run/arenas/pettingzoo/run.sh --game entombed_competitive --mode ai` |
| `entombed_cooperative` | `bash scripts/run/arenas/pettingzoo/run.sh --game entombed_cooperative --mode ai` |
| `flag_capture` | `bash scripts/run/arenas/pettingzoo/run.sh --game flag_capture --mode ai` |
| `foozpong` | `bash scripts/run/arenas/pettingzoo/run.sh --game foozpong --mode ai` |
| `ice_hockey` | `bash scripts/run/arenas/pettingzoo/run.sh --game ice_hockey --mode ai` |
| `joust` | `bash scripts/run/arenas/pettingzoo/run.sh --game joust --mode ai` |
| `mario_bros` | `bash scripts/run/arenas/pettingzoo/run.sh --game mario_bros --mode ai` |
| `maze_craze` | `bash scripts/run/arenas/pettingzoo/run.sh --game maze_craze --mode ai` |
| `othello` | `bash scripts/run/arenas/pettingzoo/run.sh --game othello --mode ai` |
| `pong` | `bash scripts/run/arenas/pettingzoo/run.sh --game pong --mode ai` |
| `space_invaders` | `bash scripts/run/arenas/pettingzoo/run.sh --game space_invaders --mode ai` |
| `space_war` | `bash scripts/run/arenas/pettingzoo/run.sh --game space_war --mode ai` |
| `surround` | `bash scripts/run/arenas/pettingzoo/run.sh --game surround --mode ai` |
| `tennis` | `bash scripts/run/arenas/pettingzoo/run.sh --game tennis --mode ai` |
| `video_checkers` | `bash scripts/run/arenas/pettingzoo/run.sh --game video_checkers --mode ai` |
| `volleyball_pong` | `bash scripts/run/arenas/pettingzoo/run.sh --game volleyball_pong --mode ai` |
| `wizard_of_wor` | `bash scripts/run/arenas/pettingzoo/run.sh --game wizard_of_wor --mode ai` |

## 4. Dummy Startup Commands

All integrated Dummy games:

| Game ID | Script Command |
| --- | --- |
| `basketball_pong` | `bash scripts/run/arenas/pettingzoo/run.sh --game basketball_pong --mode dummy` |
| `boxing` | `bash scripts/run/arenas/pettingzoo/run.sh --game boxing --mode dummy` |
| `combat_plane` | `bash scripts/run/arenas/pettingzoo/run.sh --game combat_plane --mode dummy` |
| `combat_tank` | `bash scripts/run/arenas/pettingzoo/run.sh --game combat_tank --mode dummy` |
| `double_dunk` | `bash scripts/run/arenas/pettingzoo/run.sh --game double_dunk --mode dummy` |
| `entombed_competitive` | `bash scripts/run/arenas/pettingzoo/run.sh --game entombed_competitive --mode dummy` |
| `entombed_cooperative` | `bash scripts/run/arenas/pettingzoo/run.sh --game entombed_cooperative --mode dummy` |
| `flag_capture` | `bash scripts/run/arenas/pettingzoo/run.sh --game flag_capture --mode dummy` |
| `foozpong` | `bash scripts/run/arenas/pettingzoo/run.sh --game foozpong --mode dummy` |
| `ice_hockey` | `bash scripts/run/arenas/pettingzoo/run.sh --game ice_hockey --mode dummy` |
| `joust` | `bash scripts/run/arenas/pettingzoo/run.sh --game joust --mode dummy` |
| `mario_bros` | `bash scripts/run/arenas/pettingzoo/run.sh --game mario_bros --mode dummy` |
| `maze_craze` | `bash scripts/run/arenas/pettingzoo/run.sh --game maze_craze --mode dummy` |
| `othello` | `bash scripts/run/arenas/pettingzoo/run.sh --game othello --mode dummy` |
| `pong` | `bash scripts/run/arenas/pettingzoo/run.sh --game pong --mode dummy` |
| `space_invaders` | `bash scripts/run/arenas/pettingzoo/run.sh --game space_invaders --mode dummy` |
| `space_war` | `bash scripts/run/arenas/pettingzoo/run.sh --game space_war --mode dummy` |
| `surround` | `bash scripts/run/arenas/pettingzoo/run.sh --game surround --mode dummy` |
| `tennis` | `bash scripts/run/arenas/pettingzoo/run.sh --game tennis --mode dummy` |
| `video_checkers` | `bash scripts/run/arenas/pettingzoo/run.sh --game video_checkers --mode dummy` |
| `volleyball_pong` | `bash scripts/run/arenas/pettingzoo/run.sh --game volleyball_pong --mode dummy` |
| `wizard_of_wor` | `bash scripts/run/arenas/pettingzoo/run.sh --game wizard_of_wor --mode dummy` |

## 5. websocketRGB and Human Record Shortcuts

Recommended websocketRGB smoke test:

```bash
bash scripts/run/arenas/pettingzoo/run.sh --game space_invaders --mode ws_dummy
```

Human vs Human record mode:

```bash
bash scripts/run/arenas/pettingzoo/run.sh --game space_invaders --mode human_record
```

## 6. Run + Replay Helper

If you want one command that runs and then opens replay automatically, you can still use:

```bash
bash scripts/run/arenas/replay/run_and_open.sh --game pettingzoo --mode dummy
```

or:

```bash
OPENAI_API_KEY="<YOUR_KEY>" \
bash scripts/run/arenas/replay/run_and_open.sh --game pettingzoo --mode ai
```

Notes:

- This helper uses one representative PettingZoo config, not the full per-game matrix above.
- For per-game startup, prefer `scripts/run/arenas/pettingzoo/run.sh`.

## 7. Supplemental Guide

This section merges the former standalone PettingZoo Game Arena guide into this command index,
so installation, recommended startup paths, parameter locations, and replay details now live in one document.

### 7.1 Overview

PettingZoo Atari currently covers three common startup paths:

- LLM vs LLM with ws_rgb live view
- Standard LLM runs for API-backed evaluation
- Human vs Human record runs for browser-based interaction

The recommended entry is now the shared script workflow instead of hand-written `python run.py` commands.

### 7.2 Related Configs and Tools

| Type | Path | Purpose |
| --- | --- | --- |
| Config directory | `config/custom/pettingzoo/` | All PettingZoo game configs live here |
| Recommended AI ws_rgb config | `config/custom/pettingzoo/space_invaders_ai_ws_rgb.yaml` | Recommended live-view LLM vs LLM example |
| LiteLLM local-model example | `config/custom/pettingzoo/space_invaders_litellm_ai.yaml` | Example config for a local or self-hosted OpenAI-compatible model |
| Standard AI config | `config/custom/pettingzoo/space_invaders_ai.yaml` | Standard LLM demo without ws_rgb live view |
| Human record config | `config/custom/pettingzoo/space_invaders_human_vs_human_record.yaml` | Browser-based human input config |
| ws_rgb replay tool | `src/gage_eval/tools/ws_rgb_replay.py` | Replay server used underneath `replay.sh` |

### 7.3 Prerequisites

Use the same Python interpreter for installation and for `run.py`.

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
from pettingzoo.atari import space_invaders_v2

env = space_invaders_v2.env(render_mode="rgb_array")
env.reset(seed=0)
print("PettingZoo Atari ROM check: OK")
env.close()
PY
```

### 7.4 Recommended Startup Paths

#### 7.4.1 Recommended smoke test: dummy + ws_rgb live view

```bash
bash scripts/run/arenas/pettingzoo/run.sh \
  --game space_invaders \
  --mode ws_dummy \
  --run-id "pettingzoo_space_invaders_ws_dummy_$(date +%Y%m%d_%H%M%S)"
```

Use this first to validate PettingZoo Atari, ROM setup, and ws_rgb rendering before switching to model-backed runs.

#### 7.4.2 Recommended model example: LLM vs LLM with ws_rgb live view

```bash
OPENAI_API_KEY="<YOUR_KEY>" \
RUN_ID="pettingzoo_space_invaders_ai_ws_rgb_$(date +%Y%m%d_%H%M%S)" \
CONFIG=config/custom/pettingzoo/space_invaders_ai_ws_rgb.yaml \
bash scripts/run/arenas/pettingzoo/viewer.sh
```

Use this when you want AI-controlled gameplay and live browser rendering at the same time.

If you need to switch the model or API:

- API key: export `OPENAI_API_KEY` before launch. `viewer.sh` and `run.sh` also accept `LITELLM_API_KEY` and normalize it to `OPENAI_API_KEY`.
- Hosted OpenAI-compatible API: edit `config/custom/pettingzoo/space_invaders_ai_ws_rgb.yaml` under `backends[0].config`. Change `base_url` to switch the endpoint and `model` to switch the deployed model name.
- Local OpenAI-compatible service: point `base_url` to your local server and change `model` to the served model id.
- Existing local-model example: `config/custom/pettingzoo/space_invaders_litellm_ai.yaml` already uses `backends[].config.api_base` and `backends[].config.model`, with env helpers `PZ_LITELLM_API_BASE` and `PZ_LITELLM_MODEL`.

#### 7.4.3 Standard LLM run

```bash
export OPENAI_API_KEY="<YOUR_KEY>"

bash scripts/run/arenas/pettingzoo/run.sh \
  --game space_invaders \
  --mode ai \
  --run-id "pettingzoo_space_invaders_ai_$(date +%Y%m%d_%H%M%S)"
```

Supported startup modes in `scripts/run/arenas/pettingzoo/run.sh`:

- `dummy`: Maps to `<game>_dummy.yaml`
- `ai`: Maps to `<game>_ai.yaml`
- `ws_dummy`: Maps to `<game>_dummy_ws_rgb.yaml`
- `human_record`: Currently fixed to `space_invaders_human_vs_human_record.yaml`

Useful startup options:

- `--game`: PettingZoo game id such as `space_invaders`, `boxing`, or `pong`
- `--mode`: `dummy`, `ai`, `ws_dummy`, or `human_record`
- `--config`: Explicit YAML path; overrides the `--game` and `--mode` mapping
- `--run-id`: Output run id under `runs/`
- `--output-dir`: Output directory, default `runs`
- `--python-bin`: Explicit Python interpreter path

Useful environment variables:

- `WS_RGB_PORT`: Preferred viewer port for websocket modes
- `OPENAI_API_KEY`: Required for `ai` mode
- `LITELLM_API_KEY`: Accepted fallback; the script copies it into `OPENAI_API_KEY`
- `GAGE_EVAL_GAME_LOG_INLINE_LIMIT` and `GAGE_EVAL_GAME_LOG_INLINE_BYTES`: Auto-set by the script unless you override them

#### 7.4.4 Human vs Human record run

```bash
bash scripts/run/arenas/pettingzoo/run.sh \
  --game space_invaders \
  --mode human_record \
  --run-id "pettingzoo_space_invaders_human_record_$(date +%Y%m%d_%H%M%S)"
```

Default runtime endpoints printed by the script:

- Viewer URL: `http://127.0.0.1:5800/ws_rgb/viewer`
- Input queue: `http://127.0.0.1:8001`

Default key map in `space_invaders_human_vs_human_record.yaml`:

- `player_0`: `Q/W/E/A/S/D`
- `player_1`: `U/I/O/J/K/L`

#### 7.4.5 Explicit config override

Use `--config` if you want the script wrapper but need a custom YAML path:

```bash
bash scripts/run/arenas/pettingzoo/run.sh \
  --config config/custom/pettingzoo/boxing_dummy.yaml \
  --run-id "pettingzoo_boxing_custom_$(date +%Y%m%d_%H%M%S)"
```

### 7.5 Recommended Execution Order

The recommended order for PettingZoo demos is:

1. Install PettingZoo Atari dependencies and ROMs.
2. Decide the game and startup mode, or prepare a custom config path.
3. Set API keys only when using `--mode ai`.
4. Validate the environment with the dummy websocket path first, then switch to `viewer.sh` or `run.sh` for model-backed modes.
5. Open the viewer during the run for `human_record`; the AI ws_rgb helper waits for the viewer automatically.
6. Start `scripts/run/arenas/pettingzoo/replay.sh <run_id>` after the run if you want post-run playback.

### 7.6 Key Parameters and Where to Change Them

| Item | Where to change | Meaning |
| --- | --- | --- |
| API key | Shell env: `OPENAI_API_KEY` or `LITELLM_API_KEY` | Required by `--mode ai`; the script accepts either and normalizes to `OPENAI_API_KEY` |
| Game selection | `scripts/run/arenas/pettingzoo/run.sh --game` or config `environment.env_id` | Switch the Atari game and runtime env id |
| Startup mode | `scripts/run/arenas/pettingzoo/run.sh --mode` | Select `dummy`, `ai`, `ws_dummy`, or `human_record` |
| Custom config | `scripts/run/arenas/pettingzoo/run.sh --config` | Bypass the built-in game-to-config mapping |
| Model endpoint | `backends[].config.base_url` in `*_ai.yaml`, or `backends[].config.api_base` in `space_invaders_litellm_ai.yaml` | Switch between hosted and local OpenAI-compatible endpoints |
| Model name | `backends[].config.model`, or env `PZ_LITELLM_MODEL` in the LiteLLM example | Select the model used by the AI players |
| API-key enforcement | `backends[].config.require_api_key` in `*_ai.yaml` | Keep it `true` for hosted APIs; disable only for trusted local gateways |
| Runtime length | `environment.env_kwargs.max_cycles` | Environment frame limit |
| Arena turn cap | `scheduler.max_turns` | Arena-side maximum turn count |
| Viewer mode | `environment.display_mode` | Use `websocket` for ws_rgb live view |
| Viewer host/port | `human_input.ws_host` / `human_input.ws_port` and env `WS_RGB_PORT` | Bind address for the live viewer; the script passes `WS_RGB_PORT` through |
| Replay playback FPS | Env `FPS` for `scripts/run/arenas/pettingzoo/replay.sh` | Replay speed in the post-run viewer |
| Replay host/port | Env `HOST` and `PORT` for `scripts/run/arenas/pettingzoo/replay.sh` | Replay server bind address |
| Replay frame cap | Env `MAX_FRAMES` for `scripts/run/arenas/pettingzoo/replay.sh` | Stop replay after a fixed number of frames |
| Action labels | `environment.use_action_meanings` | `true` shows names such as `FIRE`, `LEFT`; `false` uses numeric ids |
| Human key mapping | `environment.action_schema.key_map` | Browser key-to-action mapping for record-mode human control |
| Human input endpoint | `human_input.host` / `human_input.port` | Queue input server for record-mode human control |
| Inlined replay log | Env `GAGE_EVAL_GAME_LOG_INLINE_LIMIT` and `GAGE_EVAL_GAME_LOG_INLINE_BYTES` | Keep the game log inside sample JSON so replay works without a rerun |

Notes:

- Most game configs use `max_cycles: 300`; `space_invaders_*` uses `3000`.
- The startup script only wraps config selection and shared runtime defaults. Game behavior still comes from the selected YAML.

### 7.7 Outputs and Replay Details

Run artifacts are written under:

```text
runs/<run_id>/
```

For post-run replay:

```bash
bash scripts/run/arenas/pettingzoo/replay.sh <run_id>
```

Useful replay environment variables:

- `PYTHON_BIN`: Python executable used by the replay script
- `HOST`: Bind host, default `127.0.0.1`
- `PORT`: Bind port, default `5800`
- `FPS`: Playback FPS, default `12`
- `MAX_FRAMES`: Frame cap, default `0` for unlimited
- `AUTO_OPEN`: Set `1` to auto-open the browser viewer

Default replay URL:

```text
http://127.0.0.1:5800/ws_rgb/viewer
```
