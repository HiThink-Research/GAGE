# Game Arena PettingZoo Guide

English | [中文](game_arena_pettingzoo_zh.md)

This is the canonical Game Arena guide for PettingZoo Atari in this repository. It consolidates installation, script-based startup, replay, and parameter-tuning notes into one place.

## 1. Overview

PettingZoo Atari currently covers three common paths:

- Dummy run for environment and viewer verification
- LLM vs LLM run for API-backed evaluation
- Human vs Human record run for browser-based interaction

These paths now share a common script entry. Use this guide as the standard entry. Older PettingZoo docs are kept for reference only.

## 2. Canonical Files

| Type | Path | Purpose |
| --- | --- | --- |
| Standard startup script | `scripts/oneclick/run_pettingzoo_game.sh` | Main entry for per-game Dummy, AI, ws_rgb Dummy, and human record runs |
| Replay script | `scripts/oneclick/run_pettingzoo_replay.sh` | Replay one finished run by `run_id` |
| ws_rgb helper | `scripts/oneclick/run_pettingzoo_ws_rgb_viewer.sh` | Specialized helper for the `space_invaders_dummy_ws_rgb` smoke test |
| Config directory | `config/custom/pettingzoo/` | All PettingZoo game configs live here |
| Recommended dummy config | `config/custom/pettingzoo/space_invaders_dummy_ws_rgb.yaml` | Recommended ws_rgb smoke test that matches the legacy PettingZoo examples |
| Recommended AI config | `config/custom/pettingzoo/space_invaders_ai.yaml` | Standard LLM demo |
| Human record config | `config/custom/pettingzoo/space_invaders_human_vs_human_record.yaml` | Browser-based human input |
| Supplemental command index | `docs/guide/pettingzoo_atari_run_commands.md` | Full per-game AI and Dummy script list |
| Replay tool | `src/gage_eval/tools/ws_rgb_replay.py` | Underlying replay server used by the replay script |

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
from pettingzoo.atari import space_invaders_v2

env = space_invaders_v2.env(render_mode="rgb_array")
env.reset(seed=0)
print("PettingZoo Atari ROM check: OK")
env.close()
PY
```

## 4. Startup Paths

### 4.1 Recommended smoke test: ws_rgb dummy run

```bash
bash scripts/oneclick/run_pettingzoo_game.sh \
  --game space_invaders \
  --mode ws_dummy \
  --run-id "pettingzoo_space_invaders_ws_dummy_$(date +%Y%m%d_%H%M%S)"
```

What the standard startup script does:

1. Picks a Python executable.
2. Resolves the config from `--game` and `--mode`, unless `--config` is provided.
3. Validates the config path.
4. Exports replay-friendly `GAGE_EVAL_GAME_LOG_INLINE_*` defaults.
5. Runs `python run.py --config ...`.
6. Prints the live viewer URL for websocket modes.

If you want the specialized helper for the default `space_invaders_dummy_ws_rgb` config, use:

```bash
CONFIG=config/custom/pettingzoo/space_invaders_dummy_ws_rgb.yaml \
bash scripts/oneclick/run_pettingzoo_ws_rgb_viewer.sh
```

Supported startup modes in `run_pettingzoo_game.sh`:

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

### 4.2 LLM run

```bash
export OPENAI_API_KEY="<YOUR_KEY>"

bash scripts/oneclick/run_pettingzoo_game.sh \
  --game space_invaders \
  --mode ai \
  --run-id "pettingzoo_space_invaders_ai_$(date +%Y%m%d_%H%M%S)"
```

You can also keep the key in `LITELLM_API_KEY`; the startup script reuses it automatically for `ai` mode.

### 4.3 Human vs Human record run

```bash
bash scripts/oneclick/run_pettingzoo_game.sh \
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

### 4.4 Explicit config override

Use `--config` if you want the script wrapper but need a custom YAML path:

```bash
bash scripts/oneclick/run_pettingzoo_game.sh \
  --config config/custom/pettingzoo/boxing_dummy.yaml \
  --run-id "pettingzoo_boxing_custom_$(date +%Y%m%d_%H%M%S)"
```

## 5. Execution Order

The repository now follows this order for PettingZoo demos:

1. Install PettingZoo Atari dependencies and ROMs.
2. Decide the game and startup mode, or prepare a custom config path.
3. Set API keys only when using `--mode ai`.
4. Start the run with `scripts/oneclick/run_pettingzoo_game.sh`.
5. Open the viewer during the run for `ws_dummy` or `human_record`.
6. Start `scripts/oneclick/run_pettingzoo_replay.sh <run_id>` after the run if you want post-run playback.

## 6. Key Parameters and Where to Change Them

| Item | Where to change | Meaning |
| --- | --- | --- |
| API key | Shell env: `OPENAI_API_KEY` or `LITELLM_API_KEY` | Required by `--mode ai`; the script accepts either and normalizes to `OPENAI_API_KEY` |
| Game selection | `run_pettingzoo_game.sh --game` or config `environment.env_id` | Switch the Atari game and runtime env id |
| Startup mode | `run_pettingzoo_game.sh --mode` | Select `dummy`, `ai`, `ws_dummy`, or `human_record` |
| Custom config | `run_pettingzoo_game.sh --config` | Bypass the built-in game-to-config mapping |
| Runtime length | `environment.env_kwargs.max_cycles` | Environment frame limit |
| Arena turn cap | `scheduler.max_turns` | Arena-side maximum turn count |
| Viewer mode | `environment.display_mode` | Use `websocket` for ws_rgb live view |
| Viewer host/port | `human_input.ws_host` / `human_input.ws_port` and env `WS_RGB_PORT` | Bind address for the live viewer; the script passes `WS_RGB_PORT` through |
| Replay playback FPS | Env `FPS` for `run_pettingzoo_replay.sh` | Replay speed in the post-run viewer |
| Replay host/port | Env `HOST` and `PORT` for `run_pettingzoo_replay.sh` | Replay server bind address |
| Replay frame cap | Env `MAX_FRAMES` for `run_pettingzoo_replay.sh` | Stop replay after a fixed number of frames |
| Action labels | `environment.use_action_meanings` | `true` shows names such as `FIRE`, `LEFT`; `false` uses numeric ids |
| Human key mapping | `environment.action_schema.key_map` | Browser key-to-action mapping for record-mode human control |
| Human input endpoint | `human_input.host` / `human_input.port` | Queue input server for record-mode human control |
| Inlined replay log | Env `GAGE_EVAL_GAME_LOG_INLINE_LIMIT` and `GAGE_EVAL_GAME_LOG_INLINE_BYTES` | Keep the game log inside sample JSON so replay works without a rerun |

Notes:

- Most game configs use `max_cycles: 300`; `space_invaders_*` uses `3000`.
- The startup script only wraps config selection and shared runtime defaults. Game behavior still comes from the selected YAML.

## 7. Outputs and Replay

Run artifacts are written under:

```text
runs/<run_id>/
```

For post-run replay:

```bash
bash scripts/oneclick/run_pettingzoo_replay.sh <run_id>
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

## 8. Other Docs

The following files are still kept in the repository:

- `docs/guide/pettingzoo_user_guide.md`
- `docs/guide/pettingzoo_user_guide_zh.md`
- `docs/guide/pettingzoo_atari_run_commands.md`
- `docs/guide/pettingzoo_atari_run_commands_zh.md`
