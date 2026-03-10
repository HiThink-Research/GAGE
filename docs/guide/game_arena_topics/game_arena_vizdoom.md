# Game Arena ViZDoom Guide

English | [中文](game_arena_vizdoom_zh.md)

This is the canonical Game Arena guide for ViZDoom in this repository. It consolidates installation, unified startup entrypoints, replay, and parameter locations into one document.

## 1. Overview

ViZDoom currently provides two startup styles:

- websocketRGB-based human vs LLM interaction
- Pygame-based local input or backend-only runs for the other paths

The recommended validation flow is: dummy smoke test first, then the browser-based `human vs LLM` path.

## 2. Canonical Files

| Type | Path | Purpose |
| --- | --- | --- |
| websocketRGB helper | `scripts/run/arenas/vizdoom/viewer.sh` | Generic helper that waits for the websocketRGB viewer to become reachable |
| Human vs LLM websocketRGB config | `config/custom/vizdoom_human_vs_llm_tick_ws_rgb_strategy.yaml` | Recommended browser-based human vs LLM example with tick scheduler |
| Dummy websocketRGB config | `config/custom/vizdoom_dummy_vs_dummy_ws_rgb.yaml` | Optional dummy-only websocket config for environment checks |
| Human vs Dummy script | `scripts/run/arenas/vizdoom/run.sh --mode human-vs-dummy` | Local pygame validation run |
| Human Solo script | `scripts/run/arenas/vizdoom/run.sh --mode human-solo` | Human-only playground |
| Human vs LLM script | `scripts/run/arenas/vizdoom/run.sh --mode human-vs-llm` | Local human input against an LLM |
| Human vs LLM record script | `scripts/run/arenas/vizdoom/run.sh --mode human-vs-llm-record` | Record scheduler variant |
| LLM vs LLM script | `scripts/run/arenas/vizdoom/run.sh --mode llm-vs-llm` | Two LLM players |
| AI vs AI script | `scripts/run/arenas/vizdoom/run.sh --mode ai-vs-ai` | Alias path for the AI-vs-AI flow |
| Agent vs LLM script | `scripts/run/arenas/vizdoom/run.sh --mode agent-vs-llm` | Human/agent incremental variant |
| Replay script | `scripts/run/arenas/vizdoom/replay.sh` | Replay a finished run by `run_id` |
| Websocket human config | `config/custom/vizdoom_human_vs_llm_record_ws_rgb.yaml` | Browser-based human input through websocketRGB |
| Main config directory | `config/custom/vizdoom_*.yaml` | All ViZDoom run configs |

## 3. Prerequisites

Install project dependencies first:

```bash
pip install -r requirements.txt
```

Minimum verification:

```bash
python - <<'PY'
import vizdoom
import pygame
print("vizdoom ok")
print("pygame ok")
PY
```

Notes:

- Human modes that use the canonical run scripts require a desktop environment because input is captured by a local `pygame` window.
- LLM-backed modes require `OPENAI_API_KEY` or `LITELLM_API_KEY`.

## 4. Startup Paths

### 4.1 Recommended smoke test: dummy + websocketRGB

```bash
RUN_ID="vizdoom_human_vs_llm_ws_rgb_$(date +%Y%m%d_%H%M%S)" \
RUN_ID="vizdoom_dummy_ws_rgb_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run/arenas/vizdoom/viewer.sh
```

Use this first to validate ViZDoom, websocketRGB, and browser viewing before adding model-backed runs.

What the websocketRGB helper does:

1. Picks a Python executable.
2. Validates the config path.
3. Chooses a free `WS_RGB_PORT`.
4. Starts `python run.py --config ...` in the background.
5. Waits until `http://127.0.0.1:<port>/ws_rgb/viewer` is reachable.
6. Prints the ready viewer URL and optionally auto-opens the browser.

Useful variables for this example:

- `PYTHON_BIN`: Python executable
- `CFG`: Defaults to `config/custom/vizdoom_dummy_vs_dummy_ws_rgb.yaml`
- `RUN_ID`: Output run id under `runs/`
- `OUTPUT_DIR`: Defaults to `runs`
- `WS_RGB_HOST`: Defaults to `127.0.0.1`
- `WS_RGB_PORT`: Defaults to `5800`

If you want the local-window validation path instead, use:

```bash
bash scripts/run/arenas/vizdoom/run.sh --mode human-vs-dummy
```

Default local-script variables:

- `PYTHON_BIN`: Python executable
- `CFG`: Defaults to `config/custom/vizdoom_human_vs_dummy.yaml`
- `RUN_ID`: Defaults to `vizdoom_human_vs_dummy_<timestamp>`
- `OUTPUT_DIR`: Defaults to `runs/`

Local key map printed by the pygame script:

- `A` or `Left`: action `2`
- `D` or `Right`: action `3`
- `Space` or `J`: action `1`

### 4.2 Recommended model example: human vs LLM with websocketRGB live view

```bash
OPENAI_API_KEY="<YOUR_KEY>" \
VIZDOOM_P1_SCHEME_ID=S3_text_image_current \
RUN_ID="vizdoom_human_vs_llm_ws_rgb_$(date +%Y%m%d_%H%M%S)" \
CFG=config/custom/vizdoom_human_vs_llm_tick_ws_rgb_strategy.yaml \
bash scripts/run/arenas/vizdoom/viewer.sh
```

This is the recommended documented example when you want a real browser-based `human vs LLM` session.

Useful variables for this example:

- `PYTHON_BIN`: Python executable
- `CFG`: Use `config/custom/vizdoom_human_vs_llm_tick_ws_rgb_strategy.yaml` for the documented human-vs-LLM path
- `VIZDOOM_P1_SCHEME_ID`: Selects the LLM strategy scheme, such as `S3_text_image_current`
- `RUN_ID`: Output run id under `runs/`
- `OUTPUT_DIR`: Defaults to `runs`
- `WS_RGB_HOST`: Defaults to `127.0.0.1`
- `WS_RGB_PORT`: Defaults to `5800`

Where to change the model/API for this command:

- API key: export `OPENAI_API_KEY` before launch. The wrapper also accepts `LITELLM_API_KEY` and copies it into `OPENAI_API_KEY`. Keep one of them non-empty even for a local OpenAI-compatible gateway, because the startup scripts check it before starting.
- Recommended websocketRGB example: edit `config/custom/vizdoom_human_vs_llm_tick_ws_rgb_strategy.yaml` under `backends[0].config`. Change `api_base` to switch the endpoint, keep `provider: openai` for OpenAI-compatible services, and change `model` or env `VIZDOOM_P1_MODEL` to switch the served model.
- Other model commands use the same fields in their own YAMLs: `human-vs-llm` uses `config/custom/vizdoom_human_vs_llm.yaml`, `human-vs-llm-record` uses `config/custom/vizdoom_human_vs_llm_record.yaml`, and `llm-vs-llm` uses `config/custom/vizdoom_llm_vs_llm.yaml`.

### 4.3 Human vs LLM

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/run/arenas/vizdoom/run.sh --mode human-vs-llm
```

The script also accepts `LITELLM_API_KEY`. If only `LITELLM_API_KEY` is set, the script copies it into `OPENAI_API_KEY`.

### 4.4 Record scheduler variant

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/run/arenas/vizdoom/run.sh --mode human-vs-llm-record
```

### 4.5 LLM vs LLM

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/run/arenas/vizdoom/run.sh --mode llm-vs-llm
```

### 4.6 Optional websocket human input

Use this when you want browser-based human input instead of the local `pygame` window:

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
PYTHONPATH=src python run.py \
  --config config/custom/vizdoom_human_vs_llm_record_ws_rgb.yaml \
  --output-dir runs \
  --run-id vizdoom_human_vs_llm_record_ws
```

Default viewer URL for this websocket config:

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

## 5. Execution Order

The current ViZDoom startup sequence is:

1. Install optional runtime dependencies such as `vizdoom` and `pygame`.
2. Validate the environment with the dummy websocketRGB path first, then switch to `vizdoom_human_vs_llm_tick_ws_rgb_strategy.yaml` or another `scripts/run/arenas/vizdoom/run.sh --mode ...` path for model-backed runs.
3. Set API keys when the selected mode includes an LLM backend.
4. Start the run.
5. For websocketRGB helper runs, wait for the printed viewer URL; for pygame modes, keep the local input window focused while playing.
6. After the run, use `scripts/run/arenas/vizdoom/replay.sh` for replay.

## 6. Key Parameters and Where to Change Them

| Item | Where to change | Meaning |
| --- | --- | --- |
| API key | Shell env: `OPENAI_API_KEY` or `LITELLM_API_KEY` | Required by LLM-backed scripts |
| Script config path | Script variable `CFG` | Swap the YAML used by the canonical runner |
| Backend endpoint | `backends[].config.api_base` in the selected `config/custom/vizdoom_*.yaml` | Switch between hosted and local OpenAI-compatible endpoints |
| Model name | `backends[].config.model` and env `VIZDOOM_P1_MODEL` in the strategy websocketRGB config | Select the LLM model |
| Output location | Script variable `OUTPUT_DIR` | Base directory for run artifacts |
| Scheduler pace | `scheduler.tick_ms` | Tick interval in milliseconds |
| Record timeout | `scheduler.action_timeout_ms` | Human/LLM wait timeout in record mode |
| Step budget | `environment.max_steps` | Maximum backend steps |
| Action repeat | `environment.action_repeat` | Number of backend frames per action |
| Runtime pacing | `environment.sleep_s` | Extra delay between backend ticks |
| Local rendering | `environment.render_mode` | Window/display behavior |
| POV and automap | `environment.show_pov`, `environment.show_automap` | Control visual channels |
| Replay output dir | `environment.replay_output_dir` | Replay file output location |
| Replay capture | `environment.replay.*` | Action/frame replay settings |
| Replay playback FPS | `scripts/run/arenas/vizdoom/replay.sh` env `FPS` | Replay speed in the viewer |
| Websocket ports | `human_input.host`, `human_input.port`, `human_input.ws_port` | Browser input and viewer ports in websocket configs |

Frame-rate related notes:

- For normal scripted runs, the main pacing knobs are `scheduler.tick_ms`, `environment.action_repeat`, and `environment.sleep_s`.
- Replay playback speed is controlled separately by `scripts/run/arenas/vizdoom/replay.sh` through `FPS`.
- If you explicitly enable `environment.config_path: src/gage_eval/role/arena/games/vizdoom/_vizdoom.ini`, engine-side FPS caps are in that INI file via `cl_capfps` and `vid_maxfps`.

## 7. Outputs and Replay

Run artifacts are written under:

```text
runs/<run_id>/
```

The environment configs also default replay files to:

```text
runs/vizdoom_replays/
```

Replay command:

```bash
bash scripts/run/arenas/vizdoom/replay.sh <run_id>
```

Useful replay variables:

- `PYTHON_BIN`: Python executable
- `HOST`: Bind host, default `127.0.0.1`
- `PORT`: Replay viewer port, default `5800`
- `FPS`: Replay playback fps, default `8`
- `MAX_FRAMES`: Optional replay frame cap

Default replay URL:

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

## 8. Legacy Docs

Older ViZDoom docs are still in the repository but are no longer the standard entry:

- `docs/guide/vizdoom_setup_guide.md`
- `docs/guide/vizdoom_setup_guide_zh.md`
- `docs/guide/vizdoom_arena_summary.md`
- `docs/guide/vizdoom_arena_summary_zh.md`
