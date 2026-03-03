# Game Arena ViZDoom Guide

English | [中文](game_arena_vizdoom_zh.md)

This is the canonical Game Arena guide for ViZDoom in this repository. It consolidates installation, one-click startup, replay, and parameter locations into one document.

## 1. Overview

ViZDoom currently provides two startup styles:

- Pygame-based local human input through one-click scripts
- Optional ws_rgb browser input through a manual websocket config

The standard entry for most users is still the one-click script set under `scripts/oneclick/`.

## 2. Canonical Files

| Type | Path | Purpose |
| --- | --- | --- |
| Human vs Dummy script | `scripts/oneclick/run_vizdoom_human_vs_dummy.sh` | Recommended first validation run |
| Human Solo script | `scripts/oneclick/run_vizdoom_human_solo.sh` | Human-only playground |
| Human vs LLM script | `scripts/oneclick/run_vizdoom_human_vs_llm.sh` | Local human input against an LLM |
| Human vs LLM record script | `scripts/oneclick/run_vizdoom_human_vs_llm_record.sh` | Record scheduler variant |
| LLM vs LLM script | `scripts/oneclick/run_vizdoom_llm_vs_llm.sh` | Two LLM players |
| AI vs AI script | `scripts/oneclick/run_vizdoom_ai_vs_ai.sh` | Alias script that defaults to `config/custom/vizdoom_llm_vs_llm.yaml` |
| Agent vs LLM script | `scripts/oneclick/run_vizdoom_agent_vs_llm.sh` | Human/agent incremental variant |
| Replay script | `scripts/oneclick/run_vizdoom_replay.sh` | Replay a finished run by `run_id` |
| Websocket human config | `config/custom/vizdoom_human_vs_llm_record_ws_rgb.yaml` | Browser-based human input through ws_rgb |
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

- Human modes that use the one-click scripts require a desktop environment because input is captured by a local `pygame` window.
- LLM-backed modes require `OPENAI_API_KEY` or `LITELLM_API_KEY`.

## 4. Startup Paths

### 4.1 Recommended smoke test: human vs dummy

```bash
bash scripts/oneclick/run_vizdoom_human_vs_dummy.sh
```

Default script variables:

- `PYTHON_BIN`: Python executable
- `CFG`: Defaults to `config/custom/vizdoom_human_vs_dummy.yaml`
- `RUN_ID`: Defaults to `vizdoom_human_vs_dummy_<timestamp>`
- `OUTPUT_DIR`: Defaults to `runs/`

Local key map printed by the script:

- `A` or `Left`: action `2`
- `D` or `Right`: action `3`
- `Space` or `J`: action `1`

### 4.2 Human vs LLM

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/oneclick/run_vizdoom_human_vs_llm.sh
```

The script also accepts `LITELLM_API_KEY`. If only `LITELLM_API_KEY` is set, the script copies it into `OPENAI_API_KEY`.

### 4.3 Record scheduler variant

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/oneclick/run_vizdoom_human_vs_llm_record.sh
```

### 4.4 LLM vs LLM

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/oneclick/run_vizdoom_llm_vs_llm.sh
```

### 4.5 Optional websocket human input

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
2. Choose a one-click script or a YAML config.
3. Set API keys when the selected mode includes an LLM backend.
4. Start the run.
5. For pygame modes, keep the local input window focused while playing.
6. After the run, use `run_vizdoom_replay.sh` for replay.

## 6. Key Parameters and Where to Change Them

| Item | Where to change | Meaning |
| --- | --- | --- |
| API key | Shell env: `OPENAI_API_KEY` or `LITELLM_API_KEY` | Required by LLM-backed scripts |
| Script config path | Script variable `CFG` | Swap the YAML used by a one-click script |
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
| Replay playback FPS | `scripts/oneclick/run_vizdoom_replay.sh` env `FPS` | Replay speed in the viewer |
| Websocket ports | `human_input.host`, `human_input.port`, `human_input.ws_port` | Browser input and viewer ports in websocket configs |

Frame-rate related notes:

- For normal one-click runs, the main pacing knobs are `scheduler.tick_ms`, `environment.action_repeat`, and `environment.sleep_s`.
- Replay playback speed is controlled separately by `run_vizdoom_replay.sh` through `FPS`.
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
bash scripts/oneclick/run_vizdoom_replay.sh <run_id>
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
