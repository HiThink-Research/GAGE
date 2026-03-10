# ws_rgb Replay Usage Guide

English | [中文](ws_rgb_replay_usage_guide_zh.md)

This guide explains how to run and replay game matches with the unified `ws_rgb` replay flow.
All commands are aligned with `docs/local/指令精简.md`.

For live rendering, input routing, and integration contracts (beyond replay), see:
`docs/guide/game_arena_topics/ws_rgb_runtime_dev_guide.md`

## 1. Scope

- Replay viewer endpoint: `/ws_rgb/viewer`
- Replay service entrypoint: `python -m gage_eval.tools.ws_rgb_replay`
- Unified replay helper: `scripts/run/arenas/replay/run_and_open.sh`
- Supported games: `gomoku`, `tictactoe`, `doudizhu`, `mahjong`, `pettingzoo`

## 2. Prerequisites

Run from repository root:

```bash
cd /path/to/GAGE
```

PettingZoo Atari needs ROM setup before first run in a fresh environment. Use the same Python interpreter as `run.py`:

```bash
# 0) Pick the same interpreter used by run.py
# Replace it with your conda/venv python if needed
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -m pip -V

# 1) Install Atari dependencies
"$PYTHON_BIN" -m pip install -U \
  "pettingzoo[atari]>=1.24.3" \
  "shimmy[atari]>=1.0.0" \
  "AutoROM[accept-rom-license]>=0.6.1"

# 2) Download and install ROMs
# NOTE: Use module invocation to avoid broken AutoROM shebangs after env migration
"$PYTHON_BIN" -m AutoROM.AutoROM --accept-license
```

Minimal verification (recommended):

```bash
"$PYTHON_BIN" - <<'PY'
from pettingzoo.atari import pong_v3

env = pong_v3.env(render_mode="rgb_array")
env.reset(seed=0)
print("PettingZoo Atari ROM check: OK")
env.close()
PY
```

If you hit `AutoROM: bad interpreter` or `AutoROM: command not found`:

```bash
"$PYTHON_BIN" -m pip install --force-reinstall "AutoROM[accept-rom-license]>=0.6.1"
"$PYTHON_BIN" -m AutoROM.AutoROM --accept-license
```

For AI mode:

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
```

## 3. Replay Workflows

### 3.1 One-Click Replay (Recommended)

This workflow is `run -> replay` in one command.

Generic form:

```bash
bash scripts/run/arenas/replay/run_and_open.sh --game <game> --mode <dummy|ai>
```

Dummy:

```bash
bash scripts/run/arenas/replay/run_and_open.sh --game gomoku --mode dummy
bash scripts/run/arenas/replay/run_and_open.sh --game tictactoe --mode dummy
bash scripts/run/arenas/replay/run_and_open.sh --game doudizhu --mode dummy
bash scripts/run/arenas/replay/run_and_open.sh --game mahjong --mode dummy
bash scripts/run/arenas/replay/run_and_open.sh --game pettingzoo --mode dummy
```

AI:

```bash
bash scripts/run/arenas/replay/run_and_open.sh --game gomoku --mode ai
bash scripts/run/arenas/replay/run_and_open.sh --game tictactoe --mode ai
bash scripts/run/arenas/replay/run_and_open.sh --game doudizhu --mode ai
bash scripts/run/arenas/replay/run_and_open.sh --game mahjong --mode ai
bash scripts/run/arenas/replay/run_and_open.sh --game pettingzoo --mode ai
```

Common options:

```bash
bash scripts/run/arenas/replay/run_and_open.sh \
  --game gomoku \
  --mode dummy \
  --port 5860 \
  --auto-open 0
```

```bash
bash scripts/run/arenas/replay/run_and_open.sh \
  --game mahjong \
  --mode ai \
  --python-bin "$(command -v python)" \
  --run-id mahjong_ai_replay_demo
```

### 3.2 Post-Run Manual Replay (PettingZoo example)

If you already finished a run and want replay later, use this flow.

Dummy run:

```bash
python run.py --config config/custom/pettingzoo/pong_dummy.yaml --output-dir runs --run-id pettingzoo_dummy_run
```

Replay from artifacts:

```bash
RUN_ID=pettingzoo_dummy_run
SAMPLE_JSON=$(find "runs/${RUN_ID}/samples" -name '*.json' | head -n 1)

PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo \
  --auto-open 1
```

AI run:

```bash
python run.py --config config/custom/pettingzoo/pong_ai.yaml --output-dir runs --run-id pettingzoo_ai_run
```

Replay from artifacts:

```bash
RUN_ID=pettingzoo_ai_run
SAMPLE_JSON=$(find "runs/${RUN_ID}/samples" -name '*.json' | head -n 1)

PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo \
  --auto-open 1
```

## 4. Notes and Troubleshooting

- Viewer URL is printed by `ws_rgb_replay` at startup.
- Use `--auto-open 1` to open the browser automatically (open manually in headless environments).
- `BrokenPipeError` during `/ws_rgb/frame_image` usually means the browser canceled a request; server now tolerates this.
- If replay cannot start, check:
  - `RUN_ID` exists under `runs/<run_id>/samples`
  - the selected sample has replay artifacts in `predict_result[*].replay_path/replay_v1_path`
  - port is free (`--port`)
