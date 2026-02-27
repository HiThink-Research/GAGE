# ws_rgb Replay Usage Guide

English | [中文](ws_rgb_replay_usage_guide_zh.md)

This guide explains how to run and replay game matches with the unified `ws_rgb` replay flow.
All commands are aligned with `docs/local/指令精简.md`.

## 1. Scope

- Replay viewer endpoint: `/ws_rgb/viewer`
- Replay service entrypoint: `python -m gage_eval.tools.ws_rgb_replay`
- One-click helper: `scripts/oneclick/run_game_replay_oneclick.sh`
- Supported games: `gomoku`, `tictactoe`, `doudizhu`, `mahjong`, `pettingzoo`

## 2. Prerequisites

Run from repository root:

```bash
cd /Users/shuo/code/GAGE
```

PettingZoo Atari needs ROM setup on first run:

```bash
/Users/shuo/mamba/envs/gage/bin/AutoROM --accept-license
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
bash scripts/oneclick/run_game_replay_oneclick.sh --game <game> --mode <dummy|ai>
```

Dummy:

```bash
bash scripts/oneclick/run_game_replay_oneclick.sh --game gomoku --mode dummy
bash scripts/oneclick/run_game_replay_oneclick.sh --game tictactoe --mode dummy
bash scripts/oneclick/run_game_replay_oneclick.sh --game doudizhu --mode dummy
bash scripts/oneclick/run_game_replay_oneclick.sh --game mahjong --mode dummy
bash scripts/oneclick/run_game_replay_oneclick.sh --game pettingzoo --mode dummy
```

AI:

```bash
bash scripts/oneclick/run_game_replay_oneclick.sh --game gomoku --mode ai
bash scripts/oneclick/run_game_replay_oneclick.sh --game tictactoe --mode ai
bash scripts/oneclick/run_game_replay_oneclick.sh --game doudizhu --mode ai
bash scripts/oneclick/run_game_replay_oneclick.sh --game mahjong --mode ai
bash scripts/oneclick/run_game_replay_oneclick.sh --game pettingzoo --mode ai
```

Common options:

```bash
bash scripts/oneclick/run_game_replay_oneclick.sh \
  --game gomoku \
  --mode dummy \
  --port 5860 \
  --auto-open 0
```

```bash
bash scripts/oneclick/run_game_replay_oneclick.sh \
  --game mahjong \
  --mode ai \
  --python-bin /Users/shuo/mamba/envs/gage/bin/python \
  --run-id mahjong_ai_replay_demo
```

### 3.2 Post-Run Manual Replay (PettingZoo example)

If you already finished a run and want replay later, use this flow.

Dummy run:

```bash
/Users/shuo/mamba/envs/gage/bin/python run.py --config config/custom/pettingzoo/pong_dummy.yaml --output-dir runs --run-id pettingzoo_dummy_run
```

Replay from artifacts:

```bash
RUN_ID=pettingzoo_dummy_run
SAMPLE_JSON=$(find "runs/${RUN_ID}/samples" -name '*.json' | head -n 1)

PYTHONPATH=src /Users/shuo/mamba/envs/gage/bin/python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo
```

AI run:

```bash
/Users/shuo/mamba/envs/gage/bin/python run.py --config config/custom/pettingzoo/pong_ai.yaml --output-dir runs --run-id pettingzoo_ai_run
```

Replay from artifacts:

```bash
RUN_ID=pettingzoo_ai_run
SAMPLE_JSON=$(find "runs/${RUN_ID}/samples" -name '*.json' | head -n 1)

PYTHONPATH=src /Users/shuo/mamba/envs/gage/bin/python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo
```

## 4. Notes and Troubleshooting

- Viewer URL is printed by `ws_rgb_replay` at startup.
- `BrokenPipeError` during `/ws_rgb/frame_image` usually means the browser canceled a request; server now tolerates this.
- If replay cannot start, check:
  - `RUN_ID` exists under `runs/<run_id>/samples`
  - the selected sample has replay artifacts in `predict_result[*].replay_path/replay_v1_path`
  - port is free (`--port`)

