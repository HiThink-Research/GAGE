# Doudizhu Showdown Guide

English | [中文](doudizhu_showdown_zh.md)

This document is the canonical Doudizhu Showdown entry in this repository. It consolidates one-click startup, replay flow, result structure, prompt assembly, and common troubleshooting notes.

## Quick Start

Prerequisites:
- Node.js + npm
- First time setup: `cd frontend/rlcard-showdown && npm install --legacy-peer-deps`
- Set API key: `OPENAI_API_KEY` (or `LITELLM_API_KEY`)
- Ensure `PYTHON_BIN` in `run_doudizhu_showdown` points to the correct environment

One-click Start:
```bash
scripts/oneclick/run_doudizhu_showdown.sh       # 3 AI players
scripts/oneclick/run_doudizhu_showdown_human.sh # Human vs AI
```

After startup, the script prints:
```
[oneclick] replay url: http://127.0.0.1:<port>/replay/doudizhu?...
```
If the browser does not open automatically, open this URL manually.

Common Environment Variables:
- `RUN_ID`: Run identifier (defaults to timestamp)
- `OUTPUT_DIR`: Output directory (defaults to `./runs`)
- `FRONTEND_PORT` / `REPLAY_PORT`: Frontend and replay server ports (auto-increment if occupied)
- `AUTO_OPEN=0`: Disable auto-opening the browser
- `FRONTEND_DIR`: Frontend directory (defaults to `frontend/rlcard-showdown`)

## Script Flow Explanation

(Corresponding to `scripts/oneclick/run_doudizhu_showdown.sh`)

The main flow of the script:
1. Parses project root and Python path, then reads the default config `config/custom/doudizhu_litellm_local.yaml`.
2. Checks whether `OPENAI_API_KEY` / `LITELLM_API_KEY`, Node.js/npm, and frontend dependencies are ready.
3. Automatically selects free ports and starts the replay server.
4. Starts the frontend with `npm run start` and tries to open the replay page.
5. Runs `run.py` to execute the game and prints the replay link.

Built-in defaults (can be changed in the script if needed):
- `PYTHON_BIN`: Python interpreter path (defaults to the project venv)
- `CFG`: Run config path (defaults to `config/custom/doudizhu_litellm_local.yaml`)
- `SAMPLE_ID`: Replay sample ID (defaults to `doudizhu_litellm_0001`)

## Replay and Output

Replay file path (default):
```
runs/<run_id>/replays/doudizhu_replay_<sample_id>.json
```

Replays are served by the replay server and read by the frontend through the `replay_url` parameter.
If you need to inspect the replay directly, read the JSON file from the path above.

## GameResult and Result Flow

The GAGE arena step produces a `GameResult` after the game ends and writes it into the sample `predict_result` for downstream consumers (`judge` / `auto_eval`):
- Write location: `src/gage_eval/evaluation/task_planner.py` -> `append_predict_result()`
- Format source: `src/gage_eval/role/adapters/arena.py::_format_result()`

Standard fields (consistent with Gomoku):
- `winner`: Winner player_id (or `null`)
- `result`: `"win" | "draw" | "loss"`
- `reason`: Terminal reason such as `terminal` / `illegal_move` / `max_turns`
- `move_count` / `illegal_move_count`
- `final_board`: Final board snapshot (text)
- `game_log`: Per-turn action log
- `rule_profile` / `win_direction` / `line_length`

Gomoku `game_log` structure (example):
```json
{"index": 1, "player": "player_0", "coord": "H8", "row": 7, "col": 7}
```

Doudizhu `game_log` structure (`doudizhu_arena_v1`):
```json
{
  "index": 1,
  "player": "player_0",
  "action_id": 123,
  "action_text": "333444",
  "action_cards": ["S3","H3","D3","C4","S4","H4"],
  "chat": "Raise first",
  "timestamp_ms": 1730000000000
}
```

Notes:
- Doudizhu `final_board` is a text snapshot from `_snapshot_board()`, including Public/Private State, Legal Moves preview, and Chat Log.
- If downstream logic only needs the winner/result, read `winner` / `result` directly. Use `game_log` for replay details.

## Execution Logic (Key Flow)

### 1) Dataset Input (System Prompt)

Location: `tests/data/Test_Doudizhu_LiteLLM.jsonl`

Core fields:
- `messages`: System prompt controlling AI persona, tone, and output format
- `metadata.player_ids`: Player IDs such as `player_0/1/2`
- `metadata.start_player_id`: Starting player

### 2) Runtime Observation (Per-turn Context)

The backend constructs an observation every turn and passes it to the LLM:
- File: `src/gage_eval/role/arena/games/doudizhu/env.py`
- Entry: `observe()` and `_format_board_text()`

Observation includes:
- `Public State` / `Private State` (JSON)
- `Legal Moves (preview)` (truncated by default)
- `Chat Log` (when chat is enabled)
- `UI_STATE_JSON` (structured state required by frontend rendering)

`metadata` also carries:
- `player_ids` / `player_names`
- `public_state` / `private_state`
- `chat_log` / `chat_mode`

### 3) LLM Prompt Assembly

Location: `src/gage_eval/role/arena/players/llm_player.py`

Assembly order:
1. Dataset `messages` (system prompt)
2. Runtime observation (board_text + legal moves + instructions)

When `chat_mode` is `ai-only` / `all`, the model is asked to output:
```json
{"action": "<action>", "chat": "<short line>"}
```

Example (actual context seen by the model):
```text
[system]
Start Doudizhu. Output exactly one legal action string such as 'pass' or card ranks like '33'. You may also output JSON: {"action": "pass", "chat": "..."}.

[user]
Active player: Player 0 (player_0)
Opponent last move: pass

Current State:
Public State:
{"round":2,"landlord_id":"player_0","last_move":"pass",...}

Private State:
{"hand":["S3","H3","D3","C4","S4","BJ","RJ",...],...}

Legal Moves (preview): pass, 33, 44, 34567, ...

Chat Log:
[{"player_id":"player_1","text":"I'll pass."}]...

UI_STATE_JSON:
{"player_ids":["player_0","player_1","player_2"],"hands":[...],"latest_actions":[...],...}

Status:
- Legal moves (preview): pass, 33, 44, 34567, ...

Instructions:
- Choose exactly one legal action string from the legal moves.
- Include a short table-talk line every turn.
- Output JSON: {"action": "<action>", "chat": "<short line>"}
```

### 4) Replay Writing and Frontend Reading

Replay is written by `doudizhu_arena_v1`:
- File: `src/gage_eval/role/arena/games/doudizhu/env.py`
- Output path: `runs/<run_id>/replays/doudizhu_replay_<sample_id>.json`

Replay service:
- File: `src/gage_eval/tools/replay_server.py`
- URL: `/tournament/replay?run_id=...&sample_id=...`

Frontend reader:
- File: `frontend/rlcard-showdown/src/view/ReplayView/DoudizhuReplayView.js`
- URL: `/replay/doudizhu?run_id=...&sample_id=...&live=1`

## AI Persona and Chat Configuration

### 1) System Prompt (Main Entry for Persona and Style)

The current version of `doudizhu_arena_v1` does not read the `ai_persona` field.
AI persona and style are primarily controlled through the dataset system prompt.

Edit file:
`tests/data/Test_Doudizhu_LiteLLM.jsonl`

Example (key structure only):
```json
{
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are a calm, analytical Doudizhu player. Keep chat short and witty."
        }
      ]
    }
  ]
}
```

Notes:
- All players share the same `messages`.
- To give different players different personas, extend the dataset or modify player prompt injection.

### 2) Chat Toggle and Frequency

Configuration location:
`config/custom/doudizhu_litellm_local.yaml`

Example:
```yaml
role_adapters:
  - adapter_id: doudizhu_arena
    role_type: arena
    params:
      environment:
        impl: doudizhu_arena_v1
        chat_mode: ai-only   # off | ai-only | all
        chat_every_n: 2      # Record chat every N steps
```

### 3) Sampling Parameters (Tone and Randomness)

You can set per-player sampling parameters such as `temperature`:
```yaml
players:
  - player_id: player_0
    type: backend
    ref: doudizhu_player_0
    sampling_params:
      temperature: 0.7
```

## Manual Start Commands

If you need to start the stack manually, split it into three steps:

1) Start the replay server:
```bash
PYTHONPATH=src python -m gage_eval.tools.replay_server --port 8000 --replay-dir ./runs
```

2) Start the frontend:
```bash
cd frontend/rlcard-showdown
REACT_APP_GAGE_API_URL="http://127.0.0.1:8000" NODE_OPTIONS="--openssl-legacy-provider" npm run start
```

3) Run backend inference:
```bash
python run.py --config config/custom/doudizhu_litellm_local.yaml --output-dir runs --run-id doudizhu_showdown_local
```

## FAQ

- Browser cannot open the page (`ERR_CONNECTION_REFUSED`)
  Usually the frontend failed to start or the port is occupied. Confirm the port printed by the script and open the corresponding URL manually.

- Node reports `ERR_OSSL_EVP_UNSUPPORTED`
  Use `NODE_OPTIONS=--openssl-legacy-provider` (the one-click script already sets it).
