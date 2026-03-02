# PettingZoo Atari User Guide

This document is the standard guide for running **PettingZoo Atari** games within the GAGE framework.
Whether you want to run AI battles, test environment configurations, or replay recorded games, this manual covers it all.

## 1. Quick Start

We recommend the **"Run & Replay"** automation workflow.
This allows you to inspect AI decisions and watch matches directly in the `ws_rgb` web viewer.

### Example: Running Space Invaders

Copy and execute the following commands:

```bash
# 1. Set the game name (supports 22 games, see list below)
export GAME="space_invaders"
export RUN_ID="pz_${GAME}_auto_$(date +%s)"
# Force game_log to be inlined into sample.json (compatible with ws_rgb_replay)
export GAGE_EVAL_GAME_LOG_INLINE_LIMIT=-1
export GAGE_EVAL_GAME_LOG_INLINE_BYTES=0

# 2. Run AI simulation
python run.py \
  --config config/custom/pettingzoo/${GAME}_ai.yaml \
  --output-dir runs \
  --run-id "$RUN_ID"

# 3. Start web replay service (new ws_rgb flow, auto-open by default)
SAMPLE_JSON="$(find runs/$RUN_ID/samples -type f -name "*.json" | head -n 1)"
[ -n "$SAMPLE_JSON" ] || { echo "No sample file found under runs/$RUN_ID/samples"; exit 1; }
PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo \
  --auto-open 1

# 4. Manual fallback if browser did not open automatically
echo "http://127.0.0.1:5800/ws_rgb/viewer"
```
*(Note: `--auto-open 1` relies on desktop/browser availability. In headless environments, open the URL manually.)*

---

## 2. Supported Games

GAGE currently supports the following **22 Two-Player Atari Games**:

| Game ID | Description |
| :--- | :--- |
| `basketball_pong` | Basketball Pong |
| `boxing` | Boxing (Classic) |
| `combat_plane` | Combat Plane |
| `combat_tank` | Combat Tank |
| `double_dunk` | Double Dunk |
| `entombed_competitive` | Entombed (Competitive) |
| `entombed_cooperative` | Entombed (Cooperative) |
| `flag_capture` | Flag Capture |
| `foozpong` | Foozpong |
| `ice_hockey` | Ice Hockey |
| `joust` | Joust |
| `mario_bros` | Mario Bros. |
| `maze_craze` | Maze Craze |
| `othello` | Othello (Reversi) |
| `pong` | Pong |
| **`space_invaders`** | **Space Invaders (Recommended Demo)** |
| `space_war` | Space War |
| `surround` | Surround (Snake-like) |
| `tennis` | Tennis |
| `video_checkers` | Video Checkers |
| `volleyball_pong` | Volleyball Pong |
| `wizard_of_wor` | Wizard of Wor |

To run other games, simply change the environment variable:
`export GAME="boxing"`

> [!TIP]
> For a complete list of start commands (Dummy and AI modes), please refer to:
> 🔗 [**PettingZoo Atari Run Commands**](./pettingzoo_atari_run_commands.md)

---

## 3. Running Modes

We offer two running configurations: **Dummy** (for testing) and **AI** (for actual battles).

### 3.1 AI Mode (Requires API Key)

*   **Config Path**: `config/custom/pettingzoo/<game>_ai.yaml`
*   **Purpose**: Call OpenAI API (e.g., gpt-4o, o1-preview) for real battles.
*   **Command**:
    ```bash
    export OPENAI_API_KEY="sk-..."
    python run.py --config config/custom/pettingzoo/boxing_ai.yaml ...
    ```

### 3.2 Dummy Mode (No API Key Required)

*   **Config Path**: `config/custom/pettingzoo/<game>_dummy.yaml`
*   **Purpose**: Quickly test environment installation, verify rendering, and debug scripts.
*   **Policy**: Uses a **Random Policy** (random legal moves) to ensure the game runs continuously.
*   **Command**:
    ```bash
    python run.py --config config/custom/pettingzoo/boxing_dummy.yaml ...
    ```

---

## 4. Web Replay Tool (ws_rgb Viewer)

`scripts/replay_pettingzoo.py` is a legacy replay path and is no longer recommended.
Use the unified `ws_rgb` web-based flow instead.

### 4.1 Post-run replay (from existing artifacts)

```bash
RUN_ID=<your_run_id>
# If you hit sample_game_log_missing, rerun run.py with these two env vars first
export GAGE_EVAL_GAME_LOG_INLINE_LIMIT=-1
export GAGE_EVAL_GAME_LOG_INLINE_BYTES=0
SAMPLE_JSON="$(find runs/$RUN_ID/samples -type f -name "*.json" | head -n 1)"
[ -n "$SAMPLE_JSON" ] || { echo "No sample file found under runs/$RUN_ID/samples"; exit 1; }

PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo \
  --auto-open 1
```

Viewer URL:
`http://127.0.0.1:5800/ws_rgb/viewer`

### 4.2 One-click run with auto-open (live view)

```bash
AUTO_OPEN=1 \
CONFIG=config/custom/pettingzoo/pong_dummy_ws_rgb.yaml \
RUN_ID="pz_pong_ws_$(date +%Y%m%d_%H%M%S)" \
bash scripts/oneclick/run_pettingzoo_ws_rgb_viewer.sh
```

---

## 5. Troubleshooting (FAQ)

### Q1: Why doesn't the ship/character move at the beginning of the replay?
**A**: This is the **Warm-up Phase**.
Atari games often have a few seconds of startup animation (like flashing "INSERT COINS" or "PLAYER 1"). During this time, the game engine locks all inputs. Although AI tries to act, its inputs are ignored. Once the screen flashes and the game officially starts, the actions will take effect.

### Q2: Since the AI is playing blindly, how does it know what to do?
**A**: The AI sees a **Text Observation**, for example:
`Status: Legal moves: [FIRE, LEFT, RIGHT]`.
It doesn't need pixel input; it reasons based on the list of legal moves and history.

### Q3: How to change the game duration?
**A**: Modify `max_cycles` and `max_turns` in `config/custom/pettingzoo/<game>_ai.yaml`. Currently defaults to 3000 frames (approx. 50 seconds).
