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
