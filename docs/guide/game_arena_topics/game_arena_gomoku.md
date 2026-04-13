# Gomoku Game Arena Guide

English | [中文](game_arena_gomoku_zh.md)

Gomoku uses the current GameKit board runtime and the unified Arena Visual browser page. It is the best first topic for checking turn-based board flow, legal coordinate actions, and visual session artifacts.

## 1. Canonical Files

| Use | File |
| --- | --- |
| Dummy headless smoke | `config/custom/gomoku/gomoku_dummy_gamekit.yaml` |
| Dummy visual smoke | `config/custom/gomoku/gomoku_dummy_visual_gamekit.yaml` |
| Local test LLM headless | `config/custom/gomoku/gomoku_llm_headless_gamekit.yaml` |
| Local test LLM visual | `config/custom/gomoku/gomoku_llm_visual_gamekit.yaml` |
| OpenAI LLM headless | `config/custom/gomoku/gomoku_llm_headless_openai_gamekit.yaml` |
| OpenAI LLM visual | `config/custom/gomoku/gomoku_llm_visual_openai_gamekit.yaml` |
| Human vs local test LLM visual | `config/custom/gomoku/gomoku_human_visual_gamekit.yaml` |
| Human vs OpenAI visual | `config/custom/gomoku/gomoku_human_visual_openai_gamekit.yaml` |
| 15x15 human vs local test LLM visual | `config/custom/gomoku/gomoku_human_visual_15x15_gamekit.yaml` |
| 15x15 human vs OpenAI visual | `config/custom/gomoku/gomoku_human_visual_15x15_openai_gamekit.yaml` |
| Fixture data | `tests/data/Test_Gomoku.jsonl` |

## 2. Prerequisites

Gomoku itself uses in-repo GameKit code and fixture JSONL data; no external game asset is required.

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
# Optional: defaults to gpt-5.4.
export GAGE_GAME_ARENA_LLM_MODEL="gpt-5.4"
# Optional: OpenAI-compatible service override.
export OPENAI_API_BASE="https://api.openai.com/v1"
```

For the official OpenAI API, keep the default endpoint or leave `OPENAI_API_BASE` unset. For an open-source model served through an OpenAI-compatible API, set `OPENAI_API_BASE` and `GAGE_GAME_ARENA_LLM_MODEL`; no YAML backend edit is required.

Normal runs do not require Node/npm. Node/npm is only needed when developing or rebuilding `frontend/arena-visual`; see [Arena Visual Browser Control](game_arena_visual_control.md#2-backend-and-frontend-assets).

## 3. Quick Runs

Run commands from the repository root after activating the project Python environment.

```bash
python run.py --config config/custom/gomoku/gomoku_dummy_gamekit.yaml --max-samples 1
```

```bash
python run.py --config config/custom/gomoku/gomoku_dummy_visual_gamekit.yaml --max-samples 1
```

```bash
python run.py --config config/custom/gomoku/gomoku_llm_visual_openai_gamekit.yaml --max-samples 1
```

```bash
python run.py --config config/custom/gomoku/gomoku_human_visual_openai_gamekit.yaml --max-samples 1
```

Use `--max-samples 0` with any OpenAI config to validate config loading without executing a sample.

## 4. Mode and Config Mapping

Gomoku currently has no dedicated arena `run.sh`; run these configs directly with `python run.py --config`.

| Entry | Config | Use |
| --- | --- | --- |
| direct dummy | `config/custom/gomoku/gomoku_dummy_gamekit.yaml` | Headless deterministic smoke. |
| direct dummy visual | `config/custom/gomoku/gomoku_dummy_visual_gamekit.yaml` | Arena Visual board smoke. |
| direct llm headless openai | `config/custom/gomoku/gomoku_llm_headless_openai_gamekit.yaml` | OpenAI LLM black vs dummy white without browser. |
| direct llm visual openai | `config/custom/gomoku/gomoku_llm_visual_openai_gamekit.yaml` | OpenAI LLM black vs dummy white with browser. |
| direct human visual openai | `config/custom/gomoku/gomoku_human_visual_openai_gamekit.yaml` | Human black vs OpenAI white. |
| direct human visual 15x15 openai | `config/custom/gomoku/gomoku_human_visual_15x15_openai_gamekit.yaml` | 15x15 human browser variant. |

`--config <path>` always overrides the script mode mapping when a script is available.

## 5. Browser Control

Visual configs use `visualizer.mode: arena_visual`. The browser opens a session URL shaped like:

```text
http://127.0.0.1:<visual_port>/sessions/<sample_id>?run_id=<run_id>
```

For the shared command deck, transport controls, utility rail, timeline, and replay states, see [Arena Visual Browser Control](game_arena_visual_control.md).

![Gomoku Arena Visual stage](../../assets/arena-visual-gomoku-stage-20260409.png)

## 6. Human Input

Click a highlighted intersection in the board stage, or submit a coordinate from the legal move list such as `A1`. The input mapper accepts `action`, `move`, `coord`, `selected_action`, `selected_move`, `selected_coord`, `value`, or `text`; `action_index`, `move_index`, or `index` can select from the current legal move list. Illegal coordinates are rejected before they reach the environment.

## 7. Output and Replay Artifacts

Visual runs write both evaluation output and replayable Arena Visual artifacts:

```text
runs/<run_id>/
  summary.json
  samples.jsonl
  replays/<sample_id>/
    replay.json
    events.jsonl
    arena_visual_session/v1/
      manifest.json
      timeline.jsonl
      scenes/
      media/
```

`sample.predict_result[0].arena_trace` contains per-step actions, legality, timing, retries, and scheduler metadata. `sample.predict_result[0].game_arena` contains the terminal footer. When visualization is enabled, `artifacts.visual_session_ref` points at the `arena_visual_session/v1/manifest.json` sidecar. Finished runs replay from the same Arena Visual session artifacts.

## 8. Runtime Contracts

The shared runtime contract is documented in [Arena Visual Browser Control](game_arena_visual_control.md#3-runtime-contracts). For this game, check these fields first:

- `ArenaObservation.view.text` is the board view used by the LLM prompt and visual panel.
- `legal_actions.items` is the authoritative coordinate list.
- `GameResult.move_log` records coordinate, row, column, and actor for each move.

The built-in LLM player receives the sample messages plus one user turn derived from the current `ArenaObservation`: active player, view text, legal moves, and the instruction to return exactly one legal move. A returned move is wrapped as `ArenaAction` and applied by the GameKit environment.

## 9. Common Parameters

| Adjustment | Where |
| --- | --- |
| Board size | `runtime_overrides.board_size` |
| Win length | `runtime_overrides.win_len` |
| Coordinate scheme | `runtime_overrides.coord_scheme` |
| Observation image export | `runtime_overrides.obs_image` |
| Human input route | `human_input` |
| Browser port | `visualizer.port` |

## 10. Troubleshooting

| Symptom | Check |
| --- | --- |
| OpenAI config fails before runtime starts | Export `OPENAI_API_KEY` before using any `*_openai_gamekit.yaml` config. |
| Wrong model is used | Set `GAGE_GAME_ARENA_LLM_MODEL` before launch, or unset it to use the documented default. |
| Dependency import fails | Run `pip install -r requirements.txt` in the same Python environment used by `run.py` or the arena script. |
| ROM or desktop/render error appears | This game-specific note is listed below; make sure you are not using a frame-game config when running a board or table game. |
| Browser stays loading | Check the printed session URL, `visualizer.port`, and `runs/<run_id>/replays/<sample_id>/arena_visual_session/v1/manifest.json`. |
| Port is already in use | Change `visualizer.port` in the config or pass a different `--run-id`/output directory to avoid stale session confusion. |
| Human input is ignored | Confirm `human_input.enabled: true`, the browser URL includes the current `run_id`, and the active actor is the human player. |
| LLM returns an illegal action | Use the legal move list in the browser or trace panel; the built-in LLM driver falls back according to the player fallback policy when configured. |
| ROM or desktop/render error in this topic | Gomoku does not need ROMs or a render backend; verify the config path points under `config/custom/gomoku/`. |
