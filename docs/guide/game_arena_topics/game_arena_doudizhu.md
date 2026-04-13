# Doudizhu Game Arena Guide

English | [中文](game_arena_doudizhu_zh.md)

Doudizhu uses the GameKit phase-card runtime and the Arena Visual table scene. Use this topic for three-seat card flow, legal action text, chat metadata, and human-vs-LLM browser runs.

## 1. Canonical Files

| Use | File |
| --- | --- |
| Dummy headless smoke | `config/custom/doudizhu/doudizhu_dummy_gamekit.yaml` |
| Dummy visual smoke | `config/custom/doudizhu/doudizhu_dummy_visual_gamekit.yaml` |
| Local test LLM headless | `config/custom/doudizhu/doudizhu_llm_headless_gamekit.yaml` |
| Local test LLM visual | `config/custom/doudizhu/doudizhu_llm_visual_gamekit.yaml` |
| OpenAI LLM headless | `config/custom/doudizhu/doudizhu_llm_headless_openai_gamekit.yaml` |
| OpenAI LLM visual | `config/custom/doudizhu/doudizhu_llm_visual_openai_gamekit.yaml` |
| Human vs local test LLM visual | `config/custom/doudizhu/doudizhu_human_visual_gamekit.yaml` |
| Human vs OpenAI visual | `config/custom/doudizhu/doudizhu_human_visual_openai_gamekit.yaml` |
| Manual acceptance local test LLM | `config/custom/doudizhu/doudizhu_human_visual_acceptance_gamekit.yaml` |
| Manual acceptance OpenAI | `config/custom/doudizhu/doudizhu_human_visual_acceptance_openai_gamekit.yaml` |
| Fixture data | `tests/data/Test_Doudizhu_GameKit.jsonl` |

## 2. Prerequisites

Doudizhu uses `rlcard`, which is listed in `requirements.txt`; no external card-game server is required.

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
bash scripts/run/arenas/doudizhu/run.sh --mode dummy_visual --max-samples 1
```

```bash
OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>" bash scripts/run/arenas/doudizhu/run.sh --mode llm_visual_openai --max-samples 1
```

```bash
OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>" bash scripts/run/arenas/doudizhu/run.sh --mode human_visual_openai --max-samples 1
```

Use `--max-samples 0` with any OpenAI config to validate config loading without executing a sample.

## 4. Mode and Config Mapping

The `scripts/run/arenas/doudizhu/run.sh` script selects Python from `--python-bin`, then `PYTHON_BIN`, then the active virtualenv/conda env, then `python`/`python3`. It prints the resolved Python, mode, config, output directory, and run id before calling `run.py`.

| Entry | Config | Use |
| --- | --- | --- |
| `dummy` | `config/custom/doudizhu/doudizhu_dummy_gamekit.yaml` | Headless deterministic smoke. |
| `dummy_visual` | `config/custom/doudizhu/doudizhu_dummy_visual_gamekit.yaml` | Arena Visual table smoke. |
| `llm_headless` | `config/custom/doudizhu/doudizhu_llm_headless_gamekit.yaml` | Local test LLM landlord vs dummy farmers. |
| `llm_visual` | `config/custom/doudizhu/doudizhu_llm_visual_gamekit.yaml` | Local test LLM visual run. |
| `llm_headless_openai` | `config/custom/doudizhu/doudizhu_llm_headless_openai_gamekit.yaml` | OpenAI LLM without browser. |
| `llm_visual_openai` | `config/custom/doudizhu/doudizhu_llm_visual_openai_gamekit.yaml` | OpenAI LLM with browser. |
| `human_visual` | `config/custom/doudizhu/doudizhu_human_visual_gamekit.yaml` | Human landlord vs local test LLM farmers. |
| `human_visual_openai` | `config/custom/doudizhu/doudizhu_human_visual_openai_gamekit.yaml` | Human landlord vs OpenAI farmers. |
| `human_acceptance` | `config/custom/doudizhu/doudizhu_human_visual_acceptance_gamekit.yaml` | Longer manual acceptance run. |
| `human_acceptance_openai` | `config/custom/doudizhu/doudizhu_human_visual_acceptance_openai_gamekit.yaml` | OpenAI manual acceptance run. |

`--config <path>` always overrides the script mode mapping when a script is available.

## 5. Browser Control

Visual configs use `visualizer.mode: arena_visual`. The browser opens a session URL shaped like:

```text
http://127.0.0.1:<visual_port>/sessions/<sample_id>?run_id=<run_id>
```

For the shared command deck, transport controls, utility rail, timeline, and replay states, see [Arena Visual Browser Control](game_arena_visual_control.md).

![Doudizhu Arena Visual table](../../assets/arena-visual-doudizhu-stage-20260409.png)

## 6. Human Input

Submit one exact legal action string from the table controls, or select by `action_index`, `move_index`, or `index`. Payload fields `action`, `move`, `selected_action`, `selected_move`, `value`, and `text` are accepted. `chat`, `message`, or `text_message` can travel with the action. The aliases `skip`, `fold`, and `no` resolve to `pass` when `pass` is legal.

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

- `ArenaObservation.context` identifies turn mode and step.
- `view.state` carries public/private card state and table UI data.
- `legal_actions.items` is the exact action string list accepted by the parser.
- `GameResult.move_log` records actor, action text, card payload, and chat metadata when present.

The built-in LLM player receives the sample messages plus one user turn derived from the current `ArenaObservation`: active player, view text, legal moves, and the instruction to return exactly one legal move. A returned move is wrapped as `ArenaAction` and applied by the GameKit environment.

## 9. Common Parameters

| Adjustment | Where |
| --- | --- |
| Environment preset | `env: classic_3p` or `classic_3p_real` |
| Start player | `runtime_overrides.start_player_id` |
| Chat cadence | `runtime_overrides.chat_mode` / `chat_every_n` |
| Illegal policy | `runtime_overrides.illegal_policy` |
| Live replay writes | `runtime_overrides.replay_live` |
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
| ROM or desktop/render error in this topic | Doudizhu needs `rlcard` only; ROM and realtime render dependencies are not part of this topic. |
