# PettingZoo Atari Game Arena Guide

English | [中文](game_arena_pettingzoo_atari_zh.md)

PettingZoo Atari uses the GameKit AEC environment runtime and the Arena Visual frame scene. The packaged topic focuses on Space Invaders with dummy, single-LLM, double-LLM, human, and low-latency visual paths.

## 1. Canonical Files

| Use | File |
| --- | --- |
| Dummy headless smoke | `config/custom/pettingzoo/space_invaders_dummy_gamekit.yaml` |
| Dummy visual smoke | `config/custom/pettingzoo/space_invaders_dummy_visual_gamekit.yaml` |
| Binary stream visual smoke | `config/custom/pettingzoo/space_invaders_dummy_visual_binary_stream_gamekit.yaml` |
| Low-latency visual smoke | `config/custom/pettingzoo/space_invaders_dummy_visual_low_latency_channel_gamekit.yaml` |
| Human visual | `config/custom/pettingzoo/space_invaders_human_visual_gamekit.yaml` |
| Local test LLM headless | `config/custom/pettingzoo/space_invaders_llm_headless_gamekit.yaml` |
| Local test LLM visual | `config/custom/pettingzoo/space_invaders_llm_visual_gamekit.yaml` |
| Local test double LLM visual | `config/custom/pettingzoo/space_invaders_double_llm_visual_gamekit.yaml` |
| Local test double LLM low-latency visual | `config/custom/pettingzoo/space_invaders_double_llm_visual_low_latency_channel_gamekit.yaml` |
| OpenAI LLM headless | `config/custom/pettingzoo/space_invaders_llm_headless_openai_gamekit.yaml` |
| OpenAI LLM visual | `config/custom/pettingzoo/space_invaders_llm_visual_openai_gamekit.yaml` |
| Double OpenAI LLM visual | `config/custom/pettingzoo/space_invaders_double_llm_visual_openai_gamekit.yaml` |
| Double OpenAI low-latency visual | `config/custom/pettingzoo/space_invaders_double_llm_visual_low_latency_channel_openai_gamekit.yaml` |

## 2. Prerequisites

The Atari path depends on `pettingzoo[atari]` and `shimmy[atari]`, both listed in `requirements.txt`. If your local ALE install has no Space Invaders ROM, import or install the required Atari ROM before running real backend modes. Dummy config loading can still be checked with `--max-samples 0`.

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
bash scripts/run/arenas/pettingzoo/run.sh --mode dummy_visual --max-samples 1
```

```bash
OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>" bash scripts/run/arenas/pettingzoo/run.sh --mode llm_visual_openai --max-samples 1
```

```bash
OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>" bash scripts/run/arenas/pettingzoo/run.sh --mode double_llm_visual_openai --max-samples 1
```

```bash
bash scripts/run/arenas/pettingzoo/run.sh --mode human_visual --max-samples 1
```

Use `--max-samples 0` with any OpenAI config to validate config loading without executing a sample.

## 4. Mode and Config Mapping

The `scripts/run/arenas/pettingzoo/run.sh` script selects Python from `--python-bin`, then `PYTHON_BIN`, then the active virtualenv/conda env, then `python`/`python3`. It prints the resolved Python, mode, config, output directory, and run id before calling `run.py`.

| Entry | Config | Use |
| --- | --- | --- |
| `dummy` | `config/custom/pettingzoo/space_invaders_dummy_gamekit.yaml` | Headless dummy cycle. |
| `dummy_visual` | `config/custom/pettingzoo/space_invaders_dummy_visual_gamekit.yaml` | `http_pull` Arena Visual frame smoke. |
| `binary_stream` | `config/custom/pettingzoo/space_invaders_dummy_visual_binary_stream_gamekit.yaml` | Binary stream media path smoke. |
| `low_latency` | `config/custom/pettingzoo/space_invaders_dummy_visual_low_latency_channel_gamekit.yaml` | Low-latency channel media path smoke. |
| `human_visual` | `config/custom/pettingzoo/space_invaders_human_visual_gamekit.yaml` | Human pilot vs dummy pilot. |
| `llm_headless` | `config/custom/pettingzoo/space_invaders_llm_headless_gamekit.yaml` | Local test LLM pilot without browser. |
| `llm_visual` | `config/custom/pettingzoo/space_invaders_llm_visual_gamekit.yaml` | Local test LLM pilot with browser. |
| `double_llm_visual` | `config/custom/pettingzoo/space_invaders_double_llm_visual_gamekit.yaml` | Two local test LLM pilots with browser. |
| `double_llm_low_latency` | `config/custom/pettingzoo/space_invaders_double_llm_visual_low_latency_channel_gamekit.yaml` | Two local test LLM pilots using low-latency transport. |
| `llm_headless_openai` | `config/custom/pettingzoo/space_invaders_llm_headless_openai_gamekit.yaml` | OpenAI pilot without browser. |
| `llm_visual_openai` | `config/custom/pettingzoo/space_invaders_llm_visual_openai_gamekit.yaml` | OpenAI pilot with browser. |
| `double_llm_visual_openai` | `config/custom/pettingzoo/space_invaders_double_llm_visual_openai_gamekit.yaml` | Two OpenAI pilots with browser. |
| `double_llm_low_latency_openai` | `config/custom/pettingzoo/space_invaders_double_llm_visual_low_latency_channel_openai_gamekit.yaml` | Two OpenAI pilots using low-latency transport. |

`--config <path>` always overrides the script mode mapping when a script is available.

## 5. Browser Control

Visual configs use `visualizer.mode: arena_visual`. The browser opens a session URL shaped like:

```text
http://127.0.0.1:<visual_port>/sessions/<sample_id>?run_id=<run_id>
```

For the shared command deck, transport controls, utility rail, timeline, and replay states, see [Arena Visual Browser Control](game_arena_visual_control.md).

![PettingZoo Space Invaders Arena Visual stage](../../assets/arena-visual-space-invaders-stage-20260409.png)

## 6. Human Input

The current visual plugin submits frame action controls when the decision window opens. The mapper accepts `action`, `move`, `selected_action`, `selected_move`, `value`, `text`, `action_id`, or an index field. Keyboard events are accepted only when a custom key map is supplied; the packaged Space Invaders visual path should be treated as action-control first.

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

- `ArenaObservation.metadata` carries active agent, reward, termination, truncation, and action-meaning facts.
- `view.media` and `arena_visual_session/v1/media/` carry frame references.
- `legal_actions.items` is the discrete action id or action meaning list used by the parser.
- `GameResult` summarizes move count, illegal count, final board text, and episode result.

The built-in LLM player receives the sample messages plus one user turn derived from the current `ArenaObservation`: active player, view text, legal moves, and the instruction to return exactly one legal move. A returned move is wrapped as `ArenaAction` and applied by the GameKit environment.

## 9. Common Parameters

| Adjustment | Where |
| --- | --- |
| Backend mode | `runtime_overrides.backend_mode` |
| Cycle budget | `runtime_overrides.max_cycles` |
| Action meanings | `runtime_overrides.use_action_meanings` |
| Raw observation payload | `runtime_overrides.include_raw_obs` |
| Live scene transport | `visualizer.live_scene_scheme` |
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
| ROM or ALE error | Install/import the Space Invaders ROM for your ALE/PettingZoo environment, then retry the same config. |
