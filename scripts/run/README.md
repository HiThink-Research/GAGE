# Run Entrypoints

`scripts/run/` contains the canonical local launchers for GAGE. The wrappers under
`scripts/oneclick/` still exist for compatibility, but new docs and commands should
link here.

## Layout

- `appworld/`: AppWorld evaluation and connectivity checks.
- `backends/`: provider demos, backend smoke tests, and local templates.
- `arenas/`: Game Arena launchers for PettingZoo, Retro Mario, ViZDoom, Doudizhu, Mahjong, and Arena Visual artifact helpers.
- `common/`: shared shell helpers for workspace detection, Python detection, and run directories.

## Workspace Defaults

The shared helper `scripts/run/common/env.sh` resolves local paths in this order:

- `GAGE_WORKSPACE_ROOT`: explicit workspace root override.
- Parent workspace containing `env/.venv`, `env/scripts/run.env`, or `env/localenv`.
- The repository root as the final fallback.

Runtime outputs are not written back to the repo by default:

- `GAGE_RUNS_DIR`: overrides the run artifact directory.
- `gage_default_runs_dir`: defaults to `${GAGE_WORKSPACE_ROOT}/runs`.
- `GAGE_SCRIPT_STATE_DIR`: overrides generated helper state; defaults to `${GAGE_WORKSPACE_ROOT}/env/scripts/generated`.
- `PYTHON_BIN`: overrides the Python interpreter used by launchers.
- `VENV_PATH`, `VIRTUAL_ENV`, or `CONDA_PREFIX`: used by `gage_default_python` before falling back to `python` or `python3`.

For the local conda setup used by the Game Arena smoke tests:

```bash
${HOME}/miniconda3/bin/conda run -n gage-eval python -V
PYTHON_BIN=${HOME}/miniconda3/envs/gage-eval/bin/python \
  bash scripts/run/arenas/pettingzoo/run.sh --mode llm_visual --max-samples 1
```

## Game Arena Launchers

The current Game Arena launchers map CLI modes to `GameKit + arena_visual` YAML files.

| Game | Command | Current modes |
| --- | --- | --- |
| PettingZoo Space Invaders | `bash scripts/run/arenas/pettingzoo/run.sh --mode llm_visual_openai` | `dummy`, `dummy_visual`, `binary_stream`, `low_latency`, `llm_headless`, `llm_visual`, `llm_headless_openai`, `llm_visual_openai`, `human_visual`, `double_llm_visual`, `double_llm_visual_openai`, `double_llm_low_latency`, `double_llm_low_latency_openai` |
| Retro Mario | `bash scripts/run/arenas/retro_mario/run.sh --mode llm_visual_openai` | `dummy`, `llm_headless`, `llm_visual`, `llm_headless_openai`, `llm_visual_openai`, `human_visual` |
| ViZDoom | `bash scripts/run/arenas/vizdoom/run.sh --mode llm_visual_openai` | `dummy`, `llm_headless`, `llm_visual`, `llm_headless_openai`, `llm_visual_openai`, `human_visual` |
| Mahjong | `bash scripts/run/arenas/mahjong/run.sh --mode llm_visual_openai` | `dummy`, `dummy_visual`, `llm_headless`, `llm_visual`, `llm_headless_openai`, `llm_visual_openai`, `human_visual`, `human_visual_openai`, `human_acceptance`, `human_acceptance_openai` |
| Doudizhu | `bash scripts/run/arenas/doudizhu/run.sh --mode llm_visual_openai` | `dummy`, `dummy_visual`, `llm_headless`, `llm_visual`, `llm_headless_openai`, `llm_visual_openai`, `human_visual`, `human_visual_openai`, `human_acceptance`, `human_acceptance_openai` |
| Arena Visual artifacts | `bash scripts/run/arenas/replay/run_and_open.sh --run-id <run_id>` | Opens the current Arena Visual session artifact for a completed visual run. |

Examples:

```bash
OPENAI_API_KEY='<your-token-here>' bash scripts/run/arenas/doudizhu/run.sh --mode llm_visual_openai --max-samples 1
OPENAI_API_KEY='<your-token-here>' bash scripts/run/arenas/mahjong/run.sh --mode human_visual_openai
bash scripts/run/arenas/vizdoom/run.sh --mode human_visual
OPENAI_API_KEY='<your-token-here>' bash scripts/run/arenas/retro_mario/run.sh --mode llm_visual_openai --max-samples 1
```

## Other Common Commands

```bash
bash scripts/run/prepare_env.sh
bash scripts/run/backends/demos/run_demo_echo.sh
HF_PROVIDER=together HF_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct HF_API_TOKEN='<your-token-here>' \
  bash scripts/run/backends/demos/run_multi_provider_http_demo.sh
OPENAI_API_KEY='<your-token-here>' bash scripts/run/backends/demos/run_kimi_demo.sh
```

Run results usually contain `summary.json`, `events.jsonl`, `samples/*.json`, or
`samples/*.jsonl` under `${GAGE_RUNS_DIR:-${GAGE_WORKSPACE_ROOT}/runs}/<run_id>/`.
