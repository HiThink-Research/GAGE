<div align="center">

# 📐 GAGE: General AI evaluation and Gauge Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Code Style](https://img.shields.io/badge/code%20style-google-blueviolet)](https://google.github.io/styleguide/pyguide.html)
[![License](https://img.shields.io/badge/license-TBD-lightgrey)]()
[![Status](https://img.shields.io/badge/Status-Alpha-orange)]()

**English** · [中文](README_zh.md)

📧 **Contact:** [zhangrongjunchen@myhexin.com](mailto:zhangrongjunchen@myhexin.com)

[Overview](docs/guide/framework_overview.md) · [Sample Schema](docs/guide/sample.md) · [Game Arena](docs/guide/game_arena.md) · [Arena Visual Control](docs/guide/game_arena_topics/game_arena_visual_control.md) · [Agent Eval](docs/guide/agent_evaluation.md) · [Benchmark](docs/guide/benchmark.md) · [Contributing](CONTRIBUTING.md) · [Standards](AGENTS.md)

</div>

---

**GAGE** is a unified, extensible evaluation framework for large language models, multimodal models, audio models, diffusion models, agents, and game environments. It provides one evaluation engine for datasets, model backends, metrics, arena runtimes, structured outputs, and replayable artifacts.

## Game Arena Showcase

<p align="center"><img src="docs/assets/gamearena-gomoku-20260413.gif" width="33.3333%" alt="Gomoku GameArena demo"><!--
--><img src="docs/assets/gamearena-doudizhu-20260413.gif" width="33.3333%" alt="Doudizhu GameArena demo"><!--
--><img src="docs/assets/gamearena-mahjong-20260413.gif" width="33.3333%" alt="Mahjong GameArena demo"></p>

<p align="center">
  <img src="docs/assets/space-invaders-game.gif" width="33.3333%" alt="Space Invaders demo">
  <img src="docs/assets/mario-game.gif" width="33.3333%" alt="Mario demo">
  <img src="docs/assets/vizdoom-game.gif" width="32%" alt="VizDoom demo">
</p>

## Why GAGE?

- **Fast evaluation engine**: Run local smoke tests, model-backed jobs, and larger benchmark batches through the same pipeline shape.
- **Unified evaluation surface**: Datasets, backends, role adapters, metrics, and output contracts are configured instead of hand-wired per benchmark.
- **Game and agent sandboxing**: Game Arena, AppWorld, SWE-bench-style agent tasks, GUI interaction, and tool-augmented workflows share the same run/output model.
- **Replayable GameKit runtime**: Gomoku, Tic-Tac-Toe, Doudizhu, Mahjong, PettingZoo Space Invaders, Retro Mario, and ViZDoom now emit structured arena traces plus `arena_visual` sessions.
- **Operational visibility**: Runs write `summary.json`, sample outputs, logs, and visual artifacts so failures can be inspected after the fact.

## Design Overview

> Core design philosophy: everything is a step, everything is configurable.

### Architecture Design

![End-to-end flow](docs/assets/process-arch.png)

### Orchestration Design

![Step view](docs/assets/step-chain.png)

### Game Arena Design

![GameArena runtime core design](docs/assets/game-arena-runtime-core-design-20260413.png)

## Quick Start

### 1. Installation

```bash
# If you are in the mono-repo root:
cd gage-eval-main

# Python 3.10+ recommended
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For Game Arena LLM configs, use the `*_openai_gamekit.yaml` variants and export `OPENAI_API_KEY`. The model defaults to `gpt-5.4`; set `GAGE_GAME_ARENA_LLM_MODEL` to override it, or set `OPENAI_API_BASE` for an OpenAI-compatible endpoint.

### 2. Run a Basic Demo

```bash
python run.py \
  --config config/run_configs/demo_echo_run_1.yaml \
  --output-dir runs \
  --run-id demo_echo
```

### 3. View Reports

Default output structure:

```text
runs/<run_id>/
  events.jsonl
  samples.jsonl
  summary.json
  samples/
    <task_id>/
      <sample_id>.json
```

## Advanced Configurations

| Scenario | Config Example | Description |
| :--- | :--- | :--- |
| **GameArena Human-vs-AI** | `config/custom/doudizhu/doudizhu_human_visual_gamekit.yaml` | Browser-controlled Doudizhu match against LLM players |
| **GameArena Pure Human Control** | `config/custom/retro_mario/retro_mario_human_visual_gamekit.yaml` | Browser-controlled real-time Retro Mario session |
| **Agent Evaluation** | `config/custom/appworld/appworld_official_jsonl.yaml` | AppWorld sandbox evaluation |
| **Code Gen** | `config/custom/swebench_pro/swebench_pro_smoke_agent.yaml` | SWE-bench style smoke run; Docker required |
| **Text** | `config/custom/aime24/aime2024_chat.yaml` | AIME, GPQA, Math500, and related text benchmarks |
| **Multimodal** | `config/custom/mathvista/chat.yaml` | MathVista and related multimodal benchmarks |
| **LLM Judge** | `config/custom/examples/single_task_local_judge_qwen.yaml` | Local LLM judge example |

## Roadmap

- **Agent evaluation**: Add stronger native agent benchmarking support with trajectory scoring and safety checks.
- **Game Arena expansion**: Grow the GameKit catalog and keep browser control, replay, and output contracts consistent.
- **Gage-Client**: Add a client tool for configuration management, failure diagnostics, and benchmark onboarding.
- **Distributed inference**: Support multi-node task sharding and load balancing for large runs.
- **Benchmark expansion**: Continue adding benchmark configs, metrics, and troubleshooting guidance.

## Status

This project is in internal validation; APIs, configs, and docs may change rapidly.
