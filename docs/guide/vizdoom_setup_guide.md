# ViZDoom Setup Guide

English | [中文](vizdoom_setup_guide_zh.md)

This guide explains how to install the ViZDoom runtime environment and start the game in the current project.

## 1. Scope

This guide is for the current local ViZDoom integration in `gage-eval`.

Current behavior:

- Human modes use a local `pygame` input window
- Replay can be viewed through `ws_rgb`
- Human local play requires a desktop environment

## 2. Prerequisites

- Python `3.10+`
- Conda is recommended
- Run commands from the repository root

If you use Human modes, your machine must support local GUI windows.

## 3. Install Environment

### 3.1 Activate Conda Environment

If you already have the project environment:

```bash
conda activate GAGE
```

If you need to create one:

```bash
conda create -n GAGE python=3.10 -y
conda activate GAGE
```

### 3.2 Install Project Dependencies

```bash
pip install -r requirements.txt
```

### 3.3 Verify Installation

```bash
python - <<'PY'
import vizdoom
import pygame
print("vizdoom ok")
print("pygame ok")
PY
```

## 4. Repository Root

Before running the game:

```bash
cd /Users/zzzck/PyProject/GAGE
```

## 5. API Key Setup

The following modes require an API key:

- `human_vs_llm`
- `llm_vs_llm`
- `ai_vs_ai`
- `agent_vs_llm`
- `human_vs_llm_record`

Set one of:

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
```

or

```bash
export LITELLM_API_KEY="<YOUR_KEY>"
```

Modes that do not require a key:

- `human_vs_dummy`
- `human_solo`

## 6. Startup Commands

All commands below assume the repository root as the working directory.

### 6.1 Human vs Dummy

Recommended first validation run:

```bash
bash scripts/oneclick/run_vizdoom_human_vs_dummy.sh
```

### 6.2 Human Solo

```bash
bash scripts/oneclick/run_vizdoom_human_solo.sh
```

### 6.3 Human vs LLM

```bash
bash scripts/oneclick/run_vizdoom_human_vs_llm.sh
```

### 6.4 Human vs LLM with Record Scheduler

```bash
bash scripts/oneclick/run_vizdoom_human_vs_llm_record.sh
```

### 6.5 LLM vs LLM

```bash
bash scripts/oneclick/run_vizdoom_llm_vs_llm.sh
```

### 6.6 AI vs AI

```bash
bash scripts/oneclick/run_vizdoom_ai_vs_ai.sh
```

### 6.7 Agent vs LLM

```bash
bash scripts/oneclick/run_vizdoom_agent_vs_llm.sh
```

## 7. Controls

Current Human key mapping:

- `A` / `Left`: `2`
- `D` / `Right`: `3`
- `Space` / `J`: `1`

## 8. Run Outputs

After a run finishes, outputs are written under:

```text
runs/<run_id>/
```

Important files and directories:

- `summary.json`
- `samples/`
- `samples.jsonl`
- `replays/`

## 9. Replay Command

Replay one finished run by `run_id`:

```bash
bash scripts/oneclick/run_vizdoom_replay.sh <run_id>
```

Example:

```bash
bash scripts/oneclick/run_vizdoom_replay.sh vizdoom_human_vs_llm_20260228_102306
```

Default replay viewer:

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

## 10. Recommended Quick Start

If you only want to confirm the environment is working:

1. Install dependencies
2. Run `human_vs_dummy`
3. Confirm the game window appears
4. Confirm outputs are written to `runs/<run_id>/`
5. Replay the same run with `run_vizdoom_replay.sh`

## 11. Troubleshooting

### 11.1 `ImportError: No module named vizdoom`

Install ViZDoom:

```bash
pip install vizdoom
```

### 11.2 Human input does not respond

- Make sure the local input window has focus
- Human modes currently depend on local `pygame`

### 11.3 Replay script cannot find `samples`

Replay depends on sample artifacts. Confirm:

- `runs/<run_id>/samples/` exists
- at least one sample JSON exists under that directory
