# ViZDoom Arena Summary

English | [中文](vizdoom_arena_summary_zh.md)

This document summarizes the ViZDoom work completed on the current Arena architecture and lists the available startup commands.

## 1. Implemented Content

- Implemented Arena A line schedulers:
  - `record_scheduler`
  - `simultaneous_scheduler`
  - `multi_timeline_scheduler`
- Integrated ViZDoom into the refactored Arena runtime.
- Added ViZDoom environment wrappers for single-process and multi-process execution.
- Added ViZDoom parser and action codec for discrete action IDs.
- Added Human / LLM / Dummy combinations under the new Arena flow.
- Added replay generation for `action`, `frame`, and `both` modes.
- Added replay startup support through `run_id`.
- Added unit tests and integration tests for schedulers, replay, parser, and ViZDoom runtime.

## 2. Key Code Paths

- Arena schedulers:
  - [`src/gage_eval/role/arena/schedulers/`](../../src/gage_eval/role/arena/schedulers/)
- Arena adapter:
  - [`src/gage_eval/role/adapters/arena.py`](../../src/gage_eval/role/adapters/arena.py)
- ViZDoom environment:
  - [`src/gage_eval/role/arena/games/vizdoom/env.py`](../../src/gage_eval/role/arena/games/vizdoom/env.py)
  - [`src/gage_eval/role/arena/games/vizdoom/env_vizdoom_mp.py`](../../src/gage_eval/role/arena/games/vizdoom/env_vizdoom_mp.py)
  - [`src/gage_eval/role/arena/games/vizdoom/env_vizdoom_mp_proc.py`](../../src/gage_eval/role/arena/games/vizdoom/env_vizdoom_mp_proc.py)
- ViZDoom parser:
  - [`src/gage_eval/role/arena/parsers/vizdoom_parser.py`](../../src/gage_eval/role/arena/parsers/vizdoom_parser.py)
  - [`src/gage_eval/role/arena/parsers/vizdoom_action_codec.py`](../../src/gage_eval/role/arena/parsers/vizdoom_action_codec.py)
- Replay writer:
  - [`src/gage_eval/role/arena/replay_schema_writer.py`](../../src/gage_eval/role/arena/replay_schema_writer.py)

## 3. Startup Commands

Run commands assume the repository root as the working directory.

### 3.1 Human vs LLM

```bash
bash scripts/oneclick/run_vizdoom_human_vs_llm.sh
```

Requires:

- `OPENAI_API_KEY` or `LITELLM_API_KEY`

### 3.2 Human vs Dummy

```bash
bash scripts/oneclick/run_vizdoom_human_vs_dummy.sh
```

### 3.3 Human Solo

```bash
bash scripts/oneclick/run_vizdoom_human_solo.sh
```

### 3.4 LLM vs LLM

```bash
bash scripts/oneclick/run_vizdoom_llm_vs_llm.sh
```

Requires:

- `OPENAI_API_KEY` or `LITELLM_API_KEY`

### 3.5 AI vs AI

```bash
bash scripts/oneclick/run_vizdoom_ai_vs_ai.sh
```

Requires:

- `OPENAI_API_KEY` or `LITELLM_API_KEY`

### 3.6 Agent vs LLM

```bash
bash scripts/oneclick/run_vizdoom_agent_vs_llm.sh
```

Requires:

- `OPENAI_API_KEY` or `LITELLM_API_KEY`

### 3.7 Record Scheduler Run

```bash
bash scripts/oneclick/run_vizdoom_human_vs_llm_record.sh
```

Requires:

- `OPENAI_API_KEY` or `LITELLM_API_KEY`

## 4. Replay Command

Replay one finished run by `run_id`:

```bash
bash scripts/oneclick/run_vizdoom_replay.sh <run_id>
```

Example:

```bash
bash scripts/oneclick/run_vizdoom_replay.sh vizdoom_human_vs_llm_20260228_102306
```

Default viewer URL:

```text
http://127.0.0.1:5800/ws_rgb/viewer
```
