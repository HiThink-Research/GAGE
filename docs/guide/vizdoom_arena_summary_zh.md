# ViZDoom Arena 实现说明

[English](vizdoom_arena_summary.md) | 中文

本文档简要说明本周在当前 Arena 架构下完成的 ViZDoom 相关代码，以及可直接使用的启动命令。

## 1. 已实现内容

- 实现 Arena A 线调度器：
  - `record_scheduler`
  - `simultaneous_scheduler`
  - `multi_timeline_scheduler`
- 将 ViZDoom 接入重构后的 Arena 运行时。
- 增加 ViZDoom 单进程和多进程环境封装。
- 增加 ViZDoom 离散动作 parser 与 action codec。
- 支持 Human / LLM / Dummy 组合运行。
- 支持 `action`、`frame`、`both` 三种 replay 输出模式。
- 支持按 `run_id` 启动 replay 回放。
- 增加调度器、replay、parser 和 ViZDoom 运行时相关测试。

## 2. 关键代码位置

- Arena 调度器：
  - [`src/gage_eval/role/arena/schedulers/`](../../src/gage_eval/role/arena/schedulers/)
- Arena 适配层：
  - [`src/gage_eval/role/adapters/arena.py`](../../src/gage_eval/role/adapters/arena.py)
- ViZDoom 环境：
  - [`src/gage_eval/role/arena/games/vizdoom/env.py`](../../src/gage_eval/role/arena/games/vizdoom/env.py)
  - [`src/gage_eval/role/arena/games/vizdoom/env_vizdoom_mp.py`](../../src/gage_eval/role/arena/games/vizdoom/env_vizdoom_mp.py)
  - [`src/gage_eval/role/arena/games/vizdoom/env_vizdoom_mp_proc.py`](../../src/gage_eval/role/arena/games/vizdoom/env_vizdoom_mp_proc.py)
- ViZDoom parser：
  - [`src/gage_eval/role/arena/parsers/vizdoom_parser.py`](../../src/gage_eval/role/arena/parsers/vizdoom_parser.py)
  - [`src/gage_eval/role/arena/parsers/vizdoom_action_codec.py`](../../src/gage_eval/role/arena/parsers/vizdoom_action_codec.py)
- Replay 写出：
  - [`src/gage_eval/role/arena/replay_schema_writer.py`](../../src/gage_eval/role/arena/replay_schema_writer.py)

## 3. 启动命令

以下命令默认在仓库根目录执行。

### 3.1 Human vs LLM

```bash
bash scripts/oneclick/run_vizdoom_human_vs_llm.sh
```

需要：

- `OPENAI_API_KEY` 或 `LITELLM_API_KEY`

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

需要：

- `OPENAI_API_KEY` 或 `LITELLM_API_KEY`

### 3.5 AI vs AI

```bash
bash scripts/oneclick/run_vizdoom_ai_vs_ai.sh
```

需要：

- `OPENAI_API_KEY` 或 `LITELLM_API_KEY`

### 3.6 Agent vs LLM

```bash
bash scripts/oneclick/run_vizdoom_agent_vs_llm.sh
```

需要：

- `OPENAI_API_KEY` 或 `LITELLM_API_KEY`

### 3.7 Record Scheduler 模式

```bash
bash scripts/oneclick/run_vizdoom_human_vs_llm_record.sh
```

需要：

- `OPENAI_API_KEY` 或 `LITELLM_API_KEY`

## 4. 回放命令

按 `run_id` 启动一局已经完成的回放：

```bash
bash scripts/oneclick/run_vizdoom_replay.sh <run_id>
```

示例：

```bash
bash scripts/oneclick/run_vizdoom_replay.sh vizdoom_human_vs_llm_20260228_102306
```

默认查看地址：

```text
http://127.0.0.1:5800/ws_rgb/viewer
```
