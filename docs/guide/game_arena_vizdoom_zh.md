# ViZDoom Game Arena 指南

[English](game_arena_vizdoom.md) | 中文

这是当前仓库里 ViZDoom 的标准 Game Arena 文档，统一整理了安装、启动脚本、回放和关键参数位置。

## 1. 概览

ViZDoom 目前有两类主要启动方式：

- 基于 `pygame` 本地窗口的人类输入，一般通过 `scripts/oneclick/` 下的脚本启动
- 可选的 ws_rgb 浏览器输入，一般通过手动执行 websocket 配置启动

对大多数读者来说，标准入口仍然是 `scripts/oneclick/` 下的 ViZDoom 一键脚本。

## 2. 标准入口文件

| 类型 | 路径 | 用途 |
| --- | --- | --- |
| Human vs Dummy 脚本 | `scripts/oneclick/run_vizdoom_human_vs_dummy.sh` | 推荐的首个环境验证脚本 |
| Human Solo 脚本 | `scripts/oneclick/run_vizdoom_human_solo.sh` | 单人练习模式 |
| Human vs LLM 脚本 | `scripts/oneclick/run_vizdoom_human_vs_llm.sh` | 本地人类输入对战 LLM |
| Human vs LLM Record 脚本 | `scripts/oneclick/run_vizdoom_human_vs_llm_record.sh` | record 调度版本 |
| LLM vs LLM 脚本 | `scripts/oneclick/run_vizdoom_llm_vs_llm.sh` | 双 LLM 对战 |
| AI vs AI 脚本 | `scripts/oneclick/run_vizdoom_ai_vs_ai.sh` | 只是脚本别名，默认仍指向 `config/custom/vizdoom_llm_vs_llm.yaml` |
| Agent vs LLM 脚本 | `scripts/oneclick/run_vizdoom_agent_vs_llm.sh` | 增量式 human/agent 变体 |
| 回放脚本 | `scripts/oneclick/run_vizdoom_replay.sh` | 通过 `run_id` 回放一局已完成对局 |
| Websocket Human 配置 | `config/custom/vizdoom_human_vs_llm_record_ws_rgb.yaml` | 通过 ws_rgb 做浏览器输入 |
| 主配置集合 | `config/custom/vizdoom_*.yaml` | 所有 ViZDoom 运行配置 |

## 3. 前置准备

先安装项目依赖：

```bash
pip install -r requirements.txt
```

最小验证命令：

```bash
python - <<'PY'
import vizdoom
import pygame
print("vizdoom ok")
print("pygame ok")
PY
```

说明：

- 使用一键脚本的人类模式，需要桌面图形环境，因为输入来自本地 `pygame` 窗口。
- 带 LLM 的模式需要 `OPENAI_API_KEY` 或 `LITELLM_API_KEY`。

## 4. 启动路径

### 4.1 推荐冒烟路径：human vs dummy

```bash
bash scripts/oneclick/run_vizdoom_human_vs_dummy.sh
```

脚本默认变量：

- `PYTHON_BIN`：Python 解释器
- `CFG`：默认 `config/custom/vizdoom_human_vs_dummy.yaml`
- `RUN_ID`：默认 `vizdoom_human_vs_dummy_<timestamp>`
- `OUTPUT_DIR`：默认 `runs/`

脚本打印出来的本地键位：

- `A` 或 `Left`：动作 `2`
- `D` 或 `Right`：动作 `3`
- `Space` 或 `J`：动作 `1`

### 4.2 Human vs LLM

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/oneclick/run_vizdoom_human_vs_llm.sh
```

脚本也支持 `LITELLM_API_KEY`。如果只设置了 `LITELLM_API_KEY`，脚本会把它同步到 `OPENAI_API_KEY`。

### 4.3 Record 调度版本

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/oneclick/run_vizdoom_human_vs_llm_record.sh
```

### 4.4 LLM vs LLM

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/oneclick/run_vizdoom_llm_vs_llm.sh
```

### 4.5 可选的 websocket human 输入

如果你想用浏览器输入，而不是本地 `pygame` 窗口，可以直接跑下面这个配置：

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
PYTHONPATH=src python run.py \
  --config config/custom/vizdoom_human_vs_llm_record_ws_rgb.yaml \
  --output-dir runs \
  --run-id vizdoom_human_vs_llm_record_ws
```

这个 websocket 配置的默认 viewer 地址：

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

## 5. 启动顺序

当前仓库里的 ViZDoom 推荐顺序是：

1. 安装 `vizdoom`、`pygame` 等可选运行时依赖。
2. 选择一个一键脚本或 YAML 配置。
3. 如果模式包含 LLM 后端，先设置 API Key。
4. 启动运行。
5. 对 pygame 模式，确保本地输入窗口保持焦点。
6. 对局结束后，再用 `run_vizdoom_replay.sh` 启动回放。

## 6. 关键参数与修改位置

| 项目 | 修改位置 | 含义 |
| --- | --- | --- |
| API Key | Shell 环境变量 `OPENAI_API_KEY` 或 `LITELLM_API_KEY` | LLM 模式必需 |
| 脚本配置路径 | 脚本变量 `CFG` | 替换一键脚本默认使用的 YAML |
| 输出目录 | 脚本变量 `OUTPUT_DIR` | 运行产物写出位置 |
| 调度节奏 | `scheduler.tick_ms` | 调度器毫秒级 tick 间隔 |
| Record 超时 | `scheduler.action_timeout_ms` | record 模式下人类或模型等待超时 |
| 最大步数 | `environment.max_steps` | 后端最大执行步数 |
| Action Repeat | `environment.action_repeat` | 每次动作对应多少个后端帧 |
| 运行时节流 | `environment.sleep_s` | 每次后端 tick 之间附加 sleep |
| 本地渲染 | `environment.render_mode` | 本地窗口/显示行为 |
| POV 和 automap | `environment.show_pov`、`environment.show_automap` | 控制显示哪些视角 |
| Replay 输出目录 | `environment.replay_output_dir` | replay 文件写到哪里 |
| Replay 采集配置 | `environment.replay.*` | action/frame replay 的写出方式 |
| Replay 播放 FPS | `scripts/oneclick/run_vizdoom_replay.sh` 的 `FPS` | viewer 回放速度 |
| Websocket 端口 | `human_input.host`、`human_input.port`、`human_input.ws_port` | 浏览器输入和 viewer 端口 |

和帧率相关的补充说明：

- 常规一键运行时，主要节奏控制项是 `scheduler.tick_ms`、`environment.action_repeat` 和 `environment.sleep_s`。
- 回放播放速度单独由 `run_vizdoom_replay.sh` 的 `FPS` 控制。
- 如果你显式启用了 `environment.config_path: src/gage_eval/role/arena/games/vizdoom/_vizdoom.ini`，引擎层的 FPS 上限在这个 INI 文件里，通过 `cl_capfps` 和 `vid_maxfps` 控制。

## 7. 产物与回放

运行产物默认写到：

```text
runs/<run_id>/
```

环境配置里默认的 replay 输出目录是：

```text
runs/vizdoom_replays/
```

回放命令：

```bash
bash scripts/oneclick/run_vizdoom_replay.sh <run_id>
```

常用回放变量：

- `PYTHON_BIN`：Python 解释器
- `HOST`：绑定地址，默认 `127.0.0.1`
- `PORT`：回放 viewer 端口，默认 `5800`
- `FPS`：回放播放帧率，默认 `8`
- `MAX_FRAMES`：可选的回放帧数上限

默认回放地址：

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

## 8. 旧文档

下面这些旧文档仍保留在仓库里，但不再作为标准入口：

- `docs/guide/vizdoom_setup_guide.md`
- `docs/guide/vizdoom_setup_guide_zh.md`
- `docs/guide/vizdoom_arena_summary.md`
- `docs/guide/vizdoom_arena_summary_zh.md`
