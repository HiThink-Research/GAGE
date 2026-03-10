# ViZDoom Game Arena 指南

[English](game_arena_vizdoom.md) | 中文

这是当前仓库里 ViZDoom 的标准 Game Arena 文档，统一整理了安装、启动脚本、回放和关键参数位置。

## 1. 概览

ViZDoom 目前有两类主要启动方式：

- 基于 ws_rgb 的 human vs LLM 浏览器交互
- 其他路径仍可使用本地 `pygame` 输入或 backend-only 运行

现在文档里推荐的验证顺序是：先跑 dummy 冒烟，再切到浏览器里的 `human vs LLM` 示例。

## 2. 标准入口文件

| 类型 | 路径 | 用途 |
| --- | --- | --- |
| ws_rgb helper | `scripts/run/arenas/vizdoom/viewer.sh` | 通用 ws_rgb helper，会等待 viewer 可访问后再继续 |
| Human vs LLM ws_rgb 配置 | `config/custom/vizdoom_human_vs_llm_tick_ws_rgb_strategy.yaml` | 推荐的浏览器版 human vs LLM 示例，使用 tick 调度 |
| Dummy ws_rgb 配置 | `config/custom/vizdoom_dummy_vs_dummy_ws_rgb.yaml` | 可选的 dummy-only websocket 环境检查配置 |
| Human vs Dummy 脚本 | `scripts/run/arenas/vizdoom/run.sh --mode human-vs-dummy` | 本地 pygame 环境验证脚本 |
| Human Solo 脚本 | `scripts/run/arenas/vizdoom/run.sh --mode human-solo` | 单人练习模式 |
| Human vs LLM 脚本 | `scripts/run/arenas/vizdoom/run.sh --mode human-vs-llm` | 本地人类输入对战 LLM |
| Human vs LLM Record 脚本 | `scripts/run/arenas/vizdoom/run.sh --mode human-vs-llm-record` | record 调度版本 |
| LLM vs LLM 脚本 | `scripts/run/arenas/vizdoom/run.sh --mode llm-vs-llm` | 双 LLM 对战 |
| AI vs AI 脚本 | `scripts/run/arenas/vizdoom/run.sh --mode ai-vs-ai` | AI-vs-AI 流程的统一入口 |
| Agent vs LLM 脚本 | `scripts/run/arenas/vizdoom/run.sh --mode agent-vs-llm` | 增量式 human/agent 变体 |
| 回放脚本 | `scripts/run/arenas/vizdoom/replay.sh` | 通过 `run_id` 回放一局已完成对局 |
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

- 使用统一运行脚本的人类模式，需要桌面图形环境，因为输入来自本地 `pygame` 窗口。
- 带 LLM 的模式需要 `OPENAI_API_KEY` 或 `LITELLM_API_KEY`。

## 4. 启动路径

### 4.1 推荐冒烟路径：dummy + ws_rgb

```bash
RUN_ID="vizdoom_dummy_ws_rgb_$(date +%Y%m%d_%H%M%S)" \
bash scripts/run/arenas/vizdoom/viewer.sh
```

建议先用这条命令验证 ViZDoom、ws_rgb 和浏览器查看链路，再继续切到模型模式。

这个 ws_rgb helper 的执行顺序：

1. 选择 Python 解释器。
2. 校验配置文件路径。
3. 选择一个可用的 `WS_RGB_PORT`。
4. 在后台启动 `python run.py --config ...`。
5. 等待 `http://127.0.0.1:<port>/ws_rgb/viewer` 可访问。
6. 打印可用的 viewer 地址，并按配置决定是否自动打开浏览器。

这个示例里最常用的变量：

- `PYTHON_BIN`：Python 解释器
- `CFG`：默认 `config/custom/vizdoom_dummy_vs_dummy_ws_rgb.yaml`
- `RUN_ID`：写入 `runs/` 的运行编号
- `OUTPUT_DIR`：默认 `runs`
- `WS_RGB_HOST`：默认 `127.0.0.1`
- `WS_RGB_PORT`：默认 `5800`

如果你想改成本地窗口验证路径，可以执行：

```bash
bash scripts/run/arenas/vizdoom/run.sh --mode human-vs-dummy
```

本地脚本默认变量：

- `PYTHON_BIN`：Python 解释器
- `CFG`：默认 `config/custom/vizdoom_human_vs_dummy.yaml`
- `RUN_ID`：默认 `vizdoom_human_vs_dummy_<timestamp>`
- `OUTPUT_DIR`：默认 `runs/`

pygame 脚本打印出来的本地键位：

- `A` 或 `Left`：动作 `2`
- `D` 或 `Right`：动作 `3`
- `Space` 或 `J`：动作 `1`

### 4.2 推荐模型示例：human vs LLM + ws_rgb 实时查看

```bash
OPENAI_API_KEY="<YOUR_KEY>" \
VIZDOOM_P1_SCHEME_ID=S3_text_image_current \
RUN_ID="vizdoom_human_vs_llm_ws_rgb_$(date +%Y%m%d_%H%M%S)" \
CFG=config/custom/vizdoom_human_vs_llm_tick_ws_rgb_strategy.yaml \
bash scripts/run/arenas/vizdoom/viewer.sh
```

如果你的目标是“人在网页里操作，LLM 作为对手在玩，同时浏览器里能看到实时画面”，文档里推荐的就是这条命令。

这个示例里最常用的变量：

- `PYTHON_BIN`：Python 解释器
- `CFG`：文档里的 human-vs-LLM 路径使用 `config/custom/vizdoom_human_vs_llm_tick_ws_rgb_strategy.yaml`
- `VIZDOOM_P1_SCHEME_ID`：用于选择 LLM 策略，例如 `S3_text_image_current`
- `RUN_ID`：写入 `runs/` 的运行编号
- `OUTPUT_DIR`：默认 `runs`
- `WS_RGB_HOST`：默认 `127.0.0.1`
- `WS_RGB_PORT`：默认 `5800`

这个命令里如果要切换模型或 API，请改这里：

- API Key：启动前先在 shell 里设置 `OPENAI_API_KEY`。脚本也接受 `LITELLM_API_KEY`，并会自动同步到 `OPENAI_API_KEY`。即使你接的是本地 OpenAI 兼容服务，也要保证这两个变量里至少有一个非空，因为启动脚本会先检查。
- 推荐 ws_rgb 示例：编辑 `config/custom/vizdoom_human_vs_llm_tick_ws_rgb_strategy.yaml` 里的 `backends[0].config`。切换远程 API 或本地服务时改 `api_base`；如果是 OpenAI 兼容接口，`provider` 继续保持 `openai`；切模型时改 `model`，或者直接设置环境变量 `VIZDOOM_P1_MODEL`。
- 其他模型命令也是改各自 YAML 里的同一组字段：`human-vs-llm` 对应 `config/custom/vizdoom_human_vs_llm.yaml`，`human-vs-llm-record` 对应 `config/custom/vizdoom_human_vs_llm_record.yaml`，`llm-vs-llm` 对应 `config/custom/vizdoom_llm_vs_llm.yaml`。

### 4.3 Human vs LLM

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/run/arenas/vizdoom/run.sh --mode human-vs-llm
```

脚本也支持 `LITELLM_API_KEY`。如果只设置了 `LITELLM_API_KEY`，脚本会把它同步到 `OPENAI_API_KEY`。

### 4.4 Record 调度版本

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/run/arenas/vizdoom/run.sh --mode human-vs-llm-record
```

### 4.5 LLM vs LLM

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
bash scripts/run/arenas/vizdoom/run.sh --mode llm-vs-llm
```

### 4.6 可选的 websocket human 输入

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
2. 先用 dummy ws_rgb 路径验证环境，再切到 `vizdoom_human_vs_llm_tick_ws_rgb_strategy.yaml` 或其他 `scripts/run/arenas/vizdoom/run.sh --mode ...` 模式做模型验证。
3. 如果模式包含 LLM 后端，先设置 API Key。
4. 启动运行。
5. 对 ws_rgb helper，等待脚本打印 viewer 地址；对 pygame 模式，确保本地输入窗口保持焦点。
6. 对局结束后，再用 `scripts/run/arenas/vizdoom/replay.sh` 启动回放。

## 6. 关键参数与修改位置

| 项目 | 修改位置 | 含义 |
| --- | --- | --- |
| API Key | Shell 环境变量 `OPENAI_API_KEY` 或 `LITELLM_API_KEY` | LLM 模式必需 |
| 脚本配置路径 | 脚本变量 `CFG` | 替换统一运行入口默认使用的 YAML |
| 后端地址 | 当前所选 `config/custom/vizdoom_*.yaml` 里的 `backends[].config.api_base` | 在托管 API 和本地 OpenAI 兼容服务之间切换 |
| 模型名 | `backends[].config.model`，以及策略版 ws_rgb 配置里的环境变量 `VIZDOOM_P1_MODEL` | 选择 LLM 使用的模型 |
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
| Replay 播放 FPS | `scripts/run/arenas/vizdoom/replay.sh` 的 `FPS` | viewer 回放速度 |
| Websocket 端口 | `human_input.host`、`human_input.port`、`human_input.ws_port` | 浏览器输入和 viewer 端口 |

和帧率相关的补充说明：

- 常规脚本运行时，主要节奏控制项是 `scheduler.tick_ms`、`environment.action_repeat` 和 `environment.sleep_s`。
- 回放播放速度单独由 `scripts/run/arenas/vizdoom/replay.sh` 的 `FPS` 控制。
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
bash scripts/run/arenas/vizdoom/replay.sh <run_id>
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
