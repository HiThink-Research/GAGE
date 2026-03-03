# Retro Mario Game Arena 指南

[English](game_arena_retro_mario.md) | 中文

这是当前仓库里 stable-retro Mario 的标准 Game Arena 文档，统一整理了准备步骤、启动命令、回放方式和关键参数位置。

## 1. 概览

当前仓库里的 Retro Mario 仍然是纯配置驱动，还没有 Mario 专用的一键启动脚本。标准路径是：

1. 先把 ROM 导入 stable-retro。
2. 从 `config/custom/` 里选择一个 YAML 配置。
3. 通过 `python run.py --config ...` 启动。
4. 如果配置启用了 websocket，就在运行中打开 ws_rgb；否则运行后再回放。

## 2. 标准入口文件

| 类型 | 路径 | 用途 |
| --- | --- | --- |
| Dummy ws_rgb 配置 | `config/custom/retro_mario_phase1_dummy_ws.yaml` | 最快的实时查看冒烟验证 |
| Human ws_rgb 配置 | `config/custom/retro_mario_phase1_human_ws.yaml` | 人类手动控制 Mario |
| OpenAI ws_rgb 配置 | `config/custom/retro_mario_openai_ws_rgb_auto_eval.yaml` | API 驱动的实时查看 Demo |
| OpenAI headless 配置 | `config/custom/retro_mario_openai_headless_auto_eval.yaml` | API 驱动的无界面 Demo |
| Dummy headless 配置 | `config/custom/retro_mario_phase1_dummy_headless_auto_eval.yaml` | 不依赖 viewer 的离线冒烟测试 |
| 数据集 | `config/custom/retro_mario_phase1.jsonl` | 默认样本输入 |
| 环境实现 | `src/gage_eval/role/arena/games/retro/retro_env.py` | 运行时行为和 replay 写出逻辑 |
| 动作解析器 | `src/gage_eval/role/arena/parsers/retro_action_parser.py` | 解析 `{"move": "...", "hold_ticks": ...}` 动作 |

## 3. 前置准备

运行任何 Mario 配置前，都要先安装 `stable-retro` 并导入 ROM。

```bash
python -m pip install stable-retro
python -m retro.import "<rom_save_path>"
```

当前配置使用的游戏 id 是：

```text
SuperMarioBros3-Nes-v0
```

如果使用 OpenAI 配置，运行前先设置：

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
```

可选模型覆盖：

```bash
export RETRO_OPENAI_MODEL="gpt-4o-mini"
```

## 4. 启动路径

### 4.1 推荐冒烟路径：dummy + ws_rgb

```bash
export RETRO_WS_RGB_PORT=5800

env PYTHONPATH=src python run.py \
  --config config/custom/retro_mario_phase1_dummy_ws.yaml \
  --output-dir runs \
  --run-id retro_mario_dummy_ws
```

默认实时查看地址：

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

### 4.2 OpenAI + ws_rgb

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
export RETRO_WS_RGB_PORT=5800

env PYTHONPATH=src python run.py \
  --config config/custom/retro_mario_openai_ws_rgb_auto_eval.yaml \
  --output-dir runs \
  --run-id retro_mario_openai_ws
```

### 4.3 Human + ws_rgb

```bash
env PYTHONPATH=src python run.py \
  --config config/custom/retro_mario_phase1_human_ws.yaml \
  --output-dir runs \
  --run-id retro_mario_human_ws
```

当前 human 配置里的默认值：

- Viewer 地址：`http://127.0.0.1:5800/ws_rgb/viewer`
- 队列输入端口：`human_input.host: 0.0.0.0`，`human_input.port: 8001`
- 输入 FPS：`human_input.fps: 30`

浏览器侧默认按键别名来自 retro input mapper：

- 移动：`W/A/S/D` 或方向键
- Jump：`J`、`Space`、`Z`、`C`
- Run：`K` 或 `X`
- Select：`L` 或 `Shift`
- Start：`Enter`

### 4.4 Headless 路径

```bash
env PYTHONPATH=src python run.py \
  --config config/custom/retro_mario_phase1_dummy_headless_auto_eval.yaml \
  --output-dir runs \
  --run-id retro_mario_dummy_headless
```

如果你只想做离线验证、不需要 ws_rgb，就使用 headless 配置。

## 5. 启动顺序

当前仓库里的 Retro Mario 启动顺序是：

1. 安装 `stable-retro`。
2. 用 `python -m retro.import` 导入 ROM。
3. 选择一个 YAML 配置。
4. 只有 OpenAI 配置才需要设置 API Key。
5. 执行 `python run.py --config ...`。
6. 对 websocket 配置，在运行中打开 `ws_rgb/viewer`；对已有产物，再单独启动 replay。

## 6. 关键参数与修改位置

| 项目 | 修改位置 | 含义 |
| --- | --- | --- |
| API Key | Shell 环境变量 `OPENAI_API_KEY` | `retro_mario_openai_*` 配置会读取它 |
| 模型名 | `RETRO_OPENAI_MODEL` 或 `backends[].config.model` | 选择 OpenAI 模型 |
| 实时查看端口 | `human_input.ws_port` 或环境变量 `RETRO_WS_RGB_PORT` | websocket 配置的 ws_rgb 端口 |
| 显示模式 | `environment.display_mode` | `websocket` 用于实时查看，`headless` 用于离线运行 |
| 合法动作 | `environment.legal_moves` | 暴露给玩家的动作集合 |
| 持续帧数 | `environment.action_schema.hold_ticks_*` 和 `parser.hold_ticks_*` | 宏动作持续多久，以及默认值/上下界 |
| Tick 节奏 | `scheduler.tick_ms` | 主调度器的毫秒级节奏 |
| Human 输入 FPS | `human_input.fps` | human 模式下输入采样频率 |
| Replay 模式 | `environment.replay.mode` | 决定写出 `action`、`frame` 或 `both` |
| Replay 帧采样 | `environment.replay.frame_capture.*` | 控制帧采样步长、格式和帧数上限 |

补充说明：

- Mario 动作是 JSON，例如 `{"move":"right_run_jump","hold_ticks":6}`。
- 如果模型没有输出 `hold_ticks`，解析器会补默认值，并限制在配置范围内。
- `display_mode: websocket` 在运行时会被当成 headless 环境加 ws_rgb 推流，而不是本地窗口渲染。

## 7. 产物与回放

运行产物默认写到：

```text
runs/<run_id>/
```

可用单个 sample 产物启动回放：

```bash
RUN_ID="<your_run_id>"
SAMPLE_JSON="$(find "runs/${RUN_ID}/samples" -type f -name '*.json' | head -n 1)"
REPLAY_PORT="${RETRO_REPLAY_PORT:-5800}"

env PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 0.0.0.0 \
  --port "${REPLAY_PORT}" \
  --fps 12
```

回放地址：

```text
http://127.0.0.1:<REPLAY_PORT>/ws_rgb/viewer
```

启动回放前，建议先确认 sample JSON 里存在 `replay_path` 字段。

## 8. 旧文档

下面这些旧文档仍保留在仓库里，但不再作为标准入口：

- `docs/guide/stable_retro_mario_demo_guide.md`
- `docs/guide/stable_retro_mario_demo_guide_zh.md`
