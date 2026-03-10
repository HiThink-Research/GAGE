# Retro Mario Game Arena 指南

[English](game_arena_retro_mario.md) | 中文

这是当前仓库里 stable-retro Mario 的标准 Game Arena 文档，统一整理了准备步骤、脚本化启动方式、回放方式和关键参数位置。

## 1. 概览

Retro Mario 现在已经统一到脚本入口，覆盖以下常见路径：

- Dummy + websocketRGB 冒烟验证
- OpenAI + websocketRGB 实时查看
- Human + websocketRGB 人类控制
- Dummy 或 OpenAI 的 headless 运行

实际运行行为仍由 YAML 配置决定，但启动和回放入口已经统一收口到 `scripts/run/arenas/retro_mario/`。

## 2. 标准入口文件

| 类型 | 路径 | 用途 |
| --- | --- | --- |
| 标准启动脚本 | `scripts/run/arenas/retro_mario/run.sh` | Mario 常见模式的统一启动入口 |
| 回放脚本 | `scripts/run/arenas/retro_mario/replay.sh` | 通过 `run_id` 回放一局已完成运行 |
| Dummy websocketRGB 配置 | `config/custom/retro_mario_phase1_dummy_ws.yaml` | 最快的实时查看冒烟验证 |
| Human websocketRGB 配置 | `config/custom/retro_mario_phase1_human_ws.yaml` | 人类手动控制 Mario |
| OpenAI websocketRGB 配置 | `config/custom/retro_mario_openai_ws_rgb_auto_eval.yaml` | API 驱动的实时查看 Demo |
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

### 4.1 推荐冒烟路径：dummy + websocketRGB

```bash
bash scripts/run/arenas/retro_mario/run.sh \
  --mode dummy_ws \
  --run-id "retro_mario_dummy_ws_$(date +%Y%m%d_%H%M%S)"
```

默认实时查看地址：

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

标准启动脚本的执行顺序：

1. 选择 Python 解释器。
2. 按 `--mode` 解析 YAML；如果传了 `--config`，就直接使用显式配置。
3. 校验配置文件路径。
4. 对 OpenAI 模式检查 API Key。
5. 执行 `python run.py --config ...`。
6. 对 websocket 模式打印 viewer 地址或 human 输入端口提示。

`scripts/run/arenas/retro_mario/run.sh` 当前支持的模式：

- `dummy_ws`
- `openai_ws`
- `human_ws`
- `dummy_headless`
- `openai_headless`

常用启动参数：

- `--mode`：从上面的模式列表中选择
- `--config`：显式 YAML 路径，会覆盖脚本内置映射
- `--run-id`：写入 `runs/` 的运行编号
- `--output-dir`：输出目录，默认 `runs`
- `--python-bin`：指定 Python 解释器

常用环境变量：

- `RETRO_WS_RGB_PORT`：websocket 模式的 websocketRGB 端口
- `OPENAI_API_KEY`：`openai_ws` 和 `openai_headless` 必需
- `LITELLM_API_KEY`：可作为 OpenAI 模式的回退 key 来源
- `RETRO_OPENAI_MODEL`：如果配置支持，可作为模型名覆盖

### 4.2 OpenAI + websocketRGB

```bash
export OPENAI_API_KEY="<YOUR_KEY>"

bash scripts/run/arenas/retro_mario/run.sh \
  --mode openai_ws \
  --run-id "retro_mario_openai_ws_$(date +%Y%m%d_%H%M%S)"
```

这个命令里如果要切换模型或 API，请改这里：

- API Key：启动前先设置 `OPENAI_API_KEY`。`scripts/run/arenas/retro_mario/run.sh` 也接受 `LITELLM_API_KEY`，并会自动同步到 `OPENAI_API_KEY`。
- `openai_ws` 读取的是 `config/custom/retro_mario_openai_ws_rgb_auto_eval.yaml`；`openai_headless` 读取的是 `config/custom/retro_mario_openai_headless_auto_eval.yaml`。
- 托管 OpenAI 兼容 API：切换接口地址时改 `backends[0].config.base_url`；切换模型时改 `backends[0].config.model`，或者直接设置 `RETRO_OPENAI_MODEL`。
- 本地 OpenAI 兼容服务：把 `backends[0].config.base_url` 改成本地服务地址，再把 `backends[0].config.model` 改成服务暴露的模型名。当前这两份配置里 `require_api_key: true`，所以要么保持 `OPENAI_API_KEY` 非空，要么在你确认本地服务不需要鉴权时把这个开关改成 `false`。

### 4.3 Human + websocketRGB

```bash
bash scripts/run/arenas/retro_mario/run.sh \
  --mode human_ws \
  --run-id "retro_mario_human_ws_$(date +%Y%m%d_%H%M%S)"
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
bash scripts/run/arenas/retro_mario/run.sh \
  --mode dummy_headless \
  --run-id "retro_mario_dummy_headless_$(date +%Y%m%d_%H%M%S)"
```

如果你只想做离线验证、不需要 websocketRGB，就使用 headless 配置。

OpenAI headless 示例：

```bash
export OPENAI_API_KEY="<YOUR_KEY>"

bash scripts/run/arenas/retro_mario/run.sh \
  --mode openai_headless \
  --run-id "retro_mario_openai_headless_$(date +%Y%m%d_%H%M%S)"
```

### 4.5 显式配置覆盖

如果你想保留脚本入口，但手动指定 YAML，可以直接用 `--config`：

```bash
bash scripts/run/arenas/retro_mario/run.sh \
  --config config/custom/retro_mario_phase1_dummy_ws.yaml \
  --run-id "retro_mario_custom_$(date +%Y%m%d_%H%M%S)"
```

## 5. 启动顺序

当前仓库里的 Retro Mario 启动顺序是：

1. 安装 `stable-retro`。
2. 用 `python -m retro.import` 导入 ROM。
3. 选择一个标准 `--mode`，或者指定 `--config`。
4. 只有 OpenAI 模式才需要设置 API Key。
5. 执行 `scripts/run/arenas/retro_mario/run.sh`。
6. 对 websocket 模式，在运行中打开 `ws_rgb/viewer`；对已有产物，再执行 `scripts/run/arenas/retro_mario/replay.sh <run_id>`。

## 6. 关键参数与修改位置

| 项目 | 修改位置 | 含义 |
| --- | --- | --- |
| API Key | Shell 环境变量 `OPENAI_API_KEY` 或 `LITELLM_API_KEY` | OpenAI 模式必需；脚本接受这两个变量 |
| 启动模式 | `scripts/run/arenas/retro_mario/run.sh --mode` | 选择 `dummy_ws`、`openai_ws`、`human_ws`、`dummy_headless`、`openai_headless` |
| 自定义配置 | `scripts/run/arenas/retro_mario/run.sh --config` | 跳过脚本内置映射，直接使用指定 YAML |
| 模型接口地址 | 当前所选 `retro_mario_openai_*.yaml` 里的 `backends[].config.base_url` | 在托管 API 和本地 OpenAI 兼容服务之间切换 |
| 模型名 | `RETRO_OPENAI_MODEL` 或 `backends[].config.model` | 选择 OpenAI 模型 |
| API Key 强制校验 | `backends[].config.require_api_key` | 托管 API 建议保持 `true`；只有可信本地服务才建议关闭 |
| 实时查看端口 | `human_input.ws_port` 或环境变量 `RETRO_WS_RGB_PORT` | websocket 配置的 websocketRGB 端口；脚本会透传这个环境变量 |
| 显示模式 | `environment.display_mode` | `websocket` 用于实时查看，`headless` 用于离线运行 |
| 合法动作 | `environment.legal_moves` | 暴露给玩家的动作集合 |
| 持续帧数 | `environment.action_schema.hold_ticks_*` 和 `parser.hold_ticks_*` | 宏动作持续多久，以及默认值/上下界 |
| Tick 节奏 | `scheduler.tick_ms` | 主调度器的毫秒级节奏 |
| Human 输入 FPS | `human_input.fps` | human 模式下输入采样频率 |
| Replay 模式 | `environment.replay.mode` | 决定写出 `action`、`frame` 或 `both` |
| Replay 帧采样 | `environment.replay.frame_capture.*` | 控制帧采样步长、格式和帧数上限 |
| 回放 FPS | `scripts/run/arenas/retro_mario/replay.sh` 的环境变量 `FPS` | 跑后回放的播放速度 |
| 回放地址 | `scripts/run/arenas/retro_mario/replay.sh` 的环境变量 `HOST` / `PORT` | 回放服务监听地址 |
| 回放帧数上限 | `scripts/run/arenas/retro_mario/replay.sh` 的环境变量 `MAX_FRAMES` | 限制回放最大帧数 |

补充说明：

- Mario 动作是 JSON，例如 `{"move":"right_run_jump","hold_ticks":6}`。
- 如果模型没有输出 `hold_ticks`，解析器会补默认值，并限制在配置范围内。
- `display_mode: websocket` 在运行时会被当成 headless 环境加 websocketRGB 推流，而不是本地窗口渲染。

## 7. 产物与回放

运行产物默认写到：

```text
runs/<run_id>/
```

可用单个 sample 产物启动回放：

```bash
bash scripts/run/arenas/retro_mario/replay.sh <run_id>
```

常用回放环境变量：

- `PYTHON_BIN`：回放脚本使用的 Python 解释器
- `HOST`：监听地址，默认 `127.0.0.1`
- `PORT`：监听端口，默认 `5800`
- `FPS`：播放 FPS，默认 `12`
- `MAX_FRAMES`：最大回放帧数，默认 `0` 表示不限制
- `AUTO_OPEN`：设为 `1` 时自动打开浏览器

回放地址：

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

启动回放前，建议先确认 sample JSON 里存在 `replay_path` 等 replay 相关字段。


