# PettingZoo Game Arena 指南

[English](game_arena_pettingzoo.md) | 中文

这是当前仓库里 PettingZoo Atari 的标准 Game Arena 文档，统一整理了安装、启动 Demo、回放和关键参数修改位置。

## 1. 概览

PettingZoo Atari 目前主要覆盖三类启动路径：

- Dummy 路径，用于验证环境和 viewer
- LLM 对战路径，用于 API 驱动评测
- Human vs Human record 路径，用于浏览器交互

本指南是标准入口。旧的 PettingZoo 文档先保留，但不再作为主入口。

## 2. 标准入口文件

| 类型 | 路径 | 用途 |
| --- | --- | --- |
| 一键脚本 | `scripts/oneclick/run_pettingzoo_ws_rgb_viewer.sh` | 启动 ws_rgb Demo，并在可用时自动打开 viewer |
| 配置目录 | `config/custom/pettingzoo/` | 所有 PettingZoo 游戏配置都在这里 |
| 推荐 dummy 配置 | `config/custom/pettingzoo/pong_dummy_ws_rgb.yaml` | 最快的 ws_rgb 冒烟验证 |
| 推荐 AI 配置 | `config/custom/pettingzoo/space_invaders_ai.yaml` | 标准 LLM Demo |
| Human record 配置 | `config/custom/pettingzoo/space_invaders_human_vs_human_record.yaml` | 浏览器输入的人类对战 |
| 补充命令索引 | `docs/guide/pettingzoo_atari_run_commands.md` | 按游戏列出的完整 AI / Dummy 启动命令 |
| 回放工具 | `src/gage_eval/tools/ws_rgb_replay.py` | 基于 sample 产物启动回放 |

## 3. 前置准备

安装依赖和运行 `run.py` 时要使用同一个 Python 解释器。

```bash
PYTHON_BIN="${PYTHON_BIN:-$(command -v python3)}"
echo "PYTHON_BIN=$PYTHON_BIN"
"$PYTHON_BIN" -m pip install -U \
  "pettingzoo[atari]>=1.24.3" \
  "shimmy[atari]>=1.0.0" \
  "AutoROM[accept-rom-license]>=0.6.1"
"$PYTHON_BIN" -m AutoROM.AutoROM --accept-license
```

建议安装后立刻做一次 ROM 校验：

```bash
"$PYTHON_BIN" - <<'PY'
from pettingzoo.atari import pong_v3

env = pong_v3.env(render_mode="rgb_array")
env.reset(seed=0)
print("PettingZoo Atari ROM check: OK")
env.close()
PY
```

## 4. 启动路径

### 4.1 推荐冒烟路径：一键 ws_rgb dummy 运行

```bash
CONFIG=config/custom/pettingzoo/pong_dummy_ws_rgb.yaml \
RUN_ID="pz_pong_dummy_ws_rgb_$(date +%Y%m%d_%H%M%S)" \
bash scripts/oneclick/run_pettingzoo_ws_rgb_viewer.sh
```

这个脚本的执行顺序是：

1. 选择 Python 解释器。
2. 校验配置文件路径。
3. 选择一个空闲的 `WS_RGB_PORT`。
4. 执行 `python run.py --config ...`。
5. 等待 `/ws_rgb/viewer` 就绪，并在桌面环境可用时自动打开。

常用脚本环境变量：

- `CONFIG`：要运行的配置文件
- `PYTHON_BIN`：Python 解释器
- `RUN_ID`：写入 `runs/` 的运行编号
- `OUTPUT_DIR`：输出目录
- `WS_RGB_PORT`：期望的 viewer 端口，若被占用会自动顺延
- `AUTO_OPEN`：设为 `0` 时不自动打开浏览器
- `WAIT_TIMEOUT_S`：等待 viewer 就绪的超时时间

### 4.2 LLM 对战路径

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
export GAME="space_invaders"
export RUN_ID="pz_${GAME}_ai_$(date +%Y%m%d_%H%M%S)"

python run.py \
  --config "config/custom/pettingzoo/${GAME}_ai.yaml" \
  --output-dir runs \
  --run-id "$RUN_ID"
```

如果你后面要基于 sample 产物做回放，建议在运行前加上：

```bash
export GAGE_EVAL_GAME_LOG_INLINE_LIMIT=-1
export GAGE_EVAL_GAME_LOG_INLINE_BYTES=0
```

### 4.3 Human vs Human record 路径

```bash
PYTHONPATH=src python run.py \
  --config config/custom/pettingzoo/space_invaders_human_vs_human_record.yaml \
  --output-dir runs \
  --run-id pz_space_invaders_h2h_record
```

默认 viewer 地址：

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

`space_invaders_human_vs_human_record.yaml` 里当前默认键位：

- `player_0`：`Q/W/E/A/S/D`
- `player_1`：`U/I/O/J/K/L`

## 5. 启动顺序

当前仓库里 PettingZoo Demo 的推荐顺序是：

1. 安装 PettingZoo Atari 依赖和 ROM。
2. 从 `config/custom/pettingzoo/` 里选一个配置。
3. 只有在使用 LLM 配置时才设置 API Key。
4. 用 `run_pettingzoo_ws_rgb_viewer.sh` 或 `python run.py` 启动。
5. 对 `display_mode: websocket` 的配置，在运行中打开 viewer；否则运行完成后再用 `ws_rgb_replay` 回放。

## 6. 关键参数与修改位置

| 项目 | 修改位置 | 含义 |
| --- | --- | --- |
| API Key | Shell 环境变量 `OPENAI_API_KEY` | `*_ai.yaml` 里的 OpenAI 后端会读取它 |
| 游戏选择 | 配置文件名和 `environment.env_id` | 决定具体 Atari 游戏和运行时 env id |
| 运行时长 | `environment.env_kwargs.max_cycles` | 环境层的帧数上限 |
| Arena 回合上限 | `scheduler.max_turns` | Arena 层最大回合数 |
| Viewer 模式 | `environment.display_mode` | 设为 `websocket` 时启用 ws_rgb 实时查看 |
| Viewer 地址 | `human_input.ws_host` / `human_input.ws_port` | 实时 viewer 的绑定地址 |
| 回放 FPS | `python -m gage_eval.tools.ws_rgb_replay --fps` | 跑后回放的播放速度 |
| 动作显示方式 | `environment.use_action_meanings` | `true` 显示 `FIRE` 等动作名，`false` 使用离散数字 |
| 人类键位映射 | `environment.action_schema.key_map` | record 模式下浏览器按键到动作的映射 |
| 人类输入端口 | `human_input.host` / `human_input.port` | record 模式下动作队列服务地址 |

补充说明：

- 大多数游戏配置使用 `max_cycles: 300`，`space_invaders_*` 使用 `3000`。
- 一键脚本只负责 viewer 端口和启动顺序，游戏行为本身仍由 YAML 决定。

## 7. 产物与回放

运行产物默认写到：

```text
runs/<run_id>/
```

跑后回放命令：

```bash
export RUN_ID="<your_run_id>"
export GAGE_EVAL_GAME_LOG_INLINE_LIMIT=-1
export GAGE_EVAL_GAME_LOG_INLINE_BYTES=0
SAMPLE_JSON="$(find "runs/${RUN_ID}/samples" -type f -name '*.json' | head -n 1)"

PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo
```

默认回放地址：

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

## 8. 其他文档

下面这些文档仍保留在仓库里：

- `docs/guide/pettingzoo_user_guide.md`
- `docs/guide/pettingzoo_user_guide_zh.md`
- `docs/guide/pettingzoo_atari_run_commands.md`
- `docs/guide/pettingzoo_atari_run_commands_zh.md`
