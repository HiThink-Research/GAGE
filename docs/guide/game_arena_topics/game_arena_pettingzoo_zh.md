# PettingZoo Game Arena 指南

[English](game_arena_pettingzoo.md) | 中文

这是当前仓库里 PettingZoo Atari 的标准 Game Arena 文档，统一整理了安装、脚本化启动 Demo、回放和关键参数修改位置。

## 1. 概览

PettingZoo Atari 目前主要覆盖三类启动路径：

- LLM 对战 + websocketRGB 路径，用于浏览器实时查看
- 标准 LLM 路径，用于 API 驱动评测
- Human vs Human record 路径，用于浏览器交互

这三类路径现在都统一收口到脚本入口。本指南是标准入口。旧的 PettingZoo 文档先保留，但不再作为主入口。

## 2. 标准入口文件

| 类型 | 路径 | 用途 |
| --- | --- | --- |
| 标准启动脚本 | `scripts/run/arenas/pettingzoo/run.sh` | PettingZoo 的主启动入口，支持按游戏启动 Dummy、AI、websocketRGB Dummy 和 human record |
| 回放脚本 | `scripts/run/arenas/pettingzoo/replay.sh` | 通过 `run_id` 回放一局已完成对局 |
| websocketRGB 辅助脚本 | `scripts/run/arenas/pettingzoo/viewer.sh` | 通用 websocketRGB helper，会等待 viewer 可访问后再继续 |
| 配置目录 | `config/custom/pettingzoo/` | 所有 PettingZoo 游戏配置都在这里 |
| 推荐 AI websocketRGB 配置 | `config/custom/pettingzoo/space_invaders_ai_ws_rgb.yaml` | 推荐的 LLM vs LLM + 浏览器实时查看示例 |
| LiteLLM 本地模型示例 | `config/custom/pettingzoo/space_invaders_litellm_ai.yaml` | 用于本地或自建 OpenAI 兼容服务的示例配置 |
| 标准 AI 配置 | `config/custom/pettingzoo/space_invaders_ai.yaml` | 不带 websocketRGB 实时查看的标准 LLM Demo |
| Human record 配置 | `config/custom/pettingzoo/space_invaders_human_vs_human_record.yaml` | 浏览器输入的人类对战 |
| 补充命令索引 | `docs/guide/game_arena_topics/pettingzoo_atari_run_commands.md` | 按游戏列出的完整 AI / Dummy 脚本启动命令 |
| 回放工具 | `src/gage_eval/tools/ws_rgb_replay.py` | 回放脚本底层使用的 replay server |

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
from pettingzoo.atari import space_invaders_v2

env = space_invaders_v2.env(render_mode="rgb_array")
env.reset(seed=0)
print("PettingZoo Atari ROM check: OK")
env.close()
PY
```

## 4. 启动路径

### 4.1 推荐冒烟路径：dummy + websocketRGB 实时查看

```bash
bash scripts/run/arenas/pettingzoo/run.sh \
  --game space_invaders \
  --mode ws_dummy \
  --run-id "pettingzoo_space_invaders_ws_dummy_$(date +%Y%m%d_%H%M%S)"
```

建议先用这条命令验证 PettingZoo Atari、ROM 安装和 websocketRGB 实时查看链路，再切到模型模式。

这个模式下启动脚本的执行顺序：

1. 选择 Python 解释器。
2. 根据 `--game` 和 `--mode` 解析 YAML 配置。
3. 校验配置文件路径。
4. 设置内联 game log 等公共运行默认值。
5. 执行 `python run.py --config ...`。
6. 对 websocket 模式打印 viewer 地址。

### 4.2 推荐模型示例：LLM vs LLM + websocketRGB 实时查看

```bash
OPENAI_API_KEY="<YOUR_KEY>" \
RUN_ID="pettingzoo_space_invaders_ai_ws_rgb_$(date +%Y%m%d_%H%M%S)" \
CONFIG=config/custom/pettingzoo/space_invaders_ai_ws_rgb.yaml \
bash scripts/run/arenas/pettingzoo/viewer.sh
```

如果你的目标是“AI 在玩游戏，同时浏览器里能看到实时画面”，文档里推荐的就是这条命令。

这个 websocketRGB helper 的执行顺序：

1. 选择 Python 解释器。
2. 校验配置文件路径。
3. 选择一个可用的 `WS_RGB_PORT`。
4. 在后台启动 `python run.py --config ...`。
5. 等待 `http://127.0.0.1:<port>/ws_rgb/viewer` 可访问。
6. 打印可用的 viewer 地址，并按配置决定是否自动打开浏览器。

这个命令里如果要切换模型或 API，请改这里：

- API Key：启动前先在 shell 里设置 `OPENAI_API_KEY`。`viewer.sh` 和 `run.sh` 也接受 `LITELLM_API_KEY`，并会统一成 `OPENAI_API_KEY`。
- 托管 OpenAI 兼容 API：编辑 `config/custom/pettingzoo/space_invaders_ai_ws_rgb.yaml` 里的 `backends[0].config`。切换接口地址时改 `base_url`，切换模型时改 `model`。如果你用的是普通 `--mode ai` 路径，对应的是 `config/custom/pettingzoo/<game>_ai.yaml` 里的同一组字段。
- 本地 OpenAI 兼容服务：把 `base_url` 改成本地服务地址，再把 `model` 改成服务实际暴露的模型名。当前这组 `openai_http` 配置里 `require_api_key: true`，所以要么保持 `OPENAI_API_KEY` 非空，要么在你确认本地服务不需要鉴权时把这个开关改成 `false`。
- 现成的本地模型示例：`config/custom/pettingzoo/space_invaders_litellm_ai.yaml` 已经提供了 `backends[].config.api_base` 和 `backends[].config.model`，也支持用环境变量 `PZ_LITELLM_API_BASE`、`PZ_LITELLM_MODEL` 覆盖。

### 4.3 标准 LLM 对战路径

```bash
export OPENAI_API_KEY="<YOUR_KEY>"

bash scripts/run/arenas/pettingzoo/run.sh \
  --game space_invaders \
  --mode ai \
  --run-id "pettingzoo_space_invaders_ai_$(date +%Y%m%d_%H%M%S)"
```

`scripts/run/arenas/pettingzoo/run.sh` 当前支持的模式：

- `dummy`：映射到 `<game>_dummy.yaml`
- `ai`：映射到 `<game>_ai.yaml`
- `ws_dummy`：映射到 `<game>_dummy_ws_rgb.yaml`
- `human_record`：当前固定映射到 `space_invaders_human_vs_human_record.yaml`

常用启动参数：

- `--game`：PettingZoo 游戏 id，例如 `space_invaders`、`boxing`、`pong`
- `--mode`：`dummy`、`ai`、`ws_dummy`、`human_record`
- `--config`：显式 YAML 路径，会覆盖 `--game` 和 `--mode` 的映射
- `--run-id`：写入 `runs/` 的运行编号
- `--output-dir`：输出目录，默认 `runs`
- `--python-bin`：指定 Python 解释器

常用环境变量：

- `WS_RGB_PORT`：websocket 模式下的 viewer 端口
- `OPENAI_API_KEY`：`ai` 模式必需
- `LITELLM_API_KEY`：可作为 `OPENAI_API_KEY` 的回退来源
- `GAGE_EVAL_GAME_LOG_INLINE_LIMIT` 和 `GAGE_EVAL_GAME_LOG_INLINE_BYTES`：脚本默认会自动设置，除非你手动覆盖

如果你的 key 放在 `LITELLM_API_KEY`，脚本也会在 `ai` 模式下自动复用。

### 4.4 Human vs Human record 路径

```bash
bash scripts/run/arenas/pettingzoo/run.sh \
  --game space_invaders \
  --mode human_record \
  --run-id "pettingzoo_space_invaders_human_record_$(date +%Y%m%d_%H%M%S)"
```

脚本会打印的默认运行地址：

- Viewer：`http://127.0.0.1:5800/ws_rgb/viewer`
- Input queue：`http://127.0.0.1:8001`

`space_invaders_human_vs_human_record.yaml` 里当前默认键位：

- `player_0`：`Q/W/E/A/S/D`
- `player_1`：`U/I/O/J/K/L`

### 4.5 显式配置覆盖

如果你想保留脚本入口，但指定自定义 YAML，可以直接用 `--config`：

```bash
bash scripts/run/arenas/pettingzoo/run.sh \
  --config config/custom/pettingzoo/boxing_dummy.yaml \
  --run-id "pettingzoo_boxing_custom_$(date +%Y%m%d_%H%M%S)"
```

## 5. 启动顺序

当前仓库里 PettingZoo Demo 的推荐顺序是：

1. 安装 PettingZoo Atari 依赖和 ROM。
2. 决定要跑的游戏和模式，或者准备一个自定义配置路径。
3. 只有在使用 `--mode ai` 时才设置 API Key。
4. 先用 dummy websocket 路径验证环境，再切到 `scripts/run/arenas/pettingzoo/viewer.sh` 或 `scripts/run/arenas/pettingzoo/run.sh` 执行模型模式。
5. 对 `human_record`，在运行中打开 viewer；AI websocketRGB helper 会自动等待 viewer 就绪。
6. 如果要跑后回放，再执行 `scripts/run/arenas/pettingzoo/replay.sh <run_id>`。

## 6. 关键参数与修改位置

| 项目 | 修改位置 | 含义 |
| --- | --- | --- |
| API Key | Shell 环境变量 `OPENAI_API_KEY` 或 `LITELLM_API_KEY` | `--mode ai` 必需；脚本会统一成 `OPENAI_API_KEY` |
| 游戏选择 | `scripts/run/arenas/pettingzoo/run.sh --game` 或配置里的 `environment.env_id` | 决定具体 Atari 游戏和运行时 env id |
| 启动模式 | `scripts/run/arenas/pettingzoo/run.sh --mode` | 选择 `dummy`、`ai`、`ws_dummy`、`human_record` |
| 自定义配置 | `scripts/run/arenas/pettingzoo/run.sh --config` | 跳过脚本内置映射，直接使用指定 YAML |
| 模型接口地址 | `*_ai.yaml` 里的 `backends[].config.base_url`，或 `space_invaders_litellm_ai.yaml` 里的 `backends[].config.api_base` | 在托管 API 和本地 OpenAI 兼容服务之间切换 |
| 模型名 | `backends[].config.model`，或 LiteLLM 示例里的环境变量 `PZ_LITELLM_MODEL` | 选择 AI 玩家实际使用的模型 |
| API Key 强制校验 | `*_ai.yaml` 里的 `backends[].config.require_api_key` | 托管 API 建议保持 `true`；只有可信本地服务才建议关闭 |
| 运行时长 | `environment.env_kwargs.max_cycles` | 环境层的帧数上限 |
| Arena 回合上限 | `scheduler.max_turns` | Arena 层最大回合数 |
| Viewer 模式 | `environment.display_mode` | 设为 `websocket` 时启用 websocketRGB 实时查看 |
| Viewer 地址 | `human_input.ws_host` / `human_input.ws_port` 和环境变量 `WS_RGB_PORT` | 实时 viewer 绑定地址；脚本会透传 `WS_RGB_PORT` |
| 回放 FPS | `scripts/run/arenas/pettingzoo/replay.sh` 的环境变量 `FPS` | 跑后回放的播放速度 |
| 回放地址 | `scripts/run/arenas/pettingzoo/replay.sh` 的环境变量 `HOST` / `PORT` | 回放服务监听地址 |
| 回放帧数上限 | `scripts/run/arenas/pettingzoo/replay.sh` 的环境变量 `MAX_FRAMES` | 限制回放最大帧数 |
| 动作显示方式 | `environment.use_action_meanings` | `true` 显示 `FIRE` 等动作名，`false` 使用离散数字 |
| 人类键位映射 | `environment.action_schema.key_map` | record 模式下浏览器按键到动作的映射 |
| 人类输入端口 | `human_input.host` / `human_input.port` | record 模式下动作队列服务地址 |
| 内联回放日志 | 环境变量 `GAGE_EVAL_GAME_LOG_INLINE_LIMIT` 和 `GAGE_EVAL_GAME_LOG_INLINE_BYTES` | 把 game log 保留在 sample JSON 里，避免回放时需要补跑 |

补充说明：

- 大多数游戏配置使用 `max_cycles: 300`，`space_invaders_*` 使用 `3000`。
- 启动脚本只负责配置映射和公共运行默认值，游戏行为本身仍由 YAML 决定。

## 7. 产物与回放

运行产物默认写到：

```text
runs/<run_id>/
```

跑后回放命令：

```bash
bash scripts/run/arenas/pettingzoo/replay.sh <run_id>
```

常用回放环境变量：

- `PYTHON_BIN`：回放脚本使用的 Python 解释器
- `HOST`：监听地址，默认 `127.0.0.1`
- `PORT`：监听端口，默认 `5800`
- `FPS`：播放 FPS，默认 `12`
- `MAX_FRAMES`：最大回放帧数，默认 `0` 表示不限制
- `AUTO_OPEN`：设为 `1` 时自动打开浏览器

默认回放地址：

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

## 8. 其他文档

下面这些文档仍保留在仓库里：

- `docs/guide/pettingzoo_user_guide.md`
- `docs/guide/pettingzoo_user_guide_zh.md`
- `docs/guide/game_arena_topics/pettingzoo_atari_run_commands.md`
- `docs/guide/game_arena_topics/pettingzoo_atari_run_commands_zh.md`
