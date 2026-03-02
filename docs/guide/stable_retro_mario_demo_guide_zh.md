# Stable-Retro Mario YAML 配置指南（仓库配置视角）

本文只总结当前仓库里的 **YAML 配置方式**，不依赖本地运行产物。

## 1. 前期准备

​	从https://archive.org/download/super-mario-bros-3-nes/Super%20Mario%20Bros.%203.nes
​	下载游戏rom，记录保存的文件夹
​	安装stable retro

​	执行
​	python -m retro.import "rom保存路径"

## 2. Mario 相关 YAML（当前存在）

配置文件：

- `config/custom/retro_mario_openai_headless_auto_eval.yaml`
- `config/custom/retro_mario_openai_ws_rgb_auto_eval.yaml`
- `config/custom/retro_mario_phase1_dummy_headless_auto_eval.yaml`
- `config/custom/retro_mario_phase1_dummy_ws.yaml`
- `config/custom/retro_mario_phase1_human_ws.yaml`

## 3. 通用 YAML 结构（Retro Mario）

所有 Mario 配置都遵循这个主干：

```yaml
custom:
  steps:
    - step: arena
      adapter_id: retro_mario_arena
    - step: auto_eval

datasets:
  - dataset_id: retro_mario_demo
    loader: jsonl
    params:
      path: config/custom/retro_mario_phase1.jsonl

role_adapters:
  - adapter_id: retro_mario_arena
    role_type: arena
    params:
      environment: ...
      scheduler: ...
      parser: ...
      human_input: ...   # 仅 ws/human 场景需要
      players: ...

tasks:
  - task_id: ...
    dataset_id: ...
```

## 4. 核心配置项怎么配

## 4.1 `environment`（游戏环境）

必备字段（Mario）：

```yaml
environment:
  impl: retro_env_v1
  game: SuperMarioBros3-Nes-v0
  state: Start
  display_mode: headless | websocket
  legal_moves:
    - noop
    - right
    - right_run
    - right_jump
    - right_run_jump
  action_schema:
    hold_ticks_min: 1
    hold_ticks_max: 12
    hold_ticks_default: 6
  info_feeder:
    impl: info_last_v1
```

说明：

- `display_mode=headless`：不注册 ws_rgb 显示
- `display_mode=websocket`：会走 ws_rgb hub（由 `arena.py` 决定）

## 4.2 `scheduler`（调度方式）

Mario 现有demo用了两类：

- `tick`：按固定间隔思考/执行
- `record`：更偏“实时录制节奏”（human demo 用）

示例：

```yaml
scheduler:
  type: tick
  tick_ms: 100
  max_ticks: 3600
```

或：

```yaml
scheduler:
  type: record
  tick_ms: 16
  max_ticks: 3600
```

## 4.3 `parser`（动作解析）

Mario 使用：

```yaml
parser:
  impl: retro_action_v1
  hold_ticks_min: 1
  hold_ticks_max: 12
  hold_ticks_default: 6
```

## 4.4 `human_input`（ws/human 相关）

### 仅 ws_rgb 显示（AI/Dummy）

```yaml
human_input:
  enabled: true
  ws_host: ${RETRO_WS_RGB_HOST:-0.0.0.0}
  ws_port: ${RETRO_WS_RGB_PORT:-5800}
  ws_allow_origin: "*"
```

### 人类输入（record scheduler）

```yaml
human_input:
  enabled: true
  host: 0.0.0.0
  port: 8001
  fps: 30
  hold_ticks_default: 6
```

说明（代码行为）：

- `host/port` 是 action queue server（人类动作入口）
- `ws_host/ws_port` 是 ws_rgb viewer 服务地址
- 若没配 `ws_host/ws_port`，ws hub 会回退到 `host` 和默认端口 `5800`

## 4.5 `players`（玩家来源）

### Dummy

```yaml
players:
  - name: player_0
    type: backend
    ref: retro_dummy_player
```

### OpenAI

```yaml
players:
  - name: player_0
    type: backend
    ref: retro_openai_player_0
    max_retries: 1
    fallback_policy: first_legal
```

### Human

```yaml
players:
  - name: player_0
    type: human
    ref: retro_websocket_human
```

## 4.6 `environment.replay`（回放产物）

推荐统一写法：

```yaml
environment:
  replay:
    enabled: true
    mode: both         # action | frame | both
    primary_mode: true # 可选，是否把 replay_path 作为主回放路径
    frame_capture:
      enabled: true
      frame_stride: 3
      max_frames: 900
      format: jpeg
      quality: 75
```

关键点：

- `enabled: true` 才会写 replay 产物
- `mode: action` 只记录 action 事件
- `mode: frame` 或 `both` 才有 frame 事件
- 想稳定用于 `ws_rgb_replay`，优先 `mode: both`

## 5. 五个 Mario 配置的“当前差异”

## 5.1 Headless + OpenAI

文件：`retro_mario_openai_headless_auto_eval.yaml`

- `display_mode: headless`
- `scheduler: tick (1200/120)`
- 无 `human_input`
- 无 `environment.replay`

## 5.2 Headless + Dummy

文件：`retro_mario_phase1_dummy_headless_auto_eval.yaml`

- `display_mode: headless`
- `scheduler: tick (100/3600)`
- 无 `human_input`
- 无 `environment.replay`

## 5.3 ws_rgb + OpenAI

文件：`retro_mario_openai_ws_rgb_auto_eval.yaml`

- `display_mode: websocket`
- `obs_image: true`
- `human_input.ws_host/ws_port` 已配置
- `replay.enabled: true`
- 当前 `replay.mode: action`

## 5.4 ws_rgb + Dummy

文件：`retro_mario_phase1_dummy_ws.yaml`

- `display_mode: websocket`
- `obs_image: true`
- `human_input.ws_host/ws_port` 已配置
- `replay.enabled: true`
- 当前 `replay.mode: action`

## 5.5 ws_rgb + Human（record）

文件：`retro_mario_phase1_human_ws.yaml`

- `display_mode: websocket`
- `scheduler.type: record`
- `human_input.host/port/fps` 已配置（`8001`）
- `replay.enabled: true`
- 当前 `replay.mode: both`，`primary_mode: true`

## 6. 直接可复用的 YAML 模板

## 6.1 模板 A：可视化 + 回放友好（推荐）

```yaml
params:
  human_input:
    enabled: true
    ws_host: ${RETRO_WS_RGB_HOST:-0.0.0.0}
    ws_port: ${RETRO_WS_RGB_PORT:-5800}
    ws_allow_origin: "*"
  environment:
    impl: retro_env_v1
    game: SuperMarioBros3-Nes-v0
    state: Start
    display_mode: websocket
    obs_image: true
    legal_moves: [noop, right, right_run, right_jump, right_run_jump]
    action_schema:
      hold_ticks_min: 1
      hold_ticks_max: 12
      hold_ticks_default: 6
    info_feeder:
      impl: info_last_v1
    replay:
      enabled: true
      mode: both
      frame_capture:
        enabled: true
        frame_stride: 3
        max_frames: 900
        format: jpeg
        quality: 75
```

## 6.2 模板 B：纯 headless 评估

```yaml
params:
  environment:
    impl: retro_env_v1
    game: SuperMarioBros3-Nes-v0
    state: Start
    display_mode: headless
    obs_image: false
    legal_moves: [noop, right, right_run, right_jump, right_run_jump]
    action_schema:
      hold_ticks_min: 1
      hold_ticks_max: 12
      hold_ticks_default: 6
    info_feeder:
      impl: info_last_v1
  scheduler:
    type: tick
    tick_ms: 100
    max_ticks: 3600
```

## 7. 启动命令（只保留配置驱动，不依赖历史产物）

```bash
env PYTHONPATH=src .venv/bin/python run.py \
  --config <one_of_retro_mario_yaml> \
  --output-dir runs \
  --run-id <your_run_id>
```

如果是 OpenAI 配置，先设：

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
```

## 8. 实时查看与回放链接

## 8.1 实时查看（ws_rgb）

适用配置：

- `retro_mario_openai_ws_rgb_auto_eval.yaml`
- `retro_mario_phase1_dummy_ws.yaml`
- `retro_mario_phase1_human_ws.yaml`

默认链接（端口默认 `5800`）：

- 本机：`http://127.0.0.1:5800/ws_rgb/viewer`
- 宿主机访问容器：`http://localhost:5800/ws_rgb/viewer`

如果你通过环境变量改了端口（`RETRO_WS_RGB_PORT`），把链接里的 `5800` 改成对应端口。

可直接打印当前链接（按环境变量自动替换端口）：

```bash
WS_PORT="${RETRO_WS_RGB_PORT:-5800}"
echo "实时查看(本机): http://127.0.0.1:${WS_PORT}/ws_rgb/viewer"
echo "实时查看(宿主机): http://localhost:${WS_PORT}/ws_rgb/viewer"
```

## 8.2 回放（ws_rgb_replay）

回放服务启动示例（`SAMPLE_JSON` 用本次 run 产物替换）：

```bash
REPLAY_PORT="${RETRO_REPLAY_PORT:-5800}"

env PYTHONPATH=src .venv/bin/python -m gage_eval.tools.ws_rgb_replay \
  --sample-json <SAMPLE_JSON> \
  --host 0.0.0.0 \
  --port "${REPLAY_PORT}" \
  --fps 12
```

回放页面链接：

- 本机：`http://127.0.0.1:5800/ws_rgb/viewer`
- 宿主机访问容器：`http://localhost:5800/ws_rgb/viewer`

可直接打印当前回放链接（按环境变量自动替换端口）：

```bash
REPLAY_PORT="${RETRO_REPLAY_PORT:-5800}"
echo "回放(本机): http://127.0.0.1:${REPLAY_PORT}/ws_rgb/viewer"
echo "回放(宿主机): http://localhost:${REPLAY_PORT}/ws_rgb/viewer"
```

建议先在样本 JSON 里确认有 `replay_path` 字段，再启动回放。
