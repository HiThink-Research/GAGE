# PettingZoo Atari 使用说明书 (User Guide)

本文档是 **PettingZoo Atari** 游戏在 GAGE 框架下的标准使用指南。
无论您是想运行 AI 对战，还是测试环境配置，或者是回放录像，都可以参考本手册。

## 1. 快速开始 (Quick Start)

我们推荐使用 **"运行即回放" (Run & Replay)** 的自动化流程。
这不仅能让您看到 AI 的决策过程，还能通过 `ws_rgb` 网页端直观查看对战画面。

### 示例：运行 Space Invaders

复制并执行以下命令：

```bash
# 1. 设置游戏名称 (支持 22 款游戏，见后文列表)
export GAME="space_invaders"
export RUN_ID="pz_${GAME}_auto_$(date +%s)"

# 2. 启动 AI 对战
python run.py \
  --config config/custom/pettingzoo/${GAME}_ai.yaml \
  --output-dir runs \
  --run-id "$RUN_ID"

# 3. 启动网页回放服务（ws_rgb 新链路，默认自动弹出浏览器）
SAMPLE_JSON="$(find runs/$RUN_ID/samples -name "*.json" | head -n 1)"
PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo \
  --auto-open 1

# 4. 如果未自动弹出，手动打开
echo "http://127.0.0.1:5800/ws_rgb/viewer"
```
*(注：`--auto-open 1` 依赖默认浏览器环境；若无图形环境请用手动地址打开。)*

---

## 2. 支持的游戏列表 (Supported Games)

GAGE 目前已完美适配以下 **22 款双人 Atari 游戏**：

| Game ID | Description |
| :--- | :--- |
| `basketball_pong` | 投篮乒乓球 |
| `boxing` | 拳击 (经典对战) |
| `combat_plane` | 空战 |
| `combat_tank` | 坦克大战 |
| `double_dunk` | 双人篮球 |
| `entombed_competitive` | 迷宫竞赛 |
| `entombed_cooperative` | 迷宫合作 |
| `flag_capture` | 夺旗 |
| `foozpong` | 桌上足球 |
| `ice_hockey` | 冰球 |
| `joust` | 鸵鸟骑士 |
| `mario_bros` | 马里奥兄弟 (水管工) |
| `maze_craze` | 疯狂迷宫 |
| `othello` | 黑白棋 (Reversi) |
| `pong` | 乒乓球 (Atari 鼻祖) |
| **`space_invaders`** | **太空侵略者 (推荐演示)** |
| `space_war` | 太空大战 |
| `surround` | 围困 (贪吃蛇对战) |
| `tennis` | 网球 |
| `video_checkers` | 跳棋 |
| `volleyball_pong` | 排球 |
| `wizard_of_wor` | 巫师瓦尔 |

要运行其他游戏，只需将环境变量修改为对应 ID 即可：
`export GAME="boxing"`

> [!TIP]
> 完整的所有游戏启动命令（Dummy 和 AI 模式），请查阅：
> 🔗 [**PettingZoo Atari Run Commands**](./pettingzoo_atari_run_commands.md)

---

## 3. 运行模式 (Running Modes)

我们提供两种运行配置：**Dummy** (测试用) 和 **AI** (实战用)。

### 3.1 AI 模式 (需要 API Key)

*   **配置路径**: `config/custom/pettingzoo/<game>_ai.yaml`
*   **用途**: 调用 OpenAI API (如 gpt-4o, o1-preview) 进行真实对战。
*   **命令**:
    ```bash
    export OPENAI_API_KEY="sk-..."
    python run.py --config config/custom/pettingzoo/boxing_ai.yaml ...
    ```

### 3.2 Dummy 模式 (无需 API Key)

*   **配置路径**: `config/custom/pettingzoo/<game>_dummy.yaml`
*   **用途**: 快速测试环境安装、验证渲染、调试脚本。
*   **策略**: 使用 **Random Policy**（随机合法动作），保证游戏能一直运行下去。
*   **命令**:
    ```bash
    python run.py --config config/custom/pettingzoo/boxing_dummy.yaml ...
    ```

---

## 4. 网页回放工具 (ws_rgb Viewer)

`scripts/replay_pettingzoo.py` 属于旧版回放路径，已不再推荐。
当前统一使用 `ws_rgb` 网页查看链路。

### 4.1 跑后回放（基于已有产物）

```bash
RUN_ID=<your_run_id>
SAMPLE_JSON="$(find runs/$RUN_ID/samples -name "*.json" | head -n 1)"

PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo \
  --auto-open 1
```

网页地址：
`http://127.0.0.1:5800/ws_rgb/viewer`

### 4.2 一键启动并自动打开网页（实时查看）

```bash
AUTO_OPEN=1 \
CONFIG=config/custom/pettingzoo/pong_dummy_ws_rgb.yaml \
RUN_ID="pz_pong_ws_$(date +%Y%m%d_%H%M%S)" \
bash scripts/oneclick/run_pettingzoo_ws_rgb_viewer.sh
```

---

## 5. 常见问题 (Troubleshooting)

### Q1: 为什么回放刚开始时，飞船/小人不动？
**A**: 这是 **Warm-up Phase (热身阶段)**。
Atari 游戏启动时通常有几秒钟的过场动画（如闪烁的 "INSERT COINS" 或 "PLAYER 1"）。在此期间，游戏引擎会锁住所有输入。我们的 AI 虽然会尝试操作，但在这一阶段会被忽略。等到画面一闪，游戏正式开始，操作就会生效。

### Q2: 既然 AI 是盲玩的，它怎么知道该干什么？
**A**: AI 看到的是**文本状态 (Text Observation)**，例如：
`Status: Legal moves: [FIRE, LEFT, RIGHT]`.
它不需要像素画面，只需要根据当前合法动作列表和历史信息进行推理。并且后续会支持图片输入。

### Q3: 如何修改游戏时长？
**A**: 修改 `config/custom/pettingzoo/<game>_ai.yaml` 中的 `max_cycles` 和 `max_turns` 参数。目前默认为 3000 帧（约 50秒）。

---

## 6. Space Invaders 双人 Human 操作说明（Record Scheduler）

本节对应配置文件：
`config/custom/pettingzoo/space_invaders_human_vs_human_record.yaml`

开发者链路文档（架构/时序/排障）：
`docs/guide/ws_rgb_runtime_dev_guide_zh.md`

### 6.1 启动对局

```bash
cd /Users/shuo/code/GAGE
PYTHONPATH=src ./.venv/bin/python run.py \
  --config config/custom/pettingzoo/space_invaders_human_vs_human_record.yaml \
  --output-dir runs \
  --run-id pz_space_invaders_h2h_record
```

启动后打开 viewer：
`http://127.0.0.1:5800/ws_rgb/viewer`

### 6.2 动作编号说明

当前配置使用离散数字动作（`use_action_meanings: false`）：

- `0`: NOOP（不操作）
- `1`: FIRE（开火）
- `2`: RIGHT（右移）
- `3`: LEFT（左移）
- `4`: RIGHTFIRE（右移并开火）
- `5`: LEFTFIRE（左移并开火）

### 6.3 键盘映射（viewer Key Capture）

当前已配置为「一个键盘双人对战」：

- `player_0`（左手区，`Q/W/E + A/S/D`）
- `Q` -> `5`（LEFTFIRE）
- `W` -> `1`（FIRE）
- `E` -> `4`（RIGHTFIRE）
- `A` -> `3`（LEFT）
- `S` -> `0`（NOOP）
- `D` -> `2`（RIGHT）
- `player_1`（右手区，`U/I/O + J/K/L`）
- `U` -> `5`（LEFTFIRE）
- `I` -> `1`（FIRE）
- `O` -> `4`（RIGHTFIRE）
- `J` -> `3`（LEFT）
- `K` -> `0`（NOOP）
- `L` -> `2`（RIGHT）

### 6.4 双人输入方式

在 viewer 页面勾选 `Key Capture` 后，以上两套按键会自动分别路由到：
- 左手按键 -> `player_0`
- 右手按键 -> `player_1`

也就是说，两名玩家可以直接在一个键盘上同时操作，不需要再手工调用接口切换 `human_player_id`。

先查看当前 `display_id`：

```bash
curl -s http://127.0.0.1:5800/ws_rgb/displays
```

示例：向 `player_1` 发送动作 `1`（开火）：

```bash
curl -s -X POST http://127.0.0.1:5800/ws_rgb/input \
  -H 'Content-Type: application/json' \
  -d '{
    "display_id":"task:pz_atari_demo_1:pettingzoo_space_invaders_human_record_arena:pettingzoo_aec_v1",
    "payload":{"type":"action","action":"1"},
    "context":{"human_player_id":"player_1"}
  }'
```

也可走 action server（`8001`）：

```bash
curl -s -X POST http://127.0.0.1:8001/tournament/action \
  -H 'Content-Type: application/json' \
  -d '{"action":"1","player_id":"player_1"}'
```
