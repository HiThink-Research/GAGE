# PettingZoo Atari 使用说明书 (User Guide)

本文档是 **PettingZoo Atari** 游戏在 GAGE 框架下的标准使用指南。
无论您是想运行 AI 对战，还是测试环境配置，或者是回放录像，都可以参考本手册。

## 1. 快速开始 (Quick Start)

我们推荐使用 **"运行即回放" (Run & Replay)** 的自动化流程。
这不仅能让您看到 AI 的决策过程，还能通过 Pygame 窗口直观地欣赏对战画面。

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

# 3. 启动回放
python scripts/replay_pettingzoo.py \
  "$(find runs/$RUN_ID/samples -name "*.json" | head -n 1)" \
  17
```
*(注：`17` 代表 17ms/帧，即 60FPS 实时速度)*

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

## 4. 回放工具 (Replay Tool)

如果您已经跑完了一个任务（比如在后台排队跑的），想事后查看录像，可以使用 `scripts/replay_pettingzoo.py`。

### 基础用法

```bash
python scripts/replay_pettingzoo.py <path_to_json_sample> <delay_ms>
```

*   **`path_to_json_sample`**: `runs/<run_id>/samples/.../*.json` 文件。
*   **`delay_ms`**: 帧间隔。
    *   `17`: 60 FPS (正常速度)
    *   `100`: 10 FPS (慢动作分析)
    *   `0`: 极速 (快进)

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
