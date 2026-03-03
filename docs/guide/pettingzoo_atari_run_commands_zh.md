# PettingZoo Atari 启动命令

[English](pettingzoo_atari_run_commands.md) | 中文

本文档列出当前已接入的全部 PettingZoo Atari 游戏启动命令。
请在项目根目录执行这些命令。

## 1. 回放工具（查看游戏过程）

使用回放工具可以平滑查看已录制对局，不受 LLM 推理延迟影响。

```bash
# 用法：python scripts/replay_pettingzoo.py <sample_json> <delay_ms>
# delay_ms: 17 = 60fps（接近实时），100 = 10fps（较慢）

python scripts/replay_pettingzoo.py \
  runs/pz_boxing_dummy/samples/task_pettingzoo_boxing_dummy/pettingzoo_boxing_dummy_pz_atari_demo_1.json \
  50
```

## 2. AI Agents（需要 OpenAI API Key）

使用这些配置运行 LLM 驱动的对战。

### Basketball Pong (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/basketball_pong_ai.yaml \
  --output-dir runs \
  --run-id pz_basketball_pong
```

### Boxing (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/boxing_ai.yaml \
  --output-dir runs \
  --run-id pz_boxing
```

### Combat Plane (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/combat_plane_ai.yaml \
  --output-dir runs \
  --run-id pz_combat_plane
```

### Combat Tank (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/combat_tank_ai.yaml \
  --output-dir runs \
  --run-id pz_combat_tank
```

### Double Dunk (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/double_dunk_ai.yaml \
  --output-dir runs \
  --run-id pz_double_dunk
```

### Entombed Competitive (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/entombed_competitive_ai.yaml \
  --output-dir runs \
  --run-id pz_entombed_competitive
```

### Entombed Cooperative (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/entombed_cooperative_ai.yaml \
  --output-dir runs \
  --run-id pz_entombed_cooperative
```

### Flag Capture (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/flag_capture_ai.yaml \
  --output-dir runs \
  --run-id pz_flag_capture
```

### Foozpong (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/foozpong_ai.yaml \
  --output-dir runs \
  --run-id pz_foozpong
```

### Ice Hockey (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/ice_hockey_ai.yaml \
  --output-dir runs \
  --run-id pz_ice_hockey
```

### Joust (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/joust_ai.yaml \
  --output-dir runs \
  --run-id pz_joust
```

### Mario Bros (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/mario_bros_ai.yaml \
  --output-dir runs \
  --run-id pz_mario_bros
```

### Maze Craze (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/maze_craze_ai.yaml \
  --output-dir runs \
  --run-id pz_maze_craze
```

### Othello (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/othello_ai.yaml \
  --output-dir runs \
  --run-id pz_othello
```

### Pong (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/pong_ai.yaml \
  --output-dir runs \
  --run-id pz_pong
```

### Space Invaders (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/space_invaders_ai.yaml \
  --output-dir runs \
  --run-id pz_space_invaders
```

### Space War (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/space_war_ai.yaml \
  --output-dir runs \
  --run-id pz_space_war
```

### Surround (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/surround_ai.yaml \
  --output-dir runs \
  --run-id pz_surround
```

### Tennis (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/tennis_ai.yaml \
  --output-dir runs \
  --run-id pz_tennis
```

### Video Checkers (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/video_checkers_ai.yaml \
  --output-dir runs \
  --run-id pz_video_checkers
```

### Volleyball Pong (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/volleyball_pong_ai.yaml \
  --output-dir runs \
  --run-id pz_volleyball_pong
```

### Wizard of Wor (AI)

```bash
python run.py \
  --config config/custom/pettingzoo/wizard_of_wor_ai.yaml \
  --output-dir runs \
  --run-id pz_wizard_of_wor
```

## 3. Dummy Agents（不需要 API Key）

使用这些配置验证环境安装和渲染，不产生 LLM 调用成本。
Agent 会随机行动，并通过 `fallback_policy` 保证动作合法。

### Basketball Pong (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/basketball_pong_dummy.yaml \
  --output-dir runs \
  --run-id pz_basketball_pong_dummy
```

### Boxing (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/boxing_dummy.yaml \
  --output-dir runs \
  --run-id pz_boxing_dummy
```

### Combat Plane (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/combat_plane_dummy.yaml \
  --output-dir runs \
  --run-id pz_combat_plane_dummy
```

### Combat Tank (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/combat_tank_dummy.yaml \
  --output-dir runs \
  --run-id pz_combat_tank_dummy
```

### Double Dunk (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/double_dunk_dummy.yaml \
  --output-dir runs \
  --run-id pz_double_dunk_dummy
```

### Entombed Competitive (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/entombed_competitive_dummy.yaml \
  --output-dir runs \
  --run-id pz_entombed_competitive_dummy
```

### Entombed Cooperative (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/entombed_cooperative_dummy.yaml \
  --output-dir runs \
  --run-id pz_entombed_cooperative_dummy
```

### Flag Capture (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/flag_capture_dummy.yaml \
  --output-dir runs \
  --run-id pz_flag_capture_dummy
```

### Foozpong (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/foozpong_dummy.yaml \
  --output-dir runs \
  --run-id pz_foozpong_dummy
```

### Ice Hockey (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/ice_hockey_dummy.yaml \
  --output-dir runs \
  --run-id pz_ice_hockey_dummy
```

### Joust (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/joust_dummy.yaml \
  --output-dir runs \
  --run-id pz_joust_dummy
```

### Mario Bros (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/mario_bros_dummy.yaml \
  --output-dir runs \
  --run-id pz_mario_bros_dummy
```

### Maze Craze (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/maze_craze_dummy.yaml \
  --output-dir runs \
  --run-id pz_maze_craze_dummy
```

### Othello (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/othello_dummy.yaml \
  --output-dir runs \
  --run-id pz_othello_dummy
```

### Pong (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/pong_dummy.yaml \
  --output-dir runs \
  --run-id pz_pong_dummy
```

### Space Invaders (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/space_invaders_dummy.yaml \
  --output-dir runs \
  --run-id pz_space_invaders_dummy
```

### Space War (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/space_war_dummy.yaml \
  --output-dir runs \
  --run-id pz_space_war_dummy
```

### Surround (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/surround_dummy.yaml \
  --output-dir runs \
  --run-id pz_surround_dummy
```

### Tennis (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/tennis_dummy.yaml \
  --output-dir runs \
  --run-id pz_tennis_dummy
```

### Video Checkers (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/video_checkers_dummy.yaml \
  --output-dir runs \
  --run-id pz_video_checkers_dummy
```

### Volleyball Pong (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/volleyball_pong_dummy.yaml \
  --output-dir runs \
  --run-id pz_volleyball_pong_dummy
```

### Wizard of Wor (Dummy)

```bash
python run.py \
  --config config/custom/pettingzoo/wizard_of_wor_dummy.yaml \
  --output-dir runs \
  --run-id pz_wizard_of_wor_dummy
```

## 4. 自动化：运行后立即回放（推荐）

这个流程用于先运行一局 AI 对战，然后在结束后立即启动回放查看。

**通用命令（可直接复制）：**

```bash
# 1. 设置游戏名，例如 space_invaders、boxing、pong
export GAME="space_invaders"
export RUN_ID="pz_${GAME}_auto_$(date +%s)"

# 2. 运行 AI 对战
python run.py \
  --config config/custom/pettingzoo/${GAME}_ai.yaml \
  --output-dir runs \
  --run-id "$RUN_ID"

# 3. 启动回放
python scripts/replay_pettingzoo.py \
  "$(find runs/$RUN_ID/samples -name "*.json" | head -n 1)" \
  17
```
