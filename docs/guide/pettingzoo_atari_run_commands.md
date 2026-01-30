# PettingZoo Atari Run Commands

This document lists the execution commands for all integrated PettingZoo Atari games.
Run these commands from the project root.

## 1. Replay Tool (Visualizing Games)

Use the replay tool to watch recorded games smoothly (without LLM latency).

```bash
# Usage: python scripts/replay_pettingzoo.py <sample_json> <delay_ms>
# delay_ms: 17 = 60fps (realtime), 100 = 10fps (slow)

python scripts/replay_pettingzoo.py \
  runs/pz_boxing_dummy/samples/task_pettingzoo_boxing_dummy/pettingzoo_boxing_dummy_pz_atari_demo_1.json \
  50
```

## 2. AI Agents (Requires OpenAI API Key)
Use these configs to run games with LLM-powered agents.

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

## 3. Dummy Agents (No API Key Required)
Use these configs to test environment installation and rendering without LLM costs.
Agents will move randomly (fixed with `fallback_policy` to ensure legal moves).

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

## 4. Automation: Run & Replay (Recommended)
Use this workflow to run an AI game and automatically start the replay viewer immediately after it finishes.

**Generic Command (Copy & Paste):**
```bash
# 1. Set the game name (e.g., space_invaders, boxing, pong)
export GAME="space_invaders"
export RUN_ID="pz_${GAME}_auto_$(date +%s)"

# 2. Run AI simulation
python run.py \
  --config config/custom/pettingzoo/${GAME}_ai.yaml \
  --output-dir runs \
  --run-id "$RUN_ID"

# 3. Launch Replay
python scripts/replay_pettingzoo.py \
  "$(find runs/$RUN_ID/samples -name "*.json" | head -n 1)" \
  17
```
