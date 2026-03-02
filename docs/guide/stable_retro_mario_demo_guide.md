# Stable-Retro Mario YAML Configuration Guide (Repository View)

This guide focuses on the **YAML configuration patterns** currently in this repository, not on local runtime artifacts.

## 1. Preparation

1. Download the game ROM from:
   `https://archive.org/download/super-mario-bros-3-nes/Super%20Mario%20Bros.%203.nes` (only for x86)
2. Keep the folder path where the ROM is saved.
3. Install `stable-retro`.
4. Run:

```bash
python -m retro.import "<rom_save_path>"
```

## 2. Mario YAML Files (Current)

- `config/custom/retro_mario_openai_headless_auto_eval.yaml`
- `config/custom/retro_mario_openai_ws_rgb_auto_eval.yaml`
- `config/custom/retro_mario_phase1_dummy_headless_auto_eval.yaml`
- `config/custom/retro_mario_phase1_dummy_ws.yaml`
- `config/custom/retro_mario_phase1_human_ws.yaml`

## 3. Common YAML Structure (Retro Mario)

All Mario configs follow this skeleton:

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
      human_input: ...   # needed only for ws/human scenarios
      players: ...

tasks:
  - task_id: ...
    dataset_id: ...
```

## 4. How to Configure Core Blocks

## 4.1 `environment`

Required Mario fields:

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

Notes:

- `display_mode=headless`: no ws_rgb display registration
- `display_mode=websocket`: ws_rgb hub path is enabled (via `arena.py`)

## 4.2 `scheduler`

Current Mario demos use two scheduler types:

- `tick`: fixed interval think/apply loop
- `record`: more real-time recording cadence (used by human demo)

Examples:

```yaml
scheduler:
  type: tick
  tick_ms: 100
  max_ticks: 3600
```

```yaml
scheduler:
  type: record
  tick_ms: 16
  max_ticks: 3600
```

## 4.3 `parser`

Mario parser config:

```yaml
parser:
  impl: retro_action_v1
  hold_ticks_min: 1
  hold_ticks_max: 12
  hold_ticks_default: 6
```

## 4.4 `human_input` (ws/human)

### ws_rgb display only (AI/Dummy)

```yaml
human_input:
  enabled: true
  ws_host: ${RETRO_WS_RGB_HOST:-0.0.0.0}
  ws_port: ${RETRO_WS_RGB_PORT:-5800}
  ws_allow_origin: "*"
```

### human control (record scheduler)

```yaml
human_input:
  enabled: true
  host: 0.0.0.0
  port: 8001
  fps: 30
  hold_ticks_default: 6
```

Code behavior notes:

- `host/port` is the action queue server (human input endpoint)
- `ws_host/ws_port` is the ws_rgb viewer service endpoint
- if `ws_host/ws_port` is not set, ws hub falls back to `host` and default port `5800`

## 4.5 `players`

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

## 4.6 `environment.replay`

Recommended baseline:

```yaml
environment:
  replay:
    enabled: true
    mode: both         # action | frame | both
    primary_mode: true # optional: treat replay_path as the primary replay path
    frame_capture:
      enabled: true
      frame_stride: 3
      max_frames: 900
      format: jpeg
      quality: 75
```

Key points:

- replay output is written only when `enabled: true`
- `mode: action` records action events only
- `mode: frame` or `mode: both` includes frame events
- for stable `ws_rgb_replay` playback, prefer `mode: both`

## 5. Current Differences Across the 5 Mario Configs

## 5.1 Headless + OpenAI

File: `retro_mario_openai_headless_auto_eval.yaml`

- `display_mode: headless`
- `scheduler: tick (1200/120)`
- no `human_input`
- no `environment.replay`

## 5.2 Headless + Dummy

File: `retro_mario_phase1_dummy_headless_auto_eval.yaml`

- `display_mode: headless`
- `scheduler: tick (100/3600)`
- no `human_input`
- no `environment.replay`

## 5.3 ws_rgb + OpenAI

File: `retro_mario_openai_ws_rgb_auto_eval.yaml`

- `display_mode: websocket`
- `obs_image: true`
- `human_input.ws_host/ws_port` configured
- `replay.enabled: true`
- current `replay.mode: action`

## 5.4 ws_rgb + Dummy

File: `retro_mario_phase1_dummy_ws.yaml`

- `display_mode: websocket`
- `obs_image: true`
- `human_input.ws_host/ws_port` configured
- `replay.enabled: true`
- current `replay.mode: action`

## 5.5 ws_rgb + Human (record)

File: `retro_mario_phase1_human_ws.yaml`

- `display_mode: websocket`
- `scheduler.type: record`
- `human_input.host/port/fps` configured (`8001`)
- `replay.enabled: true`
- current `replay.mode: both`, `primary_mode: true`

## 6. Reusable YAML Templates

## 6.1 Template A: Visual + Replay Friendly (Recommended)

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

## 6.2 Template B: Pure Headless Evaluation

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

## 7. Start Command (Config-driven)

```bash
env PYTHONPATH=src .venv/bin/python run.py \
  --config <one_of_retro_mario_yaml> \
  --output-dir runs \
  --run-id <your_run_id>
```

For OpenAI-based configs, set key first:

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
```

## 8. Live View and Replay Links

## 8.1 Live View (ws_rgb)

Applies to:

- `retro_mario_openai_ws_rgb_auto_eval.yaml`
- `retro_mario_phase1_dummy_ws.yaml`
- `retro_mario_phase1_human_ws.yaml`

Default links (default port `5800`):

- local: `http://127.0.0.1:5800/ws_rgb/viewer`
- host->container: `http://localhost:5800/ws_rgb/viewer`

If you override port via `RETRO_WS_RGB_PORT`, print links with:

```bash
WS_PORT="${RETRO_WS_RGB_PORT:-5800}"
echo "Live view (local): http://127.0.0.1:${WS_PORT}/ws_rgb/viewer"
echo "Live view (host):  http://localhost:${WS_PORT}/ws_rgb/viewer"
```

## 8.2 Replay (`ws_rgb_replay`)

Replay server command (`<SAMPLE_JSON>` from your current run output):

```bash
REPLAY_PORT="${RETRO_REPLAY_PORT:-5800}"

env PYTHONPATH=src .venv/bin/python -m gage_eval.tools.ws_rgb_replay \
  --sample-json <SAMPLE_JSON> \
  --host 0.0.0.0 \
  --port "${REPLAY_PORT}" \
  --fps 12
```

Replay page links:

- local: `http://127.0.0.1:5800/ws_rgb/viewer`
- host->container: `http://localhost:5800/ws_rgb/viewer`

Print replay links with:

```bash
REPLAY_PORT="${RETRO_REPLAY_PORT:-5800}"
echo "Replay (local): http://127.0.0.1:${REPLAY_PORT}/ws_rgb/viewer"
echo "Replay (host):  http://localhost:${REPLAY_PORT}/ws_rgb/viewer"
```

Tip: confirm `replay_path` exists in your sample JSON before launching replay.
