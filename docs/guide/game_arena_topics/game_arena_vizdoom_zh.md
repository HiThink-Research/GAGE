# ViZDoom Game Arena 指南

[English](game_arena_vizdoom.md) | 中文

ViZDoom 使用 GameKit 实时 duel 运行时和 Arena Visual 帧场景。它覆盖离散动作 id、POV 帧、telemetry 和需要桌面环境的人类浏览器控制。

## 1. 标准文件

| 用途 | 文件 |
| --- | --- |
| Dummy headless 冒烟 | `config/custom/vizdoom/vizdoom_dummy_gamekit.yaml` |
| Human 可视化 | `config/custom/vizdoom/vizdoom_human_visual_gamekit.yaml` |
| 本地测试 LLM headless | `config/custom/vizdoom/vizdoom_llm_headless_gamekit.yaml` |
| 本地测试 LLM 可视化 | `config/custom/vizdoom/vizdoom_llm_visual_gamekit.yaml` |
| OpenAI LLM headless | `config/custom/vizdoom/vizdoom_llm_headless_openai_gamekit.yaml` |
| OpenAI LLM 可视化 | `config/custom/vizdoom/vizdoom_llm_visual_openai_gamekit.yaml` |

## 2. 前置准备

ViZDoom 依赖 `requirements.txt` 中的 `vizdoom` 和 `pygame`。可视化和人类路径需要桌面或兼容的渲染/输入环境；headless dummy 冒烟是最快的环境检查。

```bash
pip install -r requirements.txt

export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
# 可选：默认模型为 gpt-5.4。
export GAGE_GAME_ARENA_LLM_MODEL="gpt-5.4"
# 可选：覆盖为 OpenAI-compatible 服务。
export OPENAI_API_BASE="https://api.openai.com/v1"
```

使用官方 OpenAI API 时保留默认 endpoint，或不设置 `OPENAI_API_BASE`。如果要跑通过 OpenAI-compatible API 暴露的开源模型，只需要设置 `OPENAI_API_BASE` 和 `GAGE_GAME_ARENA_LLM_MODEL`，不需要改 YAML backend。

普通运行不需要 Node/npm。只有开发或重建 `frontend/arena-visual` 时才需要 Node/npm；见 [Arena Visual 浏览器控制面](game_arena_visual_control_zh.md#2-后端与前端资源)。

## 3. 快速运行

请在仓库根目录、项目 Python 环境已激活后运行。

```bash
bash scripts/run/arenas/vizdoom/run.sh --mode dummy --max-samples 1
```

```bash
OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>" bash scripts/run/arenas/vizdoom/run.sh --mode llm_visual_openai --max-samples 1
```

建议先用这条命令验证 ViZDoom、websocketRGB 和浏览器查看链路，再继续切到模型模式。

这个 websocketRGB helper 的执行顺序：

1. 选择 Python 解释器。
2. 校验配置文件路径。
3. 选择一个可用的 `WS_RGB_PORT`。
4. 在后台启动 `python run.py --config ...`。
5. 等待 `http://127.0.0.1:<port>/ws_rgb/viewer` 可访问。
6. 打印可用的 viewer 地址，并按配置决定是否自动打开浏览器。

这个示例里最常用的变量：

- `PYTHON_BIN`：Python 解释器
- `CFG`：默认 `config/custom/vizdoom/vizdoom_dummy_vs_dummy_ws_rgb.yaml`
- `RUN_ID`：写入 `runs/` 的运行编号
- `OUTPUT_DIR`：默认 `runs`
- `WS_RGB_HOST`：默认 `127.0.0.1`
- `WS_RGB_PORT`：默认 `5800`

如果你想改成本地窗口验证路径，可以执行：

```bash
bash scripts/run/arenas/vizdoom/run.sh --mode human_visual --max-samples 1
```

任何 OpenAI 配置都可以先用 `--max-samples 0` 校验配置加载，不执行样本。

## 4. Mode 与 Config 映射

`scripts/run/arenas/vizdoom/run.sh` 会按 `--python-bin`、`PYTHON_BIN`、当前 virtualenv/conda、`python`/`python3` 的顺序选择 Python。脚本调用 `run.py` 前会打印实际 Python、mode、config、输出目录和 run id。

| 入口 | 配置 | 用途 |
| --- | --- | --- |
| `dummy` | `config/custom/vizdoom/vizdoom_dummy_gamekit.yaml` | Headless dummy 冒烟。 |
| `llm_headless` | `config/custom/vizdoom/vizdoom_llm_headless_gamekit.yaml` | 本地测试 LLM 无浏览器。 |
| `llm_visual` | `config/custom/vizdoom/vizdoom_llm_visual_gamekit.yaml` | 本地测试 LLM 带浏览器。 |
| `llm_headless_openai` | `config/custom/vizdoom/vizdoom_llm_headless_openai_gamekit.yaml` | OpenAI LLM 无浏览器。 |
| `llm_visual_openai` | `config/custom/vizdoom/vizdoom_llm_visual_openai_gamekit.yaml` | OpenAI LLM 带浏览器。 |
| `human_visual` | `config/custom/vizdoom/vizdoom_human_visual_gamekit.yaml` | 人类实时浏览器输入。 |

如果脚本可用，`--config <path>` 始终覆盖脚本内置 mode 映射。

## 5. 浏览器控制

可视化配置使用 `visualizer.mode: arena_visual`。浏览器打开的 session URL 形态如下：

```text
http://127.0.0.1:<visual_port>/sessions/<sample_id>?run_id=<run_id>
```

共享的 command deck、播放控制、utility rail、timeline 和 replay 状态见 [Arena Visual 浏览器控制面](game_arena_visual_control_zh.md)。

![ViZDoom Arena Visual 画面](../../assets/arena-visual-vizdoom-stage-20260409.png)

## 6. 人类输入

帧插件会在存在匹配合法动作时把 W/Up 映射为 forward，把 A/Left 和 D/Right 映射为转向，把 Space/Enter/J 映射为 fire。直接 payload 可通过 `action`、`move`、`selected_action`、`selected_move`、`value`、`text` 提交离散动作 id 或合法动作名；`action_id`、`action_index`、`move_index`、`index` 可从合法动作列表中选择。

## 7. 输出与回放产物

可视化运行会同时写出评测输出和可回放的 Arena Visual 产物：

```text
runs/<run_id>/
  summary.json
  samples.jsonl
  replays/<sample_id>/
    replay.json
    events.jsonl
    arena_visual_session/v1/
      manifest.json
      timeline.jsonl
      scenes/
      media/
```

`sample.predict_result[0].arena_trace` 保存每一步动作、合法性、耗时、重试和 scheduler 元数据。`sample.predict_result[0].game_arena` 保存终局 footer。启用可视化时，`artifacts.visual_session_ref` 指向 `arena_visual_session/v1/manifest.json`。已完成运行的回放也读取同一套 Arena Visual session 产物。

## 8. 运行内部契约

共享运行契约见 [Arena Visual 浏览器控制面](game_arena_visual_control_zh.md#3-运行契约)。本游戏排查时优先看这些字段：

- `ArenaObservation.view` 携带 POV 帧媒体和紧凑文本 telemetry。
- `metadata` 可包含 health、ammo、frag、position、reward 和 step 字段。
- `legal_actions.items` 是 parser 接受的离散动作列表。
- `GameResult` 汇总 duel outcome、move count、illegal count 和 final board text。

内置 LLM player 会读取 sample messages，并追加一条由当前 `ArenaObservation` 生成的 user turn：active player、view text、legal moves，以及“只返回一个合法动作”的指令。模型返回值会封装成 `ArenaAction`，再交给 GameKit 环境执行。

## 9. 常用参数

| 调整项 | 位置 |
| --- | --- |
| Backend mode | `runtime_overrides.backend_mode` |
| 动作重复 | `runtime_overrides.action_repeat` |
| 帧步长 | `runtime_overrides.frame_stride` |
| POV 采集 | `runtime_overrides.capture_pov` / `show_pov` |
| 步数预算 | `runtime_overrides.max_steps` / `stub_max_rounds` |
| 实时画面传输 | `visualizer.live_scene_scheme` |

## 10. 常见排障

| 现象 | 检查项 |
| --- | --- |
| OpenAI 配置在 runtime 启动前失败 | 运行任何 `*_openai_gamekit.yaml` 前先 export `OPENAI_API_KEY`。 |
| 使用了错误模型 | 启动前设置 `GAGE_GAME_ARENA_LLM_MODEL`，或取消该变量以使用文档默认值。 |
| 依赖导入失败 | 在 `run.py` 或 arena 脚本实际使用的同一个 Python 环境里执行 `pip install -r requirements.txt`。 |
| 出现 ROM 或桌面/渲染错误 | 见下方本游戏说明；如果在棋盘或牌类游戏里看到 ROM 错误，通常是跑错了帧游戏配置。 |
| 浏览器一直 loading | 检查控制台打印的 session URL、`visualizer.port`，以及 `runs/<run_id>/replays/<sample_id>/arena_visual_session/v1/manifest.json`。 |
| 端口被占用 | 调整配置里的 `visualizer.port`，或更换 `--run-id`/输出目录，避免读到旧 session。 |
| 人类输入无效 | 确认 `human_input.enabled: true`、浏览器 URL 带当前 `run_id`，且当前 active actor 是 human player。 |
| LLM 返回非法动作 | 以浏览器或 Trace 面板里的 legal move list 为准；配置了 fallback policy 时内置 LLM driver 会按策略兜底。 |
| ViZDoom 或桌面/渲染错误 | 安装 `vizdoom` 和 `pygame`；可视化/human 运行需要桌面会话或兼容显示/渲染配置。 |
