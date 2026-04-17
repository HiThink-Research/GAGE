# 五子棋 Game Arena 指南

[English](game_arena_gomoku.md) | 中文

五子棋使用当前 GameKit 棋盘运行时和统一 Arena Visual 浏览器页面。它适合作为第一个 topic，用来确认回合制棋盘流程、合法坐标动作和 visual session 产物。

## 1. 标准文件

| 用途 | 文件 |
| --- | --- |
| Dummy headless 冒烟 | `config/custom/gomoku/gomoku_dummy_gamekit.yaml` |
| Dummy 可视化冒烟 | `config/custom/gomoku/gomoku_dummy_visual_gamekit.yaml` |
| 本地测试 LLM headless | `config/custom/gomoku/gomoku_llm_headless_gamekit.yaml` |
| 本地测试 LLM 可视化 | `config/custom/gomoku/gomoku_llm_visual_gamekit.yaml` |
| OpenAI LLM headless | `config/custom/gomoku/gomoku_llm_headless_openai_gamekit.yaml` |
| OpenAI LLM 可视化 | `config/custom/gomoku/gomoku_llm_visual_openai_gamekit.yaml` |
| Human vs 本地测试 LLM 可视化 | `config/custom/gomoku/gomoku_human_visual_gamekit.yaml` |
| Human vs OpenAI 可视化 | `config/custom/gomoku/gomoku_human_visual_openai_gamekit.yaml` |
| 15x15 human vs 本地测试 LLM 可视化 | `config/custom/gomoku/gomoku_human_visual_15x15_gamekit.yaml` |
| 15x15 human vs OpenAI 可视化 | `config/custom/gomoku/gomoku_human_visual_15x15_openai_gamekit.yaml` |
| Fixture 数据 | `tests/data/Test_Gomoku.jsonl` |

## 2. 前置准备

五子棋本身使用仓库内 GameKit 代码和 fixture JSONL 数据，不需要额外游戏资源。

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
python run.py --config config/custom/gomoku/gomoku_dummy_gamekit.yaml --max-samples 1
```

```bash
python run.py --config config/custom/gomoku/gomoku_dummy_visual_gamekit.yaml --max-samples 1
```

```bash
python run.py --config config/custom/gomoku/gomoku_llm_visual_openai_gamekit.yaml --max-samples 1
```

```bash
python run.py --config config/custom/gomoku/gomoku_human_visual_openai_gamekit.yaml --max-samples 1
```

任何 OpenAI 配置都可以先用 `--max-samples 0` 校验配置加载，不执行样本。

## 4. Mode 与 Config 映射

五子棋目前没有专用的 arena `run.sh`；直接使用 `python run.py --config` 运行这些配置。

| 入口 | 配置 | 用途 |
| --- | --- | --- |
| direct dummy | `config/custom/gomoku/gomoku_dummy_gamekit.yaml` | Headless 确定性冒烟。 |
| direct dummy visual | `config/custom/gomoku/gomoku_dummy_visual_gamekit.yaml` | Arena Visual 棋盘冒烟。 |
| direct llm headless openai | `config/custom/gomoku/gomoku_llm_headless_openai_gamekit.yaml` | OpenAI LLM 黑棋对 dummy 白棋，无浏览器。 |
| direct llm visual openai | `config/custom/gomoku/gomoku_llm_visual_openai_gamekit.yaml` | OpenAI LLM 黑棋对 dummy 白棋，带浏览器。 |
| direct human visual openai | `config/custom/gomoku/gomoku_human_visual_openai_gamekit.yaml` | 人类黑棋对 OpenAI 白棋。 |
| direct human visual 15x15 openai | `config/custom/gomoku/gomoku_human_visual_15x15_openai_gamekit.yaml` | 15x15 人类浏览器变体。 |

如果脚本可用，`--config <path>` 始终覆盖脚本内置 mode 映射。

## 5. 浏览器控制

可视化配置使用 `visualizer.mode: arena_visual`。浏览器打开的 session URL 形态如下：

```text
http://127.0.0.1:<visual_port>/sessions/<sample_id>?run_id=<run_id>
```

共享的 command deck、播放控制、utility rail、timeline 和 replay 状态见 [Arena Visual 浏览器控制面](game_arena_visual_control_zh.md)。

![五子棋 Arena Visual 画面](../../assets/arena-visual-gomoku-stage-20260409.png)

## 6. 人类输入

在棋盘 stage 点击高亮交叉点，或提交 legal move list 里的坐标，例如 `A1`。input mapper 接受 `action`、`move`、`coord`、`selected_action`、`selected_move`、`selected_coord`、`value`、`text`；也可以用 `action_index`、`move_index`、`index` 从当前合法动作列表里选择。非法坐标会在进入环境前被拒绝。

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

- `ArenaObservation.view.text` 是 LLM prompt 和可视化面板使用的棋盘视图。
- `legal_actions.items` 是权威坐标列表。
- `GameResult.move_log` 记录每步坐标、行列和 actor。

内置 LLM player 会读取 sample messages，并追加一条由当前 `ArenaObservation` 生成的 user turn：active player、view text、legal moves，以及“只返回一个合法动作”的指令。模型返回值会封装成 `ArenaAction`，再交给 GameKit 环境执行。

## 9. 常用参数

| 调整项 | 位置 |
| --- | --- |
| 棋盘大小 | `runtime_overrides.board_size` |
| 连珠长度 | `runtime_overrides.win_len` |
| 坐标体系 | `runtime_overrides.coord_scheme` |
| 观测图导出 | `runtime_overrides.obs_image` |
| 人类输入路由 | `human_input` |
| 浏览器端口 | `visualizer.port` |

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
| 本 topic 出现 ROM 或桌面/渲染错误 | 五子棋不需要 ROM 或渲染 backend；确认配置路径位于 `config/custom/gomoku/`。 |
