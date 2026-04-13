# PettingZoo Atari Game Arena 指南

[English](game_arena_pettingzoo_atari.md) | 中文

PettingZoo Atari 使用 GameKit AEC 环境运行时和 Arena Visual 帧场景。当前 topic 聚焦 Space Invaders，覆盖 dummy、单 LLM、双 LLM、人类和 low-latency 可视化路径。

## 1. 标准文件

| 用途 | 文件 |
| --- | --- |
| Dummy headless 冒烟 | `config/custom/pettingzoo/space_invaders_dummy_gamekit.yaml` |
| Dummy 可视化冒烟 | `config/custom/pettingzoo/space_invaders_dummy_visual_gamekit.yaml` |
| Binary stream 可视化冒烟 | `config/custom/pettingzoo/space_invaders_dummy_visual_binary_stream_gamekit.yaml` |
| Low-latency 可视化冒烟 | `config/custom/pettingzoo/space_invaders_dummy_visual_low_latency_channel_gamekit.yaml` |
| Human 可视化 | `config/custom/pettingzoo/space_invaders_human_visual_gamekit.yaml` |
| 本地测试 LLM headless | `config/custom/pettingzoo/space_invaders_llm_headless_gamekit.yaml` |
| 本地测试 LLM 可视化 | `config/custom/pettingzoo/space_invaders_llm_visual_gamekit.yaml` |
| 本地测试双 LLM 可视化 | `config/custom/pettingzoo/space_invaders_double_llm_visual_gamekit.yaml` |
| 本地测试双 LLM low-latency 可视化 | `config/custom/pettingzoo/space_invaders_double_llm_visual_low_latency_channel_gamekit.yaml` |
| OpenAI LLM headless | `config/custom/pettingzoo/space_invaders_llm_headless_openai_gamekit.yaml` |
| OpenAI LLM 可视化 | `config/custom/pettingzoo/space_invaders_llm_visual_openai_gamekit.yaml` |
| 双 OpenAI LLM 可视化 | `config/custom/pettingzoo/space_invaders_double_llm_visual_openai_gamekit.yaml` |
| 双 OpenAI low-latency 可视化 | `config/custom/pettingzoo/space_invaders_double_llm_visual_low_latency_channel_openai_gamekit.yaml` |

## 2. 前置准备

Atari 路径依赖 `requirements.txt` 中的 `pettingzoo[atari]` 和 `shimmy[atari]`。如果本地 ALE 没有 Space Invaders ROM，运行真实 backend mode 前需要先导入或安装 Atari ROM。Dummy 配置仍可用 `--max-samples 0` 做加载检查。

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
bash scripts/run/arenas/pettingzoo/run.sh --mode dummy_visual --max-samples 1
```

```bash
OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>" bash scripts/run/arenas/pettingzoo/run.sh --mode llm_visual_openai --max-samples 1
```

```bash
OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>" bash scripts/run/arenas/pettingzoo/run.sh --mode double_llm_visual_openai --max-samples 1
```

```bash
bash scripts/run/arenas/pettingzoo/run.sh --mode human_visual --max-samples 1
```

任何 OpenAI 配置都可以先用 `--max-samples 0` 校验配置加载，不执行样本。

## 4. Mode 与 Config 映射

`scripts/run/arenas/pettingzoo/run.sh` 会按 `--python-bin`、`PYTHON_BIN`、当前 virtualenv/conda、`python`/`python3` 的顺序选择 Python。脚本调用 `run.py` 前会打印实际 Python、mode、config、输出目录和 run id。

| 入口 | 配置 | 用途 |
| --- | --- | --- |
| `dummy` | `config/custom/pettingzoo/space_invaders_dummy_gamekit.yaml` | Headless dummy cycle。 |
| `dummy_visual` | `config/custom/pettingzoo/space_invaders_dummy_visual_gamekit.yaml` | `http_pull` Arena Visual 帧冒烟。 |
| `binary_stream` | `config/custom/pettingzoo/space_invaders_dummy_visual_binary_stream_gamekit.yaml` | Binary stream 媒体路径冒烟。 |
| `low_latency` | `config/custom/pettingzoo/space_invaders_dummy_visual_low_latency_channel_gamekit.yaml` | Low-latency channel 媒体路径冒烟。 |
| `human_visual` | `config/custom/pettingzoo/space_invaders_human_visual_gamekit.yaml` | 人类 pilot 对 dummy pilot。 |
| `llm_headless` | `config/custom/pettingzoo/space_invaders_llm_headless_gamekit.yaml` | 本地测试 LLM pilot，无浏览器。 |
| `llm_visual` | `config/custom/pettingzoo/space_invaders_llm_visual_gamekit.yaml` | 本地测试 LLM pilot，带浏览器。 |
| `double_llm_visual` | `config/custom/pettingzoo/space_invaders_double_llm_visual_gamekit.yaml` | 两个本地测试 LLM pilots，带浏览器。 |
| `double_llm_low_latency` | `config/custom/pettingzoo/space_invaders_double_llm_visual_low_latency_channel_gamekit.yaml` | 两个本地测试 LLM pilots 使用 low-latency transport。 |
| `llm_headless_openai` | `config/custom/pettingzoo/space_invaders_llm_headless_openai_gamekit.yaml` | OpenAI pilot 无浏览器。 |
| `llm_visual_openai` | `config/custom/pettingzoo/space_invaders_llm_visual_openai_gamekit.yaml` | OpenAI pilot 带浏览器。 |
| `double_llm_visual_openai` | `config/custom/pettingzoo/space_invaders_double_llm_visual_openai_gamekit.yaml` | 两个 OpenAI pilots 带浏览器。 |
| `double_llm_low_latency_openai` | `config/custom/pettingzoo/space_invaders_double_llm_visual_low_latency_channel_openai_gamekit.yaml` | 两个 OpenAI pilots 使用 low-latency transport。 |

如果脚本可用，`--config <path>` 始终覆盖脚本内置 mode 映射。

## 5. 浏览器控制

可视化配置使用 `visualizer.mode: arena_visual`。浏览器打开的 session URL 形态如下：

```text
http://127.0.0.1:<visual_port>/sessions/<sample_id>?run_id=<run_id>
```

共享的 command deck、播放控制、utility rail、timeline 和 replay 状态见 [Arena Visual 浏览器控制面](game_arena_visual_control_zh.md)。

![PettingZoo Space Invaders Arena Visual 画面](../../assets/arena-visual-space-invaders-stage-20260409.png)

## 6. 人类输入

当前 visual plugin 会在 decision window 打开时提交帧动作控件。mapper 接受 `action`、`move`、`selected_action`、`selected_move`、`value`、`text`、`action_id` 或索引字段。只有自定义 key map 时才接收键盘事件；内置 Space Invaders 可视化路径应优先使用动作控件。

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

- `ArenaObservation.metadata` 携带 active agent、reward、termination、truncation 和 action meaning 信息。
- `view.media` 与 `arena_visual_session/v1/media/` 携带帧引用。
- `legal_actions.items` 是 parser 使用的离散动作 id 或动作含义列表。
- `GameResult` 汇总 move count、illegal count、final board text 和 episode result。

内置 LLM player 会读取 sample messages，并追加一条由当前 `ArenaObservation` 生成的 user turn：active player、view text、legal moves，以及“只返回一个合法动作”的指令。模型返回值会封装成 `ArenaAction`，再交给 GameKit 环境执行。

## 9. 常用参数

| 调整项 | 位置 |
| --- | --- |
| Backend mode | `runtime_overrides.backend_mode` |
| Cycle 预算 | `runtime_overrides.max_cycles` |
| 动作含义 | `runtime_overrides.use_action_meanings` |
| 原始 observation payload | `runtime_overrides.include_raw_obs` |
| 实时画面传输 | `visualizer.live_scene_scheme` |
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
| ROM 或 ALE 错误 | 为本地 ALE/PettingZoo 环境安装或导入 Space Invaders ROM，然后重试同一配置。 |
