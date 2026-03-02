# ws_rgb Replay 使用指南

[English](ws_rgb_replay_usage_guide.md) | 中文

本文档说明如何使用统一 `ws_rgb` 回放链路进行对局回放。
命令内容与 `docs/local/指令精简.md` 保持一致。

如果需要查看 `ws_rgb` 的在线渲染、输入路由与接入契约（不止回放），请参考：
`docs/guide/ws_rgb_runtime_dev_guide_zh.md`

## 1. 范围

- 回放查看页面：`/ws_rgb/viewer`
- 回放服务入口：`python -m gage_eval.tools.ws_rgb_replay`
- 一键脚本：`scripts/oneclick/run_game_replay_oneclick.sh`
- 支持游戏：`gomoku`、`tictactoe`、`doudizhu`、`mahjong`、`pettingzoo`

## 2. 前置条件

在仓库根目录执行：

```bash
cd /path/to/GAGE
```

PettingZoo Atari 首次运行需安装 ROM：

```bash
AutoROM --accept-license
```

AI 模式需设置 Key：

```bash
export OPENAI_API_KEY="<YOUR_OPENAI_API_KEY>"
```

## 3. 回放方式

### 3.1 一键回放（推荐）

该方式是 `run -> replay` 一条命令完成。

通用形式：

```bash
bash scripts/oneclick/run_game_replay_oneclick.sh --game <game> --mode <dummy|ai>
```

Dummy：

```bash
bash scripts/oneclick/run_game_replay_oneclick.sh --game gomoku --mode dummy
bash scripts/oneclick/run_game_replay_oneclick.sh --game tictactoe --mode dummy
bash scripts/oneclick/run_game_replay_oneclick.sh --game doudizhu --mode dummy
bash scripts/oneclick/run_game_replay_oneclick.sh --game mahjong --mode dummy
bash scripts/oneclick/run_game_replay_oneclick.sh --game pettingzoo --mode dummy
```

AI：

```bash
bash scripts/oneclick/run_game_replay_oneclick.sh --game gomoku --mode ai
bash scripts/oneclick/run_game_replay_oneclick.sh --game tictactoe --mode ai
bash scripts/oneclick/run_game_replay_oneclick.sh --game doudizhu --mode ai
bash scripts/oneclick/run_game_replay_oneclick.sh --game mahjong --mode ai
bash scripts/oneclick/run_game_replay_oneclick.sh --game pettingzoo --mode ai
```

常用参数：

```bash
bash scripts/oneclick/run_game_replay_oneclick.sh \
  --game gomoku \
  --mode dummy \
  --port 5860 \
  --auto-open 0
```

```bash
bash scripts/oneclick/run_game_replay_oneclick.sh \
  --game mahjong \
  --mode ai \
  --python-bin "$(command -v python)" \
  --run-id mahjong_ai_replay_demo
```

### 3.2 跑后手动回放（PettingZoo 示例）

如果对局已跑完，后续再看回放，使用如下流程。

Dummy 运行：

```bash
python run.py --config config/custom/pettingzoo/pong_dummy.yaml --output-dir runs --run-id pettingzoo_dummy_run
```

从产物启动回放：

```bash
RUN_ID=pettingzoo_dummy_run
SAMPLE_JSON=$(find "runs/${RUN_ID}/samples" -name '*.json' | head -n 1)

PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo \
  --auto-open 1
```

AI 运行：

```bash
python run.py --config config/custom/pettingzoo/pong_ai.yaml --output-dir runs --run-id pettingzoo_ai_run
```

从产物启动回放：

```bash
RUN_ID=pettingzoo_ai_run
SAMPLE_JSON=$(find "runs/${RUN_ID}/samples" -name '*.json' | head -n 1)

PYTHONPATH=src python -m gage_eval.tools.ws_rgb_replay \
  --sample-json "$SAMPLE_JSON" \
  --host 127.0.0.1 \
  --port 5800 \
  --fps 12 \
  --game pettingzoo \
  --auto-open 1
```

## 4. 说明与排查

- `ws_rgb_replay` 启动后会打印 viewer 地址。
- 可通过 `--auto-open 1` 自动打开浏览器（无图形环境时请手动访问）。
- `/ws_rgb/frame_image` 过程中出现 `BrokenPipeError`，通常是浏览器取消了旧请求，服务端已做容错处理。
- 回放失败时请检查：
  - `runs/<run_id>/samples` 下是否存在 sample JSON
  - sample 的 `predict_result[*].replay_path/replay_v1_path` 是否存在回放产物
  - 端口是否冲突（`--port`）
