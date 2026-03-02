# ViZDoom 环境安装与启动指南

[English](vizdoom_setup_guide.md) | 中文

本文档说明如何在当前项目中安装 ViZDoom 运行环境，并将游戏启动起来。

## 1. 适用范围

本文档面向当前 `gage-eval` 项目中的 ViZDoom 接入方式。

当前形态是：

- human 模式通过本地 `pygame` 窗口输入
- replay 通过 `ws_rgb` 查看
- human 本地交互需要桌面图形环境

## 2. 前置条件

- Python `3.10+`
- 推荐使用 Conda
- 在仓库根目录执行命令

如果你要运行 human 模式，当前机器必须支持本地图形窗口。

## 3. 安装环境

### 3.1 激活 Conda 环境

如果你已经有项目环境：

```bash
conda activate GAGE
```

如果需要新建：

```bash
conda create -n GAGE python=3.10 -y
conda activate GAGE
```

### 3.2 安装项目依赖

```bash
pip install -r requirements.txt
```

### 3.3 安装 ViZDoom 依赖

主项目的 `requirements.txt` 已包含 `pygame`，但还需要安装 `vizdoom` 本体。

推荐方式：

```bash
pip install -r requirements-vizdoom.txt
```

等价的最小安装：

```bash
pip install vizdoom pygame
```

### 3.4 验证安装

```bash
python - <<'PY'
import vizdoom
import pygame
print("vizdoom ok")
print("pygame ok")
PY
```

## 4. 进入仓库根目录

启动前先进入仓库根目录：

```bash
cd /Users/zzzck/PyProject/GAGE
```

## 5. API Key 设置

以下模式需要 API Key：

- `human_vs_llm`
- `llm_vs_llm`
- `ai_vs_ai`
- `agent_vs_llm`
- `human_vs_llm_record`

设置以下任意一个：

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
```

或者：

```bash
export LITELLM_API_KEY="<YOUR_KEY>"
```

以下模式不需要 Key：

- `human_vs_dummy`
- `human_solo`

## 6. 启动命令

以下命令默认在仓库根目录执行。

### 6.1 Human vs Dummy

推荐先用这个模式验证环境是否正常：

```bash
bash scripts/oneclick/run_vizdoom_human_vs_dummy.sh
```

### 6.2 Human Solo

```bash
bash scripts/oneclick/run_vizdoom_human_solo.sh
```

### 6.3 Human vs LLM

```bash
bash scripts/oneclick/run_vizdoom_human_vs_llm.sh
```

### 6.4 Human vs LLM（Record Scheduler）

```bash
bash scripts/oneclick/run_vizdoom_human_vs_llm_record.sh
```

### 6.5 LLM vs LLM

```bash
bash scripts/oneclick/run_vizdoom_llm_vs_llm.sh
```

### 6.6 AI vs AI

```bash
bash scripts/oneclick/run_vizdoom_ai_vs_ai.sh
```

### 6.7 Agent vs LLM

```bash
bash scripts/oneclick/run_vizdoom_agent_vs_llm.sh
```

## 7. 操作按键

当前 human 输入按键映射为：

- `A` / `Left`：`2`
- `D` / `Right`：`3`
- `Space` / `J`：`1`

## 8. 运行产物位置

每次运行结束后，产物会写到：

```text
runs/<run_id>/
```

常见内容包括：

- `summary.json`
- `samples/`
- `samples.jsonl`
- `replays/`

## 9. 回放命令

按 `run_id` 回放一局已经完成的对局：

```bash
bash scripts/oneclick/run_vizdoom_replay.sh <run_id>
```

示例：

```bash
bash scripts/oneclick/run_vizdoom_replay.sh vizdoom_human_vs_llm_20260228_102306
```

默认查看地址：

```text
http://127.0.0.1:5800/ws_rgb/viewer
```

## 10. 推荐最小验证流程

如果你只想确认环境能不能跑起来，建议按下面顺序：

1. 安装依赖
2. 启动 `human_vs_dummy`
3. 确认游戏窗口正常出现
4. 确认 `runs/<run_id>/` 下生成产物
5. 用 `run_vizdoom_replay.sh` 回放刚才这局

## 11. 常见问题

### 11.1 `ImportError: No module named vizdoom`

安装 ViZDoom：

```bash
pip install vizdoom
```

### 11.2 Human 输入没有响应

- 确保本地输入窗口拿到焦点
- 当前 human 模式依赖本地 `pygame`

### 11.3 回放脚本找不到 `samples`

回放依赖 sample 产物，请确认：

- `runs/<run_id>/samples/` 存在
- 目录下至少有一个 sample JSON 文件
