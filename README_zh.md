<div align="center">

# 📐 GAGE: General AI evaluation and Gauge Engine

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)](https://www.python.org/) [![Code Style](https://img.shields.io/badge/code%20style-google-blueviolet)](https://google.github.io/styleguide/pyguide.html) [![License](https://img.shields.io/badge/license-TBD-lightgrey)]() [![Status](https://img.shields.io/badge/Status-Alpha-orange)]()

[English](README.md) · **中文**

[框架总览](docs/guide/framework_overview_zh.md) · [Sample 契约](docs/guide/sample_zh.md) · [Game Arena](docs/guide/game_arena_zh.md) · [Support 模块](docs/guide/support_cli_zh.md) · [贡献指南](CONTRIBUTING.md) · [编码规范](AGENTS.md)

</div>

---

**GAGE** 是面向大语言模型、多模态（全模态、机器人）模型、音频模型与扩散模型的统一可扩展评测框架。它是一套高性能评测引擎，强调极致执行效率、可扩展性与灵活性，为 AI 模型评测、Agent 基准与 Game Arena 对战评测提供统一底座。

## ✨ 为什么选择 GAGE？

- 🚀 **极速评测引擎**：以性能为先，充分利用 GPU 与 CPU 资源，从单机测试到百万样本、多集群评测都能平滑扩展。

- 🔗 **一体化评测接口**：以最少粘合代码评测任意数据集 × 任意模型。统一抽象数据集、模型、指标与运行时，快速接入新基准或新后端。

- 🔌 **可扩展沙箱（Game 与 Agent）**：原生支持游戏评测、Agent 环境、GUI 交互沙箱与工具增强任务。全部能力运行在同一评测引擎中，统一评测 LLM、多模态模型与 Agent。

- 🧩 **继承式扩展**：通过继承与覆写扩展已有基准，新增数据集、指标或评测逻辑无需修改核心框架或重写样板代码。

- 📡 **企业级可观测性**：不止日志，提供运行阶段的实时指标与可视化能力，便于监控评测并快速定位性能瓶颈与失败原因。

## 🧭 设计概览

> 核心设计理念：Everything is a Step, Everything is configurable.

### 架构设计

![整体流程](docs/assets/process-arch.png)

### 编排设计

![Step 视角](docs/assets/step-chain.png)

### GameArena 设计

![GameArena 设计](docs/assets/game-arena.png)

## 🚀 快速开始

### 1. 环境准备

```bash
# 推荐使用 Python 3.10+
# 如果你在 mono-repo 根目录，请先执行：cd gage-eval-main
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. 运行第一个 Demo

```bash
# 运行 Echo 演示（无需 GPU，使用 Dummy Backend）
python run.py \
  --config config/run_configs/demo_echo_run_1.yaml \
  --output-dir runs \
  --run-id demo_echo
```

### 3. 查看报告

默认产物结构如下：

```text
runs/<run_id>/
  events.jsonl  # 详细事件日志
  samples.jsonl # 包含输入输出的样本
  summary.json  # 最终评分摘要
```

## 📖 进阶配置

| 场景 | 配置文件示例 | 说明 |
| :--- | :--- | :--- |
| **基础问答** | `config/custom/piqa_qwen3.yaml` | 文本多选任务 (PIQA) |
| **LLM 裁判** | `config/custom/single_task_local_judge_qwen.yaml` | 使用本地 LLM 进行打分 |
| **Game Arena** | `config/custom/gomoku_human_vs_llm.yaml` | 五子棋 Human vs LLM 对战 |
| **代码能力** | `config/custom/swebench_pro_smoke.yaml` | SWE-bench (需 Docker 环境，实验性) |

## 🗺️ 近期计划

- 🤖 **Agent 评测能力**：完善工具调用轨迹、过程对齐与安全检查，形成可复用的 Agent 评测模板。
- 🎮 **GameArena 游戏扩展**：补充更多对战类型、规则配置与可视化能力，形成可扩展的游戏评测矩阵。
- 🛠️ **Gage-Client 工程化**：打造独立的客户端工具，优化配置管理、失败样本定位及 Benchmark 接入脚手架。
- 🌐 **多机分布式推理**：引入 `RoleType Controller` 架构，支持超大规模评测任务的任务分片与负载均衡。
- 🚀 **Benchmark 矩阵扩展**：持续丰富各领域评测集，提供开箱即用的标准配置、指标解释与排障指引。

## ⚠️ 状态

当前处于内部验证期：API、配置与文档可能随实现快速迭代。
