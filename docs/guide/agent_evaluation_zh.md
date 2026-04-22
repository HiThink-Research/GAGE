# Agent 评测指南（AppWorld / SWE-bench Pro / Tau2 / TerminalBench）

中文 | [English](agent_evaluation.md)

gage-eval 当前支持 AppWorld、SWE-bench Pro、Tau2 与 TerminalBench 四类 Agent 评测。每类 benchmark 均支持两种调度方式（scheduler surface）：`framework_loop` 和 `installed_client`。

本指南以"如何运行"为主，介绍必要的准备步骤、核心配置项与产物位置。

> 路径说明：命令默认在 `GAGE/` 仓库根目录执行，文中 `/path/to/...` 请替换为你的本地路径。

## 0. 文档导航

- 项目首页（中文）：[`README_zh.md`](../../README_zh.md)
- Sample 契约：[`sample_zh.md`](sample_zh.md)
- AppWorld 配置：[`config/custom/appworld/appworld_official_jsonl.yaml`](../../config/custom/appworld/appworld_official_jsonl.yaml)
- SWE-bench 配置：[`config/custom/swebench_pro/swebench_pro_smoke_agent.yaml`](../../config/custom/swebench_pro/swebench_pro_smoke_agent.yaml)
- Tau2 配置目录：[`config/custom/tau2/`](../../config/custom/tau2/)
- TerminalBench 配置：[`config/custom/terminal_bench/`](../../config/custom/terminal_bench/)

## 1. 通用评测链路

### 1.1 架构概览

Phase 1 矩阵由 4 个 benchmark kit 和 2 种 scheduler surface 组成：

| Benchmark Kit | framework_loop | installed_client |
| --- | --- | --- |
| `terminal_bench` | ✓ | ✓ |
| `swebench` | ✓ | ✓ |
| `tau2` | ✓ | ✓ |
| `appworld` | ✓ | ✓ |

统一入口脚本：`scripts/run_phase1_8flows.sh`

### 1.2 分层边界

每个 benchmark kit 遵循以下分层设计：

- `runtime.py`：benchmark lifecycle 和 runtime-owned state
- `sub_workflows/framework_loop.py`：framework-loop 的输入投影、结果归一化和 artifact capture
- `sub_workflows/installed_client.py`：installed-client 的 request/environment 投影、结果归一化和 artifact capture
- `units.py`：可复用的 benchmark-specific helper
- `artifacts.py`：benchmark-specific artifact 导出与诊断文件
- `judge_bridge.py`：verifier resource binding

### 1.3 两种 Scheduler Surface

- **framework_loop**：由 GAGE framework 接管 agent loop（tool calling、memory、推理）
- **installed_client**：由外部 client service（如本地 Codex proxy）自主负责 agent loop；GAGE 只负责 request/environment 投影和结果回收

## 2. 运行前准备

### 2.1 基础环境

```bash
cd /path/to/GAGE
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

脚本默认要求：

- `docker` 在 `PATH` 中
- `.venv/bin/python` 存在，或者显式传入 `PYTHON_BIN`

### 2.2 通用环境变量

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
export OLLAMA_MODEL=qwen3-vl:2b-instruct
export OLLAMA_API_KEY=dummy
export SWEBENCH_LOCAL_PATH=/path/to/local-datasets/swebench_pro
export TAU2_DATA_DIR=/path/to/tau2-bench/data
export TAU2_MAX_STEPS=12
export TAU2_MAX_ERRORS=4
export TAU2_MAX_TURNS=12
export TAU2_USER_MODEL=ollama_chat/qwen3-vl:2b-instruct
```

### 2.3 配置本地 Codex Proxy（installed_client 专用）

installed_client workflow 通过外部 HTTP client service 运行。本地场景由 `gage-installed-client-stub` 提供：

**启动 Proxy：**

```bash
cd /path/to/gage-installed-client-stub
./run_local_codex_proxy.sh
```

默认监听地址：`http://127.0.0.1:8787`

**确认 Proxy 正常：**

```bash
curl -fsS http://127.0.0.1:8787/healthz
```

期望返回包含 `status`、`codex_available`、`codex_executable` 字段。

**Runner 如何发现 Proxy（优先级从高到低）：**

1. `GAGE_CODEX_CLIENT_URL`
2. `CODEX_CLIENT_URL`
3. `GAGE_INSTALLED_CLIENT_URL`
4. 自动探测 `http://127.0.0.1:8787/healthz`

显式指定：

```bash
export GAGE_CODEX_CLIENT_URL=http://127.0.0.1:8787
```

## 3. 运行 8 条 Workflow 矩阵

### 3.1 推荐命令

```bash
cd /path/to/GAGE
RUN_PREFIX=phase1_$(date +%Y%m%d) ./scripts/run_phase1_8flows.sh
```

默认输出目录格式：

```text
runs/<RUN_PREFIX>_8flows_<HHMMSS>/
  <RUN_PREFIX>_<kit>_<scheduler>_<HHMMSS>/
```

### 3.2 常用覆盖参数

切换模型：

```bash
OLLAMA_MODEL=qwen3-vl:2b-instruct \
RUN_PREFIX=phase1_$(date +%Y%m%d) \
./scripts/run_phase1_8flows.sh
```

指定 Python：

```bash
PYTHON_BIN=/path/to/python \
RUN_PREFIX=phase1_$(date +%Y%m%d) \
./scripts/run_phase1_8flows.sh
```

### 3.3 单独运行一条 Workflow

```bash
cd /path/to/GAGE
.venv/bin/python run.py \
  --config config/custom/appworld/appworld_agent_demo_runtime_ollama.yaml \
  --run-id appworld_framework_smoke \
  --output-dir runs \
  --max-samples 1
```

## 4. 检查 Run 结果

### 4.1 Run 目录结构

```text
runs/<batch_dir>/
  <run_id>/
    events.jsonl
    samples.jsonl
    summary.json
    samples/
      runtime/
        <task_id>/
          <sample_id>/
            runtime_metadata.json
            verifier/
              result.json
            artifacts/
            logs/
              raw_error.json
```

### 4.2 第一层：看 `summary.json`

快速判断：

- 这条 run 是否完整结束
- 最终产出了哪些 metrics
- benchmark 层面的整体结果

重点字段：`metrics`、`sample_count`、`runtime_failure_code`、benchmark-specific 聚合字段

最低检查标准：

- `metrics` 存在且非空
- metric 名字符合当前 benchmark config 预期
- `count` 与实际 sample 数量一致

### 4.3 第二层：看 `samples.jsonl`

每个 sample 的行为与失败原因。重点字段：

- `predict_result[0].answer`
- `predict_result[0].agent_trace`
- `eval_result.failure_reason`
- `eval_result.diagnostic_reason`
- `eval_result.diagnostic_details`
- `eval_result.score`

### 4.4 第三层：看 per-sample runtime 目录

路径：`samples/runtime/<task_id>/<sample_id>/`

推荐排查顺序：

1. `samples.jsonl`：标准化后的最终结果
2. `verifier/result.json`：benchmark-native 证据
3. `runtime_metadata.json`：执行上下文
4. `artifacts/`：benchmark-specific 产物
5. `logs/raw_error.json`：原始失败信封

### 4.5 各 Kit 的 Artifact 形态

`terminal_bench` 常见：

- `tool_trace.json`
- `workspace_diff.json`
- `stdout.log`、`stderr.log`

`swebench` 常见：

- `submission.patch`
- `agent_trace.json`
- `final_response.txt`
- `swebench_diagnostics.json`

`tau2` 常见：

- `tau2_state.json`
- `tau2_trajectory.json`
- `tau2_cost.json`

`appworld` 常见：

- `appworld_save.json`
- `appworld_outputs.json`
- `appworld_tool_trace.json`
- `appworld_logs.json`

> **注意**：`artifacts/` 目录存在不代表 artifact 已成功落盘。runtime 会在 sample 开始时就先创建目录结构。若目录为空且 `logs/raw_error.json` 存在，通常说明失败发生在 benchmark-specific artifact capture 之前（如 `acquire_lease`、`bootstrap_runtime`、`run_scheduler` 阶段）。

### 4.6 验收清单

每条 run 至少检查：

- `summary.json` 里的 `metrics` 是否完整
- `samples.jsonl` 是否有 sample 记录
- `predict_result[0].agent_trace` 是否存在且类型正确
- per-sample 的 `artifacts/` 目录是否存在
- benchmark-specific artifact 是否已经落盘
- 失败是否能通过 `failure_reason`、`diagnostic_reason`、`diagnostic_details` 解释清楚

## 5. AppWorld 评测

### 5.1 前置条件

- 需要 Docker（verifier 在容器内运行）
- 本地已构建 AppWorld 镜像

### 5.2 构建镜像并导出数据集

```bash
cd /path/to/GAGE

# 构建 AppWorld 镜像
docker build -t appworld-mcp:latest -f docker/appworld/Dockerfile docker/appworld

# 导出 JSONL 数据集到本地
bash docker/appworld/export_datasets.sh \
  --image appworld-mcp:latest \
  --output ../local-datasets/appworld
```

生成文件：`train.jsonl`、`dev.jsonl`、`test_normal.jsonl`、`test_challenge.jsonl`、`manifest.json`

### 5.3 运行评测（framework_loop）

```bash
cd gage-eval-main
export OPENAI_API_KEY=your_key
python run.py \
  --config config/custom/appworld/appworld_official_jsonl.yaml \
  --run-id appworld_official_jsonl_run_$(date +%H%M%S) \
  --output-dir runs/appworld_official_jsonl
```

### 5.4 运行评测（installed_client）

确认本地 Codex proxy 已启动后：

```bash
cd gage-eval-main
export OPENAI_API_KEY=your_key
python run.py \
  --config config/custom/swebench_pro/swebench_pro_smoke_agent.yaml \
  --run-id swebench_pro_smoke_run_$(date +%H%M%S) \
  --output-dir runs/swebench_pro_smoke
```

### 5.5 指标与产物

- 指标：`tgc`、`sgc`、`pass`、`fail`、`difficulty`
- 产物目录：`samples/runtime/<task_id>/<sample_id>/artifacts/`
- 汇总：`summary.json`

## 6. SWE-bench Pro 评测

### 6.1 前置条件

- 需要 Docker（verifier 会在容器内执行官方测试流程）
- 评测脚本与 Dockerfiles 已包含在 `third_party/swebench_pro/`
- 本地模型示例默认使用 Ollama 兼容接口

### 6.2 准备数据集

```bash
export SWEBENCH_LOCAL_PATH=/path/to/local-datasets/swebench_pro
```

数据集通过本地路径加载，请确保路径已准备好。

### 6.3 运行评测（framework_loop）

```bash
cd /path/to/GAGE
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
export OLLAMA_MODEL=qwen3-vl:2b-instruct
export OLLAMA_API_KEY=dummy
.venv/bin/python run.py \
  --config config/custom/swebench_pro/swebench_pro_smoke_runtime_ollama_local.yaml \
  --run-id swebench_framework_$(date +%H%M%S) \
  --output-dir runs \
  --max-samples 1
```

说明：

- 推荐使用 `swebench_pro_smoke_runtime_ollama_local.yaml` 跑本地 smoke demo；它会从本地数据集读取样本，并通过 `framework_loop` 驱动 agent。
- verifier 会先关闭 agent sandbox，再用同一套 sandbox profile 重新启动干净的 sandbox 执行官方 SWE-bench Pro judge 脚本。
- 重启后的 sandbox 使用 preprocessor 按 instance 注入的同一个镜像，例如 `jefzda/swe-bench-pro-{instance_id}:...`，因此 agent 阶段和 verifier 阶段面对的是同一个实例镜像。
- 现在可以安全地追加 `--max-samples 1` 作为单条 smoke demo：本地 loader 会在应用 dataset `limit` 前，先把 smoke allowlist 中的样本排到前面，避免再出现 `sample_count=0` 的 0 sample run。
- 若不传 `--max-samples`，则按配置内置的 smoke 子集与 `tasks[].max_samples` 执行，适合跑完整 smoke 集。
- `concurrency: 1` 只表示同一时刻只处理 1 个 sample，并不表示内存一定恒定。当前 SampleLoop 默认仍会保留一个很小的预取缓冲区；同时 SWE-bench 每个 sample 都可能切换到不同的 Docker 镜像。若第 2 个 sample 触发了新的镜像拉取、解压或容器启动，宿主机内存占用可能明显高于第 1 个 sample，这不一定代表出现了并行执行。
- 运行过程如果看起来“卡住”，常见原因是 Docker 首次拉取/启动某个 sample 对应的实例镜像，或 verifier 正在执行该 sample 的官方测试脚本。

### 6.4 运行评测（installed_client）

```bash
cd /path/to/GAGE
.venv/bin/python run.py \
  --config config/custom/swebench/swebench_installed_client.yaml \
  --run-id swebench_installed_$(date +%H%M%S) \
  --output-dir runs \
  --max-samples 1
```

### 6.5 指标与产物

- 指标：`swebench_resolve_rate`、`swebench_failure_reason`
- 产物：`submission.patch`、`agent_trace.json`、`swebench_diagnostics.json`
- 运行目录：`runs/<run_id>/events.jsonl`、`runs/<run_id>/samples.jsonl`、`runs/<run_id>/logs/<instance_id>/`

## 7. Tau2 评测

### 7.1 前置条件（安装官方 Tau2 代码）

```bash
git clone https://github.com/sierra-research/tau2-bench
cd tau2-bench
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

运行 gage-eval 时需使用同一个 Python 环境，确保 `tau2` 能被导入。

可选检查：

```bash
tau2 check-data
```

### 7.2 数据集（HF 拉取）

Tau2 loader 从 HuggingFace 下载数据集（`HuggingFaceH4/tau2-bench-data`），落到 `TAU2_DATA_DIR`：

```bash
export TAU2_DATA_DIR=/path/to/tau2-bench/data
```

如 HF 需要鉴权，请设置 `HUGGINGFACEHUB_API_TOKEN`。

### 7.3 运行评测（framework_loop）

以 airline domain 为例：

```bash
cd /path/to/GAGE
export TAU2_DATA_DIR=/path/to/tau2-bench/data
.venv/bin/python run.py \
  --config config/custom/tau2/tau2_airline_framework_loop.yaml \
  --run-id tau2_framework_$(date +%H%M%S) \
  --output-dir runs \
  --max-samples 1
```

### 7.4 运行评测（installed_client）

```bash
cd /path/to/GAGE
.venv/bin/python run.py \
  --config config/custom/tau2/tau2_airline_installed_client.yaml \
  --run-id tau2_installed_$(date +%H%M%S) \
  --output-dir runs \
  --max-samples 1
```

### 7.5 指标与产物

- 指标：`tau2_reward`、`tau2_pass`、`tau2_pass_hat_k`、`tau2_agent_cost`、`tau2_user_cost`
- 汇总：`tau2_summary` 在 `summary.json` 中输出 `pass_hat_k` 与分域统计
- User Simulator 通过 LiteLLM 调用模型，请确保 `TAU2_USER_MODEL` 所需 API Key 已配置。推荐通过 `benchmark_configs.tau2.user_simulator.model` 和 `model_args` 配置；默认模型为 `gpt-4.1`。如果要和 DUT 共用后端，显式填写相同的 LiteLLM model string 与 args。

Tau2 多试次 / seed 最佳实践：

- 固定 `seed` 保证可复现
- 所有 task 使用相同 `num_trials`
- 若需要 `pass@k`，确保 `num_trials >= k`

## 8. TerminalBench 评测

### 8.1 前置条件

- 需要 Docker（sandbox 在容器内运行）

### 8.2 运行评测（framework_loop）

```bash
cd /path/to/GAGE
.venv/bin/python run.py \
  --config config/custom/terminal_bench/terminal_bench_framework_loop.yaml \
  --run-id terminal_bench_framework_$(date +%H%M%S) \
  --output-dir runs \
  --max-samples 1
```

### 8.3 运行评测（installed_client）

```bash
cd /path/to/GAGE
.venv/bin/python run.py \
  --config config/custom/terminal_bench/terminal_bench_installed_client.yaml \
  --run-id terminal_bench_installed_$(date +%H%M%S) \
  --output-dir runs \
  --max-samples 1
```

### 8.4 指标与产物

- 产物：`tool_trace.json`、`workspace_diff.json`、`stdout.log`、`stderr.log`

## 9. 如何新增一个 Benchmark Kit

下面是符合 phase 1 分层设计的接入方式。

### 9.1 新建 Kit 目录

创建：

```text
src/gage_eval/agent_eval_kits/<new_kit>/
```

建议包含：

- `kit.py`
- `runtime.py`
- `resources.py`
- `units.py`
- `artifacts.py`
- `judge_bridge.py`
- `trace_mapping.py`
- `legacy_support_migration.py`
- `sub_workflows/framework_loop.py`
- `sub_workflows/installed_client.py`

不是每个 kit 都必须一样复杂，但这是 phase 1 推荐的标准形态。

### 9.2 在 `runtime.py` 里定义生命周期

`runtime.py` 应负责：

- benchmark-owned state 初始化
- runtime-owned context bootstrap
- 暴露 sub-workflow 需要的稳定状态
- benchmark-specific 的 save/finalize lifecycle

不要把 scheduler-specific 的 payload shaping 放在这里。

### 9.3 在 `sub_workflows/*` 里定义 scheduler-specific 投影

`sub_workflows/framework_loop.py` 负责：

- framework-loop 的输入投影
- framework-loop 的 artifact capture
- framework-loop 的结果归一化

`sub_workflows/installed_client.py` 负责：

- installed-client 的 request payload 构造
- installed-client 的 environment 投影
- installed-client 的 artifact capture
- installed-client 的结果归一化

不要让 sub-workflow 接管整个 benchmark lifecycle。

### 9.4 把共享逻辑放到 `units.py` 和 `artifacts.py`

`units.py` 适合放：

- prompt builder
- 输入投影 helper
- 小型 benchmark-specific normalization helper

`artifacts.py` 适合放：

- artifact 导出
- artifact 命名
- fallback capture
- 诊断文件生成

### 9.5 在 `kit.py` 里注册入口

你的 `load_kit()` 应该返回一个 `BenchmarkKitEntry`，把这些部分接起来：

- runtime entry
- supported schedulers
- workflow resolver
- verifier resource resolver
- trace mapper
- compat shim resolver

workflow resolver 需要明确把不同 scheduler 映射到对应 workflow bundle。

### 9.6 在 `config/custom/<new_kit>/` 下添加配置

至少为两条 surface 各加一份：

- `<new_kit>_framework_loop_*.yaml`
- `<new_kit>_installed_client_*.yaml`

installed-client config 应表达 installed-client runtime path，不应再依赖 framework-loop backend 来充当主执行路径。

### 9.7 先补测试，再加进共享脚本

建议最低测试覆盖：

- `runtime.py` 的 unit tests
- `sub_workflows/framework_loop.py` 的 unit tests
- `sub_workflows/installed_client.py` 的 unit tests
- `artifacts.py` 的 unit tests
- 至少一条单 sample 的 integration test

### 9.8 最后再加到共享 runner

当 kit 自己已经单独能跑通后，再：

1. 把新的 config 加到 `scripts/run_phase1_8flows.sh` 或平行矩阵脚本里
2. 保持统一的 `run_id` 命名模式
3. benchmark-specific 逻辑继续留在 kit 里，不要塞进脚本

### 9.9 Installed-Client Contract 对新 Kit 的要求

installed-client 模式下，kit 需要向 external client handoff 两类内容：

- `request`：benchmark-owned instruction 和稳定 client metadata
- `environment`：runtime-owned context，供 installed-client service 访问 benchmark 环境

更详细的 HTTP schema 在 [`docs/installed_client_service_contract.md`](../installed_client_service_contract.md)。

实践上建议这样理解：

- installed-client service 自己负责 tool calling、memory 和完整 agent loop
- 如果新 kit 需要更多 installed-client 字段，优先在 kit 自己的 request projection 中扩展；只有当字段具备通用价值时，再扩展 shared service contract

## 10. 附录：通用概念说明

| 术语 | 解释 |
| --- | --- |
| framework_loop | 由 GAGE framework 接管 agent loop 的调度方式 |
| installed_client | 由外部 client service 自主负责 agent loop 的调度方式 |
| Benchmark Kit | 封装了特定 benchmark 生命周期的模块，包含 runtime、sub_workflows、artifacts 等 |
| runtime.py | 负责 benchmark lifecycle 和 runtime-owned state 的核心模块 |
| sub_workflows/ | 针对不同 scheduler surface 的投影与归一化逻辑 |
| artifacts.py | benchmark-specific artifact 导出与诊断文件 |
| judge_bridge.py | verifier resource binding |
| samples.jsonl | 标准化后的所有 sample 结果，最适合看 sample 级行为与失败原因 |
| summary.json | 整体 run 的汇总指标与状态 |
| verifier/result.json | benchmark-native 的 verifier 输出，作为 judge 原始证据 |
| pass@k / pass_hat@k | k 次试次中至少一次成功的估计指标（Tau2 使用） |

## 11. 常用命令速查

启动本地 Codex proxy：

```bash
cd /path/to/gage-installed-client-stub
./run_local_codex_proxy.sh
```

检查 installed-client proxy：

```bash
curl -fsS http://127.0.0.1:8787/healthz
```

运行 8-workflow 矩阵：

```bash
cd /path/to/GAGE
RUN_PREFIX=phase1_$(date +%Y%m%d) ./scripts/run_phase1_8flows.sh
```

单独运行一条 workflow：

```bash
cd /path/to/GAGE
.venv/bin/python run.py \
  --config config/custom/appworld/appworld_agent_demo_runtime_ollama.yaml \
  --run-id appworld_framework_smoke \
  --output-dir runs \
  --max-samples 1
```

查某条 run 的关键验收文件：

```bash
find runs/<batch_dir>/<run_id> -maxdepth 3 \( -name summary.json -o -name samples.jsonl -o -name result.json -o -path '*/artifacts/*' \)
```
