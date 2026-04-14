# Phase 1 八条 Workflow 运行与扩展指南

本文说明：

- 如何运行 phase 1 的 8 条 workflow 矩阵
- 如何配置本地 Codex proxy 以支持 installed-client workflow
- 如何检查 run 结果是否正确
- 如何按 phase 1 的分层约束新增一个 benchmark kit 和对应 workflow

本文对应的设计文档：

- `site/documents/gage-agent-phase1-development-plan-0408.md`
- `docs/installed_client_service_contract.md`

## 1. 这 8 条 Workflow 是什么

phase 1 的矩阵由 4 个 benchmark kit 和 2 种 scheduler surface 组成：

- `terminal_bench` + `framework_loop`
- `terminal_bench` + `installed_client`
- `swebench` + `framework_loop`
- `swebench` + `installed_client`
- `tau2` + `framework_loop`
- `tau2` + `installed_client`
- `appworld` + `framework_loop`
- `appworld` + `installed_client`

统一脚本入口是：

- `scripts/run_phase1_8flows.sh`

## 2. 分层边界

运行或扩展这套矩阵时，建议始终遵守这些边界：

- `runtime.py`：负责 benchmark lifecycle 和 runtime-owned state
- `sub_workflows/framework_loop.py`：负责 framework-loop 的输入投影、结果归一化和 artifact capture
- `sub_workflows/installed_client.py`：负责 installed-client 的 request/environment 投影、结果归一化和 artifact capture
- `units.py`：负责可复用的 benchmark-specific helper
- `artifacts.py`：负责 benchmark-specific artifact 导出与诊断文件
- `judge_bridge.py`：负责 verifier resource binding
- installed-client 下的 tool calling 和 memory 由 external client service 自己负责，不由 GAGE framework-loop 接管

如果某个逻辑是 benchmark-specific，优先放回对应 kit，而不是塞进通用 runtime core。

## 3. 运行前准备

### 3.1 基础环境

在仓库根目录执行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

脚本默认要求：

- `docker` 在 `PATH` 中
- `.venv/bin/python` 存在，或者你显式传入 `PYTHON_BIN`
- 对应 benchmark 的本地依赖已准备好

### 3.2 Benchmark 输入依赖

脚本默认使用这些环境变量：

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
export OLLAMA_MODEL=qwen3-vl:2b-instruct
export OLLAMA_API_KEY=dummy
export SWEBENCH_LOCAL_PATH=/home/amyjx/work/GAGE/local-datasets/swebench_pro
export TAU2_DATA_DIR=/home/amyjx/work/tau2-bench/data
export TAU2_MAX_STEPS=12
export TAU2_MAX_ERRORS=4
export TAU2_MAX_TURNS=12
export TAU2_USER_MODEL=ollama_chat/qwen3-vl:2b-instruct
```

如果你的本地路径或模型不同，请在运行前覆盖这些变量。

## 4. 配置本地 Codex Proxy

installed-client workflow 通过一个 HTTP installed-client service 运行。在本地 phase 1 场景里，这个 service 由 GAGE 外部的本地 Codex proxy 提供：

- `/home/amyjx/work/gage-installed-client-stub`

### 4.1 启动 Proxy

```bash
cd /home/amyjx/work/gage-installed-client-stub
./run_local_codex_proxy.sh
```

如果 `codex` 不在默认 `PATH` 中，可以这样启动：

```bash
cd /home/amyjx/work/gage-installed-client-stub
CODEX_EXECUTABLE=/path/to/codex ./run_local_codex_proxy.sh
```

默认监听地址：

```text
http://127.0.0.1:8787
```

### 4.2 检查 Proxy 是否正常

```bash
curl -fsS http://127.0.0.1:8787/healthz
```

期望至少看到这些字段：

- `status`
- `codex_available`
- `codex_executable`

### 4.3 Runner 如何发现 Proxy

脚本会按以下优先级读取 installed-client service URL：

- `GAGE_CODEX_CLIENT_URL`
- `CODEX_CLIENT_URL`
- `GAGE_INSTALLED_CLIENT_URL`

如果都没设置，`scripts/run_phase1_8flows.sh` 会自动探测：

```text
http://127.0.0.1:8787/healthz
```

探测成功后，installed-client workflow 会自动启用。

如果你想显式指定：

```bash
export GAGE_CODEX_CLIENT_URL=http://127.0.0.1:8787
```

## 5. 运行 8 条 Workflow

### 5.1 推荐命令

在 GAGE 仓库根目录执行：

```bash
cd /home/amyjx/work/GAGE
RUN_PREFIX=phase1_$(date +%Y%m%d) ./scripts/run_phase1_8flows.sh
```

这个脚本会：

- 依次运行 8 条 workflow
- 给每个 `run_id` 自动追加时间后缀
- 给整批输出目录也自动追加同一时间后缀

默认输出目录格式：

```text
runs/<RUN_PREFIX>_8flows_<HHMMSS>/
```

目录内部每条 run 的名字类似：

```text
<RUN_PREFIX>_<kit>_<scheduler>_<HHMMSS>
```

### 5.2 常用覆盖参数

切换模型：

```bash
cd /home/amyjx/work/GAGE
OLLAMA_MODEL=qwen3-vl:2b-instruct \
RUN_PREFIX=phase1_$(date +%Y%m%d) \
./scripts/run_phase1_8flows.sh
```

指定 Python：

```bash
cd /home/amyjx/work/GAGE
PYTHON_BIN=/path/to/python \
RUN_PREFIX=phase1_$(date +%Y%m%d) \
./scripts/run_phase1_8flows.sh
```

如果你真的需要固定目录，可以显式关闭时间后缀：

```bash
cd /home/amyjx/work/GAGE
RUN_TIME_SUFFIX="" OUTPUT_DIR_EXACT=1 ./scripts/run_phase1_8flows.sh
```

这不建议作为常规验收方式，因为很容易把新旧结果混在一起。

## 6. Run 完之后去哪里检查结果

每个 run 的目录结构大致是：

```text
runs/<batch_dir>/
  <run_id>/
    events.jsonl
    samples.jsonl
    summary.json
    samples/
```

### 6.1 第一层：看 `summary.json`

`summary.json` 用来快速判断：

- 这条 run 是否完整结束
- 最终产出了哪些 metrics
- benchmark 层面的整体结果是什么

重点看：

- `metrics`
- `sample_count`
- `runtime_failure_code`
- benchmark-specific 聚合字段，例如 pass/fail、reward、rate 等

最低检查标准：

- `metrics` 存在且非空
- metric 名字符合当前 benchmark config 预期
- `count` 与实际 sample 数量一致

### 6.2 第二层：看 `samples.jsonl`

`samples.jsonl` 是最适合看 sample 级行为与失败原因的文件。

重点看每个 sample 的：

- `predict_result[0].answer`
- `predict_result[0].agent_trace`
- `eval_result.failure_reason`
- `eval_result.diagnostic_reason`
- `eval_result.diagnostic_details`
- `eval_result.score`

如果你想知道“agent 到底做了什么、为什么失败”，通常先看这里。

### 6.3 第三层：看 per-sample runtime 目录

每个 sample 的 runtime 文件通常在：

```text
samples/runtime/<task_id>/<sample_id>/
```

重点看：

- `runtime_metadata.json`
- `verifier/result.json`
- `artifacts/`
- `logs/raw_error.json`

推荐排查顺序：

1. `samples.jsonl`：标准化后的最终结果
2. `verifier/result.json`：benchmark-native 证据
3. `runtime_metadata.json`：执行上下文
4. `artifacts/`：benchmark-specific 产物
5. `logs/raw_error.json`：原始失败信封

### 6.4 `artifacts/` 里通常会有什么

不同 benchmark kit 的 artifact 形态不同。

`terminal_bench` 常见：

- `tool_trace.json`
- `workspace_diff.json`
- `stdout.log`
- `stderr.log`

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

### 6.4.1 一个很重要的注意事项

`artifacts/` 目录存在，**不代表 artifact 一定已经成功落盘**。

当前 runtime artifact layout 会在 sample 开始时就先创建：

- `artifacts/`
- `verifier/`
- `logs/`

因此会出现一种常见情况：

- `artifacts/` 目录已经存在
- 但目录是空的
- 同时 `logs/raw_error.json`、`runtime_metadata.json`、`verifier/result.json` 已经存在

这通常说明失败发生在 benchmark-specific artifact capture 之前，而不是“目录丢了”。

尤其在 `framework_loop` 路径下，这个现象更常见。因为 framework-loop 的 artifact 主要依赖 benchmark kit 在 `finalize_loop_result()` 中回收；如果 run 在这些阶段提前失败：

- `acquire_lease`
- `bootstrap_runtime`
- `run_scheduler`

那么 `finalize_loop_result()` 根本不会执行，最终就会留下一个空的 `artifacts/` 目录。

因此在检查“artifact 是否正常”时，不要只看目录是否存在，建议同时一起看：

- `logs/raw_error.json`
- `runtime_metadata.json`
- `verifier/result.json`

如果这几个文件已经存在，而 `artifacts/` 为空，优先判断为：

- 失败过早，未进入 benchmark-specific artifact capture

而不是直接判断为：

- artifact sink 写坏了

### 6.5 一份实用验收清单

每条 run 至少检查：

- `summary.json` 里的 `metrics` 是否完整
- `samples.jsonl` 是否有 sample 记录
- `predict_result[0].agent_trace` 是否存在且类型正确
- per-sample 的 `artifacts/` 目录是否存在
- benchmark-specific artifact 是否已经落盘
- 失败是否能通过 `failure_reason`、`diagnostic_reason`、`diagnostic_details` 解释清楚

## 7. 如何新增一个 Benchmark Kit 和新 Workflow

下面是符合 phase 1 分层设计的接入方式。

### 7.1 新建 Kit 目录

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

### 7.2 在 `runtime.py` 里定义生命周期

`runtime.py` 应负责：

- benchmark-owned state 初始化
- runtime-owned context bootstrap
- 暴露 sub-workflow 需要的稳定状态
- benchmark-specific 的 save/finalize lifecycle

不要把 scheduler-specific 的 payload shaping 放在这里。

### 7.3 在 `sub_workflows/*` 里定义 scheduler-specific 投影

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

### 7.4 把共享逻辑放到 `units.py` 和 `artifacts.py`

`units.py` 适合放：

- prompt builder
- 输入投影 helper
- 小型 benchmark-specific normalization helper

`artifacts.py` 适合放：

- artifact 导出
- artifact 命名
- fallback capture
- 诊断文件生成

### 7.5 在 `kit.py` 里注册入口

你的 `load_kit()` 应该返回一个 `BenchmarkKitEntry`，把这些部分接起来：

- runtime entry
- supported schedulers
- workflow resolver
- verifier resource resolver
- trace mapper
- compat shim resolver

workflow resolver 需要明确把不同 scheduler 映射到对应 workflow bundle。

### 7.6 在 `config/custom/<new_kit>/` 下添加配置

至少为两条 surface 各加一份：

- `<new_kit>_framework_loop_*.yaml`
- `<new_kit>_installed_client_*.yaml`

installed-client config 应表达 installed-client runtime path，不应再依赖 framework-loop backend 来充当主执行路径。

### 7.7 先补测试，再加进共享脚本

建议最低测试覆盖：

- `runtime.py` 的 unit tests
- `sub_workflows/framework_loop.py` 的 unit tests
- `sub_workflows/installed_client.py` 的 unit tests
- `artifacts.py` 的 unit tests
- 至少一条单 sample 的 integration test

### 7.8 最后再加到共享 runner

当 kit 自己已经单独能跑通后，再：

1. 把新的 config 加到 `scripts/run_phase1_8flows.sh` 或平行矩阵脚本里
2. 保持统一的 `run_id` 命名模式
3. benchmark-specific 逻辑继续留在 kit 里，不要塞进脚本

## 8. Installed-Client Contract 对新 Kit 的要求

installed-client 模式下，kit 需要向 external client handoff 两类内容：

- `request`
- `environment`

更详细的 HTTP schema 在：

- `docs/installed_client_service_contract.md`

实践上建议这样理解：

- `request`：benchmark-owned instruction 和稳定 client metadata
- `environment`：runtime-owned context，供 installed-client service 访问 benchmark 环境
- installed-client service 自己负责 tool calling、memory 和完整 agent loop

如果新 kit 需要更多 installed-client 字段，优先在 kit 自己的 request projection 中扩展。只有当字段具备通用价值时，再扩展 shared service contract。

## 9. 推荐的调试与验证顺序

新增 kit 或排查 8-workflow 问题时，建议按这个顺序：

1. 先起本地 Codex proxy，并确认 `/healthz`
2. 单独跑一条 workflow：`python run.py --config ...`
3. 先看 `summary.json`
4. 再看 `samples.jsonl`
5. 再看 `verifier/result.json`
6. 再看 `artifacts/`
7. 最后再进入 8-workflow 脚本矩阵

这样更容易判断问题属于哪一层：

- benchmark lifecycle
- scheduler projection
- installed-client service wiring
- verifier wiring
- artifact capture

## 10. 常用命令速查

启动本地 Codex proxy：

```bash
cd /home/amyjx/work/gage-installed-client-stub
./run_local_codex_proxy.sh
```

运行 8-workflow 矩阵：

```bash
cd /home/amyjx/work/GAGE
RUN_PREFIX=phase1_$(date +%Y%m%d) ./scripts/run_phase1_8flows.sh
```

单独运行一条 workflow：

```bash
cd /home/amyjx/work/GAGE
.venv/bin/python run.py \
  --config config/custom/appworld/appworld_agent_demo_runtime_ollama.yaml \
  --run-id appworld_framework_smoke \
  --output-dir runs \
  --max-samples 1
```

检查 installed-client proxy：

```bash
curl -fsS http://127.0.0.1:8787/healthz
```

查看最终输出目录：

```bash
ls runs
```

查某条 run 的关键验收文件：

```bash
find runs/<batch_dir>/<run_id> -maxdepth 3 \( -name summary.json -o -name samples.jsonl -o -name result.json -o -path '*/artifacts/*' \)
```
