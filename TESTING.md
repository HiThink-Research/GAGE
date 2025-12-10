# gage-eval 测试体系指南 (2025-12-10)

本指南规范了项目的测试目录结构、分层策略与开发标准。所有测试代码均位于 `tests/` 目录下。

## 1. 目录结构 (Directory Structure)

```text
tests/
├── conftest.py                 # 全局 Fixture (EchoBackend, Trace, DataHelper)
├── pytest.ini                  # Pytest 配置文件 (Markers, PYTHONPATH)
├── unit/                       # [P0] 单元测试 (极速, Mock Everything)
│   ├── core/                   # 框架核心 (Config, Pipeline, Runtime)
│   │   ├── backends/           # Backend 纯逻辑/Mock（参数组装/错误解析/模板渲染）
│   │   ├── config/             # Registry/Manifest/PipelineConfig
│   │   ├── evaluation/         # TaskPlanner/SampleLoop（Mocked）+ observability
│   │   ├── pipeline/           # ReportStep/Cache（Mock I/O）
│   │   └── scripts/            # 辅助脚本逻辑测试
│   ├── assets/                 # 业务逻辑 (Preprocessors, Metrics)
│   │   ├── preprocessors/      # Transform、doc_to_*、字段映射
│   │   ├── metrics/            # ExactMatch/Latency 等
│   │   └── benchmarks/         # 特定 Benchmark 解析（如 MMMU）
│   └── utils/                  # 工具函数 (ChatTemplate, ImageUtils)
├── integration/                # [P1] 集成测试 (模块协作, Echo/Lite Runtime)
│   ├── runtime/                # 调度器与任务编排 (SampleLoop + Echo backend)
│   ├── cli/                    # 命令行入口与参数解析 (Run/Distill)
│   ├── compatibility/          # 旧版/兼容性测试 (独立运行，保持对旧逻辑的兼容)
│   └── backends/               # 真实 GPU/HTTP 集成 (需标记 gpu/network)
├── e2e/                        # [P2] 端到端测试 (真实场景, Baselines)
│   ├── reference_scores/       # 基于 JSON 基线的回归测试
│   └── validation/             # 真实性校验 (Log sink 等)
└── data/                       # 测试数据仓库
    ├── baselines/              # 评分基线 JSON 文件
    ├── samples/                # 多模态样本 (Dummy images/audio/video)
    └── configs/                # 测试用 YAML 配置文件
```

## 2. 运行规范 (Execution)

**注意：严禁直接运行 `python tests/xxx.py`。必须使用 `pytest` 命令以确保环境正确加载。**

### 常用命令
- **日常开发 (Unit)**: 
  ```bash
  pytest -m "fast"
  ```
- **提交前检查 (Unit + Integration)**:
  ```bash
  pytest -m "not gpu and not network"
  ```
- **全量回归**:
  ```bash
  pytest
  ```

### 标记 (Markers)
在 `pytest.ini` 中定义：
- `fast`: (默认) 纯内存测试，无磁盘 I/O。
- `io`: 涉及文件读写。
- `gpu`: 需要 GPU 环境。
- `network`: 需要外网访问。
- `compat`: 兼容性测试。

## 3. 编码规范 (Coding Standards)

### 3.1 环境与路径
- **PYTHONPATH**: `pytest.ini` 已配置 `pythonpath = src`，代码中**禁止**使用 `sys.path.append` 修改路径。
- **环境变量**: **必须**使用 `monkeypatch` fixture 修改 `os.environ`，严禁直接赋值，以防止污染后续测试。
- **数据路径**: 使用 `conftest.py` 提供的 `test_data_dir` fixture 获取 `tests/data/` 绝对路径。

### 3.2 Fixture 使用
优先使用全局 Fixture (`tests/conftest.py`)，减少样板代码：
- `echo_backend`: 提供可配置延迟的 Mock LLM Adapter。
- `mock_trace`: 提供捕获事件的 Trace 对象 (InMemoryRecorder)。
- `temp_workspace`: 自动配置并清理 `GAGE_EVAL_SAVE_DIR` 到临时目录。
- `sample_dataset_factory`: 快速生成标准格式的测试样本。

### 3.3 异步测试
所有 `async` 测试函数必须使用装饰器：
```python
@pytest.mark.asyncio
async def test_async_logic():
    ...
```

### 3.4 遗留脚本
- `tests/e2e/validation/log_sink/run.sh`: 这是一个遗留的 Bash 冒烟脚本，不属于 pytest 体系。长期计划是用 Python 重写并纳入 pytest 管理。