# gage-eval 测试体系指南 (2026-03-26)

本指南定义 `tests/` 目录的唯一组织规则。顶层只按测试层级划分，运行条件通过
pytest marker 表达，支撑模块统一收敛到 `_support/`。

## 1. 目录结构 (Directory Structure)

```text
tests/
├── conftest.py
├── _support/                  # 非 pytest 入口的 helper / stub
│   ├── helpers/
│   └── stubs/
├── unit/                      # [P0] 单元测试
│   ├── appworld/
│   ├── assets/
│   │   ├── datasets/
│   │   │   ├── loaders/
│   │   │   └── validation/
│   │   ├── metrics/
│   │   └── preprocessors/
│   ├── config/
│   ├── core/
│   ├── evaluation/
│   ├── judge/
│   ├── mcp/
│   ├── observability/
│   ├── pipeline/
│   ├── registry/
│   ├── role/
│   │   ├── arena/
│   │   └── model/
│   ├── runtime/
│   ├── sandbox/
│   ├── tools/
│   └── utils/
├── integration/               # [P1] 集成测试
│   ├── appworld/
│   ├── arena/
│   ├── compatibility/
│   │   ├── backends/
│   │   └── utils/
│   ├── runtime/
│   │   ├── evaluation/
│   │   └── preprocess/
│   └── scenarios/
│       ├── cli/
│       ├── distill/
│       └── metrics/
├── e2e/                       # [P2] 端到端测试
│   ├── reference_scores/
│   └── validation/
│       └── log_sink/
├── data/                      # 静态样本、基线数据
└── fixtures/                  # 轻量 fixture 资产
```

### 目录规则
- 顶层只允许 `_support`、`unit`、`integration`、`e2e`、`data`、`fixtures`。
- `tests/_support/` 里的模块不以 `test_*.py` 命名，也不参与 pytest 收集。
- 同一个测试文件 basename 只能出现一次，避免收集歧义和维护分叉。
- 新增测试时，先判断层级，再判断责任域；不要再创建根层业务桶目录。

## 2. 运行规范 (Execution)

**严禁直接运行 `python tests/xxx.py`。所有测试必须通过 `pytest` 执行。**

### 常用命令
- 日常开发：
  ```bash
  pytest -m fast
  ```
- 本地提交前：
  ```bash
  pytest -m "not gpu and not network"
  ```
- 布局检查：
  ```bash
  python scripts/check_test_layout.py
  ```
- 全量回归：
  ```bash
  pytest
  ```

### 标记 (Markers)
在 [pytest.ini](pytest.ini) 中定义：
- `fast`: 纯内存测试，无磁盘 I/O。
- `io`: 涉及文件读写。
- `gpu`: 需要 GPU 资源。
- `network`: 需要网络访问。
- `compat`: 兼容性测试。

## 3. 编码规范 (Coding Standards)

### 3.1 环境与路径
- `pytest.ini` 已配置 `pythonpath = src`，测试文件不得再手动修改 `PYTHONPATH`。
- 环境变量修改必须使用 `monkeypatch`，避免污染后续用例。
- 测试数据统一放在 `tests/data/` 或 `tests/fixtures/`，不要分散到临时目录树。

### 3.2 Fixture 使用
优先复用 [tests/conftest.py](tests/conftest.py) 中的公共 fixture：
- `test_data_dir`
- `temp_workspace`
- `mock_trace`
- `sample_dataset_factory`
- `media_assets`

### 3.3 Helper / Stub 约束
- 可复用的 helper、stub、probe 统一放入 `tests/_support/`。
- `_support/` 中的文件命名不得使用 `test_*.py`，防止被 pytest 误收集。
- 测试代码引用共享支撑逻辑时，优先从 `tests._support.*` 导入。

### 3.4 异步测试
所有 `async` 测试函数必须显式使用异步标记：

```python
@pytest.mark.asyncio
async def test_async_logic() -> None:
    ...
```

### 3.5 遗留脚本
- `tests/e2e/validation/log_sink/run.sh` 仍是遗留 Bash 冒烟脚本。
- 长期目标是将该脚本迁移为 pytest 管理的 Python 用例，但在完成重写前保留该入口。
