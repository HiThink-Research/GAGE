# Backend Demos And Matrices

`scripts/run/backends/` 提供两类入口：

- `demos/`：单次冒烟脚本，适合确认 `run.py` 全链路。
- `providers/`：按 provider 维护模板、模型列表和批跑脚本。

## 运行前准备
- 安装依赖：`bash scripts/run/prepare_env.sh`
- 在工作区根目录外配置本地环境变量文件：
  - `/Users/shuo/code/GAGE/env/scripts/run.env`
  - 或 `/Users/shuo/code/GAGE/env/localenv`
- 不再在 `repo/` 内放 `.env`、generated config 或运行结果。

## 常用入口
- dummy echo：`bash scripts/run/backends/demos/run_demo_echo.sh`
- multi-provider demo：`bash scripts/run/backends/demos/run_multi_provider_http_demo.sh`
- Kimi demo：`bash scripts/run/backends/demos/run_kimi_demo.sh`
- multi_provider_http 批跑：`bash scripts/run/backends/providers/multi_provider_http/run_matrix.sh`
- litellm 批跑：`bash scripts/run/backends/providers/litellm/run_matrix.sh`
- litellm mock 批跑：`bash scripts/run/backends/providers/litellm/run_mock_matrix.sh`
- hf endpoint echo：`bash scripts/run/backends/providers/hf_inference_endpoint/run_demo_echo.sh`
- hf endpoint MMMU：`bash scripts/run/backends/providers/hf_inference_endpoint/run_mmmu.sh`
- hf endpoint 批跑：`bash scripts/run/backends/providers/hf_inference_endpoint/run_matrix.sh`
- TGI 批跑：`bash scripts/run/backends/providers/tgi/run_matrix.sh`
- SGLang 批跑：`bash scripts/run/backends/providers/sglang/run_matrix.sh`

## 配置约定
- 模型列表统一命名为 `models.yaml` / `models.example.yaml`。
- 模板统一按用途命名，例如 `template.demo_echo.yaml`、`template.piqa.yaml`、`mmmu.template.yaml`。
- 自动渲染的配置默认落到 `/Users/shuo/code/GAGE/env/scripts/generated/`。
- 运行结果默认落到 `/Users/shuo/code/GAGE/runs/`。

## 凭据
- `HF_API_TOKEN` / `HUGGINGFACEHUB_API_TOKEN`：`multi_provider_http` 与 HF endpoint 使用。
- `OPENAI_API_KEY`、`MOONSHOT_API_KEY`、`KIMI_API_KEY`、`ANTHROPIC_API_KEY`、`GOOGLE_API_KEY`：`litellm` 按 provider 使用。
- 若缺少模型所需 key，`litellm` 的 matrix 脚本会跳过该模型。

## 结果查看
- `summary.json`：汇总指标。
- `events.jsonl`：阶段事件。
- `samples/*.json` 或 `samples/*.jsonl`：逐样本输入输出。
