# One-click backend test scripts

This folder holds small shell scripts and ready-to-run configs to smoke test the full `run.py` pipeline, either with the built-in dummy echo backend or the `multi_provider_http` backend you are working on.

## Quick start
- 准备环境（可重复执行）：`bash scripts/oneclick/prepare_env.sh`
- 验证流程（dummy 后端，无需外部依赖）：`bash scripts/oneclick/run_demo_dummy.sh`
- 真实多 provider 路径（需要 HF_API_TOKEN 等，默认只跑 1 条样本）：`HF_PROVIDER=together HF_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct HF_API_TOKEN=xxx bash scripts/oneclick/run_multi_provider_demo.sh`
- 一键串跑（依次执行 dummy + multi provider，multi provider 会自动用环境变量/默认值，默认各跑 1 条）：`bash scripts/oneclick/run_all.sh`
- Kimi（OpenAI 兼容接口，默认 1 条样本）：`OPENAI_API_KEY=xxx bash scripts/oneclick/run_kimi_demo.sh`（也支持 `MOONSHOT_API_KEY` / `KIMI_API_KEY` 或 `.env`）
- multi_provider_http 多模型批跑：`cp scripts/oneclick/backends/multi_provider_http/model_matrix.example.yaml scripts/oneclick/backends/multi_provider_http/model_matrix.yaml`（按需编辑），然后 `bash scripts/oneclick/backends/multi_provider_http/run_all_models.sh`
- litellm 多模型批跑：`cp scripts/oneclick/backends/litellm/model_matrix.example.yaml scripts/oneclick/backends/litellm/model_matrix.yaml`（填好 API key 对应的 env 名），然后 `bash scripts/oneclick/backends/litellm/run_all_models.sh`

## 环境变量
- `VENV_PATH`（可选）：虚拟环境路径，默认 `.venv`。
- `HF_PROVIDER` / `HF_MODEL_NAME` / `HF_TOKENIZER_NAME` / `HF_API_TOKEN`：多 provider 后端使用的 provider / 模型 / tokenizer / token。
- `OPENAI_API_KEY` / `MOONSHOT_API_KEY` / `KIMI_API_KEY`：Kimi（moonshot）OpenAI 兼容接口所需的 token，最终会映射到 `OPENAI_API_KEY`。
- `LITELLM_API_KEY`：litellm 默认使用的 key，若模型指定 `api_key_env`，也可单独提供（如 OPENAI_API_KEY）。
- `MAX_SAMPLES` / `CONCURRENCY`：运行样本数与并发度，run 脚本默认 `1`，便于低成本冒烟。
- `OUTPUT_DIR`：落盘目录，默认 `./runs/<auto>`。
- 也支持在 `scripts/oneclick/.env` 写入 `export HF_API_TOKEN=xxx` 等变量，`run_multi_provider_demo.sh` 会自动加载。
- multi_provider_http 批跑使用 `scripts/oneclick/backends/multi_provider_http/model_matrix.yaml` 管理模型列表（示例见 `.example`），`run_all_models.sh` 会顺序跑完并写入 `runs/mph_matrix/<model_name>/`。
- litellm 批跑使用 `scripts/oneclick/backends/litellm/model_matrix.yaml` 管理模型列表（示例见 `.example`），缺少对应 API key 的模型会被跳过并提示。

## 产出位置
每次运行会在 `runs/<run_id>`（或你指定的 `OUTPUT_DIR`）下生成：
- `summary.json`：指标、耗时、throughput。
- `events.jsonl`：执行阶段事件。
- `samples/*.jsonl`：逐样本输入/输出（可用来检查后端返回的 `model_output`）。

脚本仅依赖 Bash + Python 标准库，实际推理由 `run.py` 调用项目内代码完成。***
