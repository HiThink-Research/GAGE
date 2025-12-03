# DEVLOG

## 2025-12-08 （补充）

- 为 TGI/SGLang 后端新增单元测试，mock `requests.Session` 覆盖 stop/logprob 请求体与文本解析，确保无真实服务依赖也能自检。
- README 新增 4.10 节（TGI/SGLang HTTP 流程示意）、补充 Mermaid/表格/源码注释，同时更新 5.9 状态表与 7 章计划，标记序号 5/6 任务完成。
- 补充测试清单：`tests/backends/test_tgi_backend.py`、`tests/backends/test_sglang_backend.py`，运行通过并回写任务状态。
- 新增一键模板：`scripts/oneclick/backends/tgi|sglang/run_all_models.sh` + `template.yaml/model_matrix.yaml`，支持 DRY_RUN 生成 custom 配置；对应 dry-run 测试 `tests/backends/test_run_all_models_tgi.py`、`tests/backends/test_run_all_models_sglang.py`。
- 新增 custom 端到端自检：`config/custom/piqa_tgi_unittest.yaml`、`config/custom/piqa_sglang_unittest.yaml`（默认读 `tests/fixtures/piqa_mini.jsonl`），测试 `tests/backends/test_custom_piqa_tgi_sglang_pipeline.py` 启动本地 HTTP mock 跑通 run.py 输出 summary。

## 2025-12-08

- LiteLLM backend 新增 Grok/xAI 基址推断，支持 `XAI_API_KEY/GROK_API_KEY` 与 LiteLLM 统一调用。
- 移除 Kimi HTTP 兜底，改为默认 moonshot 基址 + LiteLLM 重试，统一失败冒泡。
- 增强 OpenAI/Azure 易用性：自动读取 `OPENAI_API_KEY`、`AZURE_OPENAI_ENDPOINT/API_KEY/API_VERSION`，Azure 注入 `api_type=azure` 与默认版本。
- 增补 `custom_llm_provider` 归一化（grok→xai，kimi→moonshot），LiteLLM 调用显式携带 provider，mock 端口统一 `/v1`。
- 更新测试：`tests/backends/test_litellm_backend.py` 覆盖 Grok 基址、Kimi 无兜底重试、Azure 易用配置。
- 文档同步更新第 4.9/5.9/7 章，标记任务 5/6 完成并描述易用性配置与本地 Flask Mock 数据流。
- 更新本地/实测配置：`config/custom/piqa_litellm.yaml` 聚合五后端（OpenAI/Anthropic/Google/Grok/Kimi），mock 端口可直接跑 PIQA，全量可切换真实基座。
- `docs/litellm_testing.md` 补充 Grok/Kimi 本地/实测运行说明与数据流。
- 一键脚本增加 `collect_summaries.py` 汇总 summary，`run_piqa_litellm_local.sh`/`run_all_models.sh` 运行后自动生成 `summary.index.json`，PIQA 配置 `max_samples` 支持 ENV 覆盖。

## 2025-12-03

- 为 `multi_provider_http_backend` 增加 typed config 默认值与 HTTP 重试兜底，规避 wrap_backend 参数缺失导致的异常。
- 为 HF Serverless / Inference Endpoint 后端补充 `async_client` 路径与信号量控制，提供异步调用与延迟回写。
- 新增测试：`tests/backends/test_multi_provider_http_backend.py`、`tests/backends/test_hf_http_backend_async.py`；文档同步新增第 7 章计划表与 5.9 状态表。
- 迁移 lighteval `litellm_model` 关键逻辑：采样参数归一、max_tokens 放宽、stop 清洗与指数重试。
- 新增 Kimi 适配与直连兜底，自动读取 `KIMI_API_KEY/MOONSHOT_API_KEY`，LiteLLM 失败时保证回答路径。
- 新增测试：`tests/backends/test_litellm_backend.py`；README 补充 4.9 小节与任务状态表更新。
