# DEVLOG

## 2025-12-03

- 为 `multi_provider_http_backend` 增加 typed config 默认值与 HTTP 重试兜底，规避 wrap_backend 参数缺失导致的异常。
- 为 HF Serverless / Inference Endpoint 后端补充 `async_client` 路径与信号量控制，提供异步调用与延迟回写。
- 新增测试：`tests/backends/test_multi_provider_http_backend.py`、`tests/backends/test_hf_http_backend_async.py`；文档同步新增第 7 章计划表与 5.9 状态表。
- 迁移 lighteval `litellm_model` 关键逻辑：采样参数归一、max_tokens 放宽、stop 清洗与指数重试。
- 新增 Kimi 适配与直连兜底，自动读取 `KIMI_API_KEY/MOONSHOT_API_KEY`，LiteLLM 失败时保证回答路径。
- 新增测试：`tests/backends/test_litellm_backend.py`；README 补充 4.9 小节与任务状态表更新。
