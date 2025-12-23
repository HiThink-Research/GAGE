# Backend 测试清单与指南

面向新增/改动的后端，提供可运行的 demo、批量脚本，以及当前已知的运行条件/阻塞项。

## 目录结构
- `scripts/oneclick/`：通用一键脚本与单个 demo。
- `scripts/oneclick/backends/<backend>/`：特定后端的模板、模型列表、批跑脚本。
- 生成配置与结果：
  - `.generated/`：自动渲染出的 PipelineConfig。
  - `runs/<...>/`：每次运行的 summary/events/samples。

## 运行前准备
- 安装依赖：`bash scripts/oneclick/prepare_env.sh`（默认 `.venv`）。
- 复制并填写环境变量文件：
  - `cp scripts/oneclick/.env.example scripts/oneclick/.env`
  - 按需填 `HF_API_TOKEN`、`OPENAI_API_KEY`、`MOONSHOT_API_KEY`、`KIMI_API_KEY` 等。

## 后端测试矩阵

### 今日新增需求测试计划（聚焦）
| backend（实现文件） | provider /接口 | model/目标 | 凭据 | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- |
| multi_provider_http<br/>`src/gage_eval/role/model/backends/multi_provider_http_backend.py` | HF Inference Providers（AsyncInferenceClient） | Qwen/Qwen2.5-7B-Instruct（可在 model_matrix 扩展） | HF_API_TOKEN/HUGGINGFACEHUB_API_TOKEN | 待执行 | 单：`run_multi_provider_demo.sh`；多：`backends/multi_provider_http/run_all_models.sh` |
| hf_inference_endpoint<br/>`src/gage_eval/role/model/backends/hf_http_backend.py` | HF Inference Endpoint（Dedicated/TGI） | 自定义 endpoint_name/model_id/instance_type | HUGGINGFACEHUB_API_TOKEN | 待执行 | 单点：`backends/hf_inference_endpoint/demo_echo/run.sh`；通用：`backends/hf_inference_endpoint/run_all_models.sh` |
| litellm<br/>`src/gage_eval/role/model/backends/litellm_backend.py` | openai | gpt-4o-mini | OPENAI_API_KEY | 未执行 | 批跑：`backends/litellm/run_all_models.sh` + `model_matrix.yaml` |
| litellm<br/>`src/gage_eval/role/model/backends/litellm_backend.py` | kimi | moonshot-v1-8k | MOONSHOT_API_KEY/KIMI_API_KEY | kimi 已跑通 | 无 liteLLM 仍可直连 |
| litellm<br/>`src/gage_eval/role/model/backends/litellm_backend.py` | anthropic | claude-3-5-sonnet-20241022 | ANTHROPIC_API_KEY | 未执行 | 需 Anthropic key |
| litellm<br/>`src/gage_eval/role/model/backends/litellm_backend.py` | google | gemini-1.5-flash | GOOGLE_API_KEY | 未执行 | 需 Google Generative AI key |

## 最小可跑的测试方案（含本地替代）
| backend | 真实接口测试 | 本地/低成本替代 | 备注 |
| --- | --- | --- | --- |
| multi_provider_http | 需要 HF_API_TOKEN/HUGGINGFACEHUB_API_TOKEN，运行 `scripts/oneclick/backends/multi_provider_http/run_all_models.sh` | 单元测试：`python -m unittest tests.backends.test_multi_provider_http_backend`（mock AsyncInferenceClient）；无付费 token 时仅能走单测 | HF Inference Providers 需付费额度，无法用本地 OpenAI 兼容服务替代 |
| multi_provider_http 本地冒烟 | - | 在 model_matrix.yaml 设置本地 OpenAI 兼容服务（如 qwen2.5-0.5b-instruct-mlx）再运行 `scripts/oneclick/backends/multi_provider_http/run_all_models.sh` | 仍需 HF_API_TOKEN 通过接口校验 |
| hf_inference_endpoint | 需 HUGGINGFACEHUB_API_TOKEN + 有权限的 endpoint（或创建权限）；`scripts/oneclick/backends/hf_inference_endpoint/demo_echo/run.sh` 或 `run_all_models.sh` | 无 | Dedicated Endpoint 必须真实 HF 权限；创建会计费 |
| litellm（各厂商） | 对应厂商 API key，`scripts/oneclick/backends/litellm/run_all_models.sh` | 若有本地 OpenAI 兼容服务，可在 model_matrix.yaml 填自定义 base_url/model/api_key 进行低成本验证 | 覆盖 openai/kimi/anthropic/google 路径 |

### multi_provider_http（基于 HF Inference Providers：AsyncInferenceClient）
- 单模型 demo：`bash scripts/oneclick/run_multi_provider_demo.sh`
  - 配置模板：`scripts/oneclick/configs/multi_provider_demo.template.yaml`（max_new_tokens=16，max_samples=1）
  - 依赖：`HF_API_TOKEN`（或 `HUGGINGFACEHUB_API_TOKEN`），默认 provider=together，模型 `Qwen/Qwen2.5-7B-Instruct`
- 多模型批跑：`bash scripts/oneclick/backends/multi_provider_http/run_all_models.sh`
  - 模型列表：`scripts/oneclick/backends/multi_provider_http/model_matrix.yaml`（示例 `.example`）
  - 生成配置：`.generated/backends/multi_provider_http/*.yaml`
  - 输出：`runs/mph_matrix/<model>/`

| backend | provider | model | key env | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- |
| multi_provider_http | together | Qwen/Qwen2.5-7B-Instruct | HF_API_TOKEN | 未执行 | 默认 demo 配置 |
| multi_provider_http | together | meta-llama/Meta-Llama-3.1-8B-Instruct | HF_API_TOKEN | 未执行 | 需在 model_matrix.yaml 添加并跑批脚本 |

> 若报 “You must provide an api_key …”，确认 `HF_API_TOKEN` 已导出或登录 `hf auth login`。

### HuggingFace Inference API（Serverless / Dedicated）
- **hf_serverless**（InferenceClient/AsyncInferenceClient，对应 InferenceProvidersClient Serverless API）
  - 计划：自建 PipelineConfig（type: hf_serverless，model_name=<repo>，max_samples=1），用 demo_echo 冒烟。
  - 依赖：`HUGGINGFACEHUB_API_TOKEN`。
- **hf_inference_endpoint**（InferenceEndpointModel Dedicated/TGI）
  - 计划：自建 PipelineConfig（type: hf_inference_endpoint，endpoint_name/model_id/instance_type/...），确保有权限创建/访问；或使用 `backends/hf_inference_endpoint/run_all_models.sh` + `model_matrix.yaml` 生成并批跑；max_samples=1 冒烟。
  - 依赖：`HUGGINGFACEHUB_API_TOKEN`。

| backend | client | model/endpoint | key env | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- |
| hf_serverless | InferenceClient (serverless) | 例：Qwen/Qwen2.5-7B-Instruct | HUGGINGFACEHUB_API_TOKEN | 未执行 | Serverless 文本推理 |
| hf_inference_endpoint | InferenceEndpoint (dedicated) | 自定义 endpoint/model_id | HUGGINGFACEHUB_API_TOKEN | 未执行 | 创建/获取 endpoint，TGI |

### litellm（OpenAI/多厂商统一）
- 单模型 demo：可用通用模板生成（默认 max_new_tokens=16，max_samples=1）。
- 多模型批跑：`bash scripts/oneclick/backends/litellm/run_all_models.sh`
  - 模型列表：`scripts/oneclick/backends/litellm/model_matrix.yaml`（示例 `.example`）
  - 生成配置：`.generated/backends/litellm/*.yaml`
  - 输出：`runs/litellm_matrix/<model>/`

| backend | provider | model | key env | 状态 | 备注 |
| --- | --- | --- | --- | --- | --- |
| litellm | openai | gpt-4o-mini | OPENAI_API_KEY | 未执行 | 需 OpenAI API key |
| litellm | kimi | moonshot-v1-8k | MOONSHOT_API_KEY/KIMI_API_KEY | 未执行 | 需 Kimi/Moonshot key；缺省可走直连 |
| litellm | anthropic | claude-3-5-sonnet-20241022 | ANTHROPIC_API_KEY | 未执行 | 需 Anthropic key |
| litellm | google | gemini-1.5-flash | GOOGLE_API_KEY | 未执行 | 需 Google Generative AI key |

> 若某模型缺少对应 API key，批跑脚本会跳过并提示缺失的 env 名。

### openai_http / 其他
- 可参考 litellm 模板快速创建：`type: openai_http`，配置 `base_url/model/api_key`，用 `demo_echo` 数据、`max_samples=1` 即可。
- 若需要批跑，可仿照上述结构新增子目录和脚本。

## 结果查看
- `summary.json`：耗时、吞吐与指标。
- `events.jsonl`：阶段事件。
- `samples/*.jsonl`：逐样本输入/输出，便于检查 backend 返回的 `model_output.answer/raw_response`。

## 已知缺口
- litellm/openai_http 模型需要对应厂商 API key，仓库中未内置，需自行提供。
- multi_provider_http 依赖 HF API Token；未登录/未设 token 会重试失败。***
