# hf_inference_endpoint 一键脚本

目录下提供 HuggingFace Inference Endpoint 的一键运行脚本与模板。所有流程均需要有效的 `HUGGINGFACEHUB_API_TOKEN`（或 `HF_API_TOKEN`）以及可访问/创建的 Endpoint 权限。

## 结构
- `template.yaml`：通用 echo 冒烟模板（文本）。
- `model_matrix.yaml`：echo 批跑模型列表（内置占位 demo，参考 `.example`）。
- `run_all_models.sh`：读取 `model_matrix.yaml` 批量渲染并运行。
- `demo_echo/run.sh`：最简单单模型冒烟（echo 数据集，默认 `max_samples=1`，带占位 endpoint/model，可直接运行）。
- `mmmu/`：MMMU 专用模板与脚本，需 vision 模型的 Endpoint。
  - `template.yaml`
  - `model_matrix.yaml`（内置占位 demo，参考 `.example`）
  - `run.sh`（渲染 `mmmu/model_matrix.yaml` 并运行）

## 快速使用
1. 在 `scripts/oneclick/.env` 或环境变量中配置 `HUGGINGFACEHUB_API_TOKEN`。
2. 直接运行（已内置占位 endpoint/model，若 token 无效会返回 401，但流程可跑通）：
   - Echo：`bash scripts/oneclick/backends/hf_inference_endpoint/demo_echo/run.sh`
   - MMMU：`bash scripts/oneclick/backends/hf_inference_endpoint/mmmu/run.sh`
   - 批量（echo 列表）：`bash scripts/oneclick/backends/hf_inference_endpoint/run_all_models.sh`
   可加 `DRY_RUN=1` 只渲染配置不调用 API。
3. 想改成真实 endpoint：在对应 `model_matrix*.yaml` 中替换 `endpoint_name`/`model_name`、显式设 `reuse_existing/auto_start`。

## 常用环境变量
- `HUGGINGFACEHUB_API_TOKEN` / `HF_API_TOKEN`：HF 认证必需。
- `ENDPOINT_NAME` / `MODEL_NAME`：demo_echo/run.sh 使用；提供其一。
- `DRY_RUN=1`：仅渲染 YAML，不执行 run.py。
- 其他参数（超时、并发、max_new_tokens 等）可在矩阵文件中修改。

> 注意：HF Endpoint 的创建/调用会产生费用，确保已登录并有额度后再去掉 `DRY_RUN`。***
