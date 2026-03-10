# hf_inference_endpoint

这里放 HuggingFace Dedicated Endpoint 的模板和运行脚本。

## 文件
- `demo_echo.template.yaml`
- `demo_echo.models.yaml`
- `demo_echo.models.example.yaml`
- `run_demo_echo.sh`
- `mmmu.template.yaml`
- `mmmu.models.yaml`
- `mmmu.models.example.yaml`
- `run_mmmu.sh`
- `run_matrix.sh`

## 快速使用
1. 在环境变量或工作区外的本地 env 文件中配置 `HUGGINGFACEHUB_API_TOKEN` 或 `HF_API_TOKEN`。
2. 运行 echo 冒烟：
   - `bash scripts/run/backends/providers/hf_inference_endpoint/run_demo_echo.sh`
3. 运行 MMMU：
   - `bash scripts/run/backends/providers/hf_inference_endpoint/run_mmmu.sh`
4. 按模型列表批跑：
   - `bash scripts/run/backends/providers/hf_inference_endpoint/run_matrix.sh`

## 说明
- 默认生成配置写到 `/Users/shuo/code/GAGE/env/scripts/generated/providers/hf_inference_endpoint/`。
- 默认运行结果写到 `/Users/shuo/code/GAGE/runs/`。
- 真实调用会产生费用，执行前确认 token 和 endpoint 权限。
