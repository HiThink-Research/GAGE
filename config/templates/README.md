## 模板使用说明

模板目录用于存放云原生风格的 PipelineConfig 示例，帮助 CLI/Portal 快速预填配置。每个模板需要在 `index.yaml` 中登记 `id`、`path`、`description` 与 `version`，并满足以下约定：

1. 模板文件必须声明 `api_version: gage/v1alpha1` 与 `kind: PipelineConfig`。
2. 支持 `builtin` 与 `custom` 两类模板：Builtin 通过 `builtin.pipeline_id` 指定 ID，Custom 直接给出 `custom.steps`。
3. 所有可以覆写的字段需在文件顶部用 YAML 注释说明推荐的 override 方式（环境变量或 CLI `--override`）。
4. 模板引用的 prompt/backend/role adapter ID 必须与 `PipelineConfig` 中定义的对象一致，避免运行期无法解析。
5. 提交模板后需要运行 `scripts/check_config.sh`，确保通过 schema 校验与 TaskPlanSpec 构建。

常用验证命令：

```bash
python -m gage_eval.tools.config_checker \
  --config config/templates/builtin/builtin_llm_eval_compat.yaml
```

如需额外验证运行期实例化，可添加 `--materialize-runtime`，但该模式会尝试读取数据集和拉起后端，请在本地确认资源可用。***
