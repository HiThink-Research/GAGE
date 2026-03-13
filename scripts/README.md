# Scripts Layout

`scripts/` 现在按职责拆成四类：

- `scripts/run/`：对外可引用的运行入口。
- `scripts/tools/`：维护仓库、配置和 registry 的工程脚本。
- `scripts/verify/`：诊断和回归验证脚本。
- `scripts/dev/` / `scripts/assets/`：开发辅助与资源处理脚本。

本地文件不再放进 `repo/`：

- 本地环境变量文件放到工作区根目录外侧的 `env/`，例如 `/Users/shuo/code/GAGE/env/scripts/run.env`。
- 运行产物放到工作区根目录外侧的 `runs/`，例如 `/Users/shuo/code/GAGE/runs/`。
- `repo/` 内不再承载 `.env`、generated config、运行日志这类不提交内容。

如果文档需要引用运行命令，只使用 `scripts/run/**` 下的 canonical 路径。
