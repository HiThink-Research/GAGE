# Run Entrypoints

`scripts/run/` 是运行相关的 canonical 入口目录。

## 目录
- `appworld/`：AppWorld 评测和连通性验证。
- `backends/`：后端 demo、provider matrix 和模板。
- `arenas/`：VizDoom、PettingZoo、Retro Mario、Mahjong、斗地主等游戏入口。
- `common/`：共享 env/python/port/browser helper。

## 快速使用
- 准备环境：`bash scripts/run/prepare_env.sh`
- dummy echo 冒烟：`bash scripts/run/backends/demos/run_demo_echo.sh`
- multi-provider demo：`HF_PROVIDER=together HF_MODEL_NAME=Qwen/Qwen2.5-7B-Instruct HF_API_TOKEN=xxx bash scripts/run/backends/demos/run_multi_provider_http_demo.sh`
- Kimi demo：`OPENAI_API_KEY=xxx bash scripts/run/backends/demos/run_kimi_demo.sh`
- VizDoom：`bash scripts/run/arenas/vizdoom/run.sh --mode human-vs-llm`
- PettingZoo：`bash scripts/run/arenas/pettingzoo/run.sh --game boxing --mode dummy`
- Retro Mario：`bash scripts/run/arenas/retro_mario/run.sh --mode human_ws`
- Mahjong：`bash scripts/run/arenas/mahjong/run.sh --mode human-vs-ai`
- 斗地主：`bash scripts/run/arenas/doudizhu/run.sh --mode showdown`

## 本地状态
- 本地环境变量文件默认读取 `/Users/shuo/code/GAGE/env/scripts/run.env`，其次回退 `/Users/shuo/code/GAGE/env/localenv`。
- 运行产物默认写到 `/Users/shuo/code/GAGE/runs/`。
- 自动渲染的临时配置默认写到 `/Users/shuo/code/GAGE/env/scripts/generated/`。
- 这些文件都不放进 `repo/`。

## 常用环境变量
- `VENV_PATH`：覆盖默认虚拟环境。
- `OUTPUT_DIR` / `RUNS_DIR`：覆盖运行结果目录。
- `HF_API_TOKEN` / `HUGGINGFACEHUB_API_TOKEN`：HF provider 与 endpoint 使用。
- `OPENAI_API_KEY` / `MOONSHOT_API_KEY` / `KIMI_API_KEY` / `LITELLM_API_KEY`：LLM 后端使用。
- `MAX_SAMPLES` / `CONCURRENCY`：低成本冒烟时常用。

运行结果通常包含 `summary.json`、`events.jsonl`、`samples/*.json` 或 `samples/*.jsonl`。
