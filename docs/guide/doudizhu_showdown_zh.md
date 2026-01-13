# 斗地主 Showdown 使用指南

## 快速启动

前置条件：
- Node.js + npm
- 首次需要安装前端依赖：`cd frontend/rlcard-showdown && npm install`
- 设置密钥：`OPENAI_API_KEY`（或 `LITELLM_API_KEY`）

一键启动：
```bash
scripts/oneclick/run_doudizhu_showdown.sh
```

启动后脚本会输出：
```
[oneclick] replay url: http://127.0.0.1:<port>/replay/doudizhu?...
```
如果没有自动打开浏览器，请手动打开这条 URL。

常用环境变量：
- `RUN_ID`：运行标识（默认带时间戳）
- `OUTPUT_DIR`：输出目录（默认 `./runs`）
- `FRONTEND_PORT` / `REPLAY_PORT`：前端与回放服务端口（端口被占用会自动顺延）
- `AUTO_OPEN=0`：禁用自动打开浏览器
- `FRONTEND_DIR`：前端目录（默认 `frontend/rlcard-showdown`）

## 回放与输出说明

回放文件路径（默认）：
```
runs/<run_id>/replays/doudizhu_replay_<sample_id>.json
```

回放由 replay server 提供，前端通过 `replay_url` 参数读取。  
如果需要定位回放文件，可从上面的路径直接读取 JSON。

## AI 性格/对话配置

### 1) 系统提示词（性格/风格的主要入口）

当前版本的 `doudizhu_arena_v1` 不会读取 `ai_persona` 字段，  
AI 的“性格/风格”主要通过数据集里的 system prompt 控制。

编辑文件：
`tests/data/Test_Doudizhu_LiteLLM.jsonl`

示例（只展示关键结构）：
```json
{
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are a calm, analytical Doudizhu player. Keep chat short and witty."
        }
      ]
    }
  ]
}
```

注意：
- 所有玩家共享同一份 `messages`。
- 如果要区分玩家性格，需要扩展数据集或改造玩家提示注入逻辑。

### 2) 对话开关与频率

配置位置：
`config/custom/doudizhu_litellm_local.yaml`

示例：
```yaml
role_adapters:
  - adapter_id: doudizhu_arena
    role_type: arena
    params:
      environment:
        impl: doudizhu_arena_v1
        chat_mode: ai-only   # off | ai-only | all
        chat_every_n: 2      # 每 N 步记录一次对话
```

### 3) 采样参数（语气/随机性）

可以为每个玩家设置采样参数（如 `temperature`）：
```yaml
players:
  - player_id: player_0
    type: backend
    ref: doudizhu_player_0
    sampling_params:
      temperature: 0.7
```

## 斗地主启动指令（手动模式）

如需手动启动，可拆成三步：

1) 启动回放服务：
```bash
PYTHONPATH=src /Users/shuo/code/GAGE/.venv/bin/python -m gage_eval.tools.replay_server --port 8000 --replay-dir ./runs
```

2) 启动前端：
```bash
cd frontend/rlcard-showdown
REACT_APP_GAGE_API_URL="http://127.0.0.1:8000" NODE_OPTIONS="--openssl-legacy-provider" npm run start
```

3) 运行后端推理：
```bash
/Users/shuo/code/GAGE/.venv/bin/python run.py --config config/custom/doudizhu_litellm_local.yaml --output-dir runs --run-id doudizhu_showdown_local
```

## 常见问题

- 浏览器打不开页面（ERR_CONNECTION_REFUSED）  
  通常是前端没启动成功或端口被占用。请确认脚本输出的端口号并打开对应 URL。

- Node 报 `ERR_OSSL_EVP_UNSUPPORTED`  
  使用 `NODE_OPTIONS=--openssl-legacy-provider`（脚本已自动加上）。
