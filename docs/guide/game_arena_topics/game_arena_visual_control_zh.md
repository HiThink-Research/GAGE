# Arena Visual 浏览器控制面

中文 | [English](game_arena_visual_control.md)

本文档说明 GameKit 运行使用的统一 Arena Visual 浏览器页面。浏览器路由是 `/sessions/<session_id>?run_id=<run_id>`，页面通过 `/arena_visual/sessions/...` API 路由访问 Python gateway。

![Arena Visual 前后端分离](../../assets/game-arena-frontend-backend-design-20260413.png)

![Arena Visual 浏览器控制页面](../../assets/arena-visual-browser-control-tictactoe-20260409.png)

帧驱动游戏使用同一套控制面。Retro Mario 会把 frame scene 保持在 stage 内，右侧 rail 默认折叠，打开 panel 后再显示：

![Retro Mario 的 Arena Visual 浏览器控制页面](../../assets/arena-visual-browser-control-retro-mario-20260409.png)

## 1. 打开 Session

大多数可视化 GameKit 配置会在以下配置启用时自动打开浏览器：

```yaml
visualizer:
  enabled: true
  mode: arena_visual
  launch_browser: true
```

手动 URL 形态：

```text
http://127.0.0.1:<visual_port>/sessions/<sample_id>?run_id=<run_id>
```

`run_id` query 参数用于区分不同 run 下重复的 sample id。如果页面由 Vite dev server 提供，请通过 `VITE_ARENA_GATEWAY_BASE_URL` 指向 Python gateway。

## 2. 后端与前端资源

普通 GameKit 运行会根据 YAML 里的 `visualizer` 配置启动 Python Arena Visual gateway。gateway 同时提供浏览器路由，以及 `/arena_visual/sessions/...` 下的 JSON、action、timeline、event、chat、control 和 media API。仓库已包含预构建的 `frontend/arena-visual/dist`，Python gateway 会从这里读取浏览器页面。普通用户只需要准备 Python 运行环境、浏览器，以及对应游戏需要的 API key、ROM 或桌面渲染能力；不需要安装 Node/npm，也不需要单独启动前端 dev server。

浏览器页面由 `frontend/arena-visual` 下的 React 应用提供。只有开发、测试、重建这个前端，或发现 `frontend/arena-visual/dist/index.html` 缺失时，才需要前端项目环境。推荐使用当前 Node.js LTS 和仓库自带的 `package-lock.json` 做可复现安装：

```bash
cd frontend/arena-visual
npm ci
npm test
npm run build
```

如果要用 Vite dev server 调试前端，请先启动一个可视化 GameKit 运行，再把 `VITE_ARENA_GATEWAY_BASE_URL` 指向运行时打印的 Python gateway：

```bash
cd frontend/arena-visual
VITE_ARENA_GATEWAY_BASE_URL=http://127.0.0.1:<visual_port> npm run dev
```

此时打开 Vite 输出的本地 URL。`<visual_port>` 是 Python Arena Visual gateway 的端口，不是 Vite 端口。常见环境问题：

- `npm ci` 失败：确认 Node/npm 可用；如果 `node_modules` 是中断安装留下的，删除后重试。
- Vite 页面能打开但没有 session 数据：确认 `VITE_ARENA_GATEWAY_BASE_URL` 指向当前正在运行的 Python gateway。
- 普通 GameKit 页面空白或加载失败：优先确认 `frontend/arena-visual/dist/index.html` 存在，再查看 Python gateway 日志和浏览器控制台；只有在 `dist` 缺失或修改过 `frontend/arena-visual` 后，才需要重新执行前端 build。

## 3. 运行契约

GameKit visual session 共享这些运行契约：

| 契约 | 查看位置 | 含义 |
| --- | --- | --- |
| `ArenaObservation` | `src/gage_eval/role/arena/types.py` | 每个 turn 交给 player 的视图：`board_text`、`view`、`legal_actions`、`metadata` 和可选 prompt 数据。 |
| LLM turn prompt | `src/gage_eval/role/arena/player_drivers/llm_backend.py` | 内置 LLM player 接收 sample messages，并追加由 active player、view text 和 legal moves 生成的一条 user message。 |
| `ArenaAction` | `src/gage_eval/role/arena/types.py` | 环境要执行的 player 输出：`player`、`move`、`raw` 和 metadata。 |
| `GameResult` | `src/gage_eval/role/arena/types.py` | 终局摘要：winner、result、reason、move counts、final board、move log 和 `arena_trace`。 |
| `arena_trace` | `sample.predict_result[0].arena_trace` | scheduler 生成的逐步 trace，包含 action、合法性、耗时、retry、reward 和 timeline 元数据。 |
| `game_arena` footer | `sample.predict_result[0].game_arena` | 报告使用的终局摘要：total steps、winner、termination reason、ranks、scores 或 returns。 |
| Visual session | `runs/<run_id>/replays/<sample_id>/arena_visual_session/v1/manifest.json` | 可回放浏览器产物，包含 session metadata、timeline、scenes 和 media refs。 |

典型可视化运行排查时，先看 `summary.json`，再看 `samples.jsonl` 中的 sample 行，然后看 `replays/<sample_id>/replay.json`、`events.jsonl` 和 `arena_visual_session/v1/manifest.json`。

## 4. 页面区域

| 区域 | 作用 |
| --- | --- |
| Session command deck | 显示 session id、game name、lifecycle、observer、scheduler family 和控制区展开入口。 |
| Stage | 通过 game plugin 渲染棋盘、牌桌或帧画面。 |
| Session controls | Live/replay 传输控制；点击 `Expand session controls` 后显示。 |
| Utility rail | 打开 Control、Players、Events、Chat 和 Trace 面板。 |
| Timeline drawer | 显示最近事件并用于检查事件流。 |

## 5. Session Command Deck

顶部 deck 是操作入口：

- `Expand session controls`：展开传输控制。
- `Collapse session controls`：收起传输控制。
- `Back to host`：返回 Arena Visual host 页面。
- 状态 pill 显示 playback mode、game plugin、lifecycle、scene seq、observer 和 scheduler state。

Deck 与 game stage 保持分离，方便 README 或游戏文档截图只裁剪玩法区域。

## 6. 播放与会话控制

展开 controls 后，页面提供：

| 控件 | 行为 |
| --- | --- |
| `Live tail` | 跟随最新 live scene。 |
| `Pause` | 停止自动 tail，停在当前 scene。 |
| `Replay` | 从当前 cursor 播放记录的 scenes。 |
| `Back to tail` | 从 replay/paused 回到最新 live scene。 |
| `0.5x / 1x / 2x` | 调整 replay 播放速度。 |
| `Step -1 / Step +1` | 可用时向前/向后移动一个 event 或 scene。 |
| `End` | 跳到最新可用 scene。 |
| `Restart` | backend 声明支持时请求 live restart。 |
| `Finish` | 请求受控结束 session。 |

控件不可用通常表示 backend 尚未声明能力、session 仍在加载，或上一个 control command 还未完成。

## 7. Utility Rail 面板

右侧 rail 打开共享面板：

- `Control`：host receipts、current mode、active actor、observer、input transport 和 readiness hints。
- `Players`：observer 选择和 player roster。
- `Events`：当前 scene 的语义事件摘要。
- `Chat`：支持时显示 chat log 和 chat submitter。
- `Trace`：trace 行和更底层的诊断信息。

同一时间只打开一个 utility panel。再次点击当前 rail 按钮即可关闭。

## 8. 人类输入

当 `human_input.enabled: true` 时，浏览器通过 session action route 提交动作：

```text
POST /arena_visual/sessions/<sample_id>/actions?run_id=<run_id>
```

实时配置可以声明 websocket input：

```text
WS /arena_visual/sessions/<sample_id>/actions/ws?run_id=<run_id>
```

棋盘 plugin 通常提交坐标，牌桌 plugin 提交合法动作文本，帧 plugin 提交 action id 或 macro move。最终输入解析由各 GameKit 自己的 input mapper 负责，因此每个游戏 topic 会说明自己的动作形态。

如果输入没有影响对局：

- 确认 session status 是 `live_running`。
- 确认 scheduler 接受 human intent。
- 打开 Control 面板检查 input transport signal。
- 当 sample id 重复时，确认 URL 带了正确的 `run_id`。

## 9. Timeline 与 Replay

Timeline drawer 显示最近事件，例如 `action_intent`、`action_committed`、`snapshot`、`frame_ref`、`chat` 和 `result`。

它适合用来确认浏览器动作是否被接受、scene cursor 是否推进、运行是否到达终局。对已完成运行，replay 使用同一套 `arena_visual_session/v1` 产物。

## 10. 常见排障

| 现象 | 检查项 |
| --- | --- |
| 页面打开后一直 loading | 确认 `runs/<run_id>/replays/<sample_id>/arena_visual_session/v1/manifest.json` 存在。 |
| 打开了错误 run | 添加 `?run_id=<run_id>` 区分重复 sample id。 |
| 控件不可用 | session 可能已关闭、仍在加载，或缺少 backend capability flags。 |
| 人类输入无效 | 检查 `human_input.enabled`、active actor 和 Control 面板的 input transport。 |
| 帧画面空白 | 检查 scene media transport：`http_pull`、`binary_stream` 或 `low_latency_channel`。 |
| 浏览器没有自动打开 | 使用控制台打印的 session URL，或确认 `visualizer.launch_browser: true`。 |
| 前端 dev server 连不上 session | 检查 `VITE_ARENA_GATEWAY_BASE_URL` 和运行时打印的 Python gateway 端口。 |
| 缺少 replay 产物 | 确认运行确实进入了样本执行阶段，且没有在 visual recorder 写出 `arena_visual_session/v1` 前停止。 |

## 11. 相关文档

- [Game Arena 总览](../game_arena_zh.md)
- [五子棋指南](game_arena_gomoku_zh.md)
- [井字棋指南](game_arena_tictactoe_zh.md)
- [斗地主指南](game_arena_doudizhu_zh.md)
- [麻将指南](game_arena_mahjong_zh.md)
- [PettingZoo Atari 指南](game_arena_pettingzoo_atari_zh.md)
- [Retro Mario 指南](game_arena_retro_mario_zh.md)
- [ViZDoom 指南](game_arena_vizdoom_zh.md)
