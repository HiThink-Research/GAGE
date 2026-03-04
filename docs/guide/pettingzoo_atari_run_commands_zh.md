# PettingZoo Atari 启动命令

[English](pettingzoo_atari_run_commands.md) | 中文

本文档列出当前已接入的全部 PettingZoo Atari 游戏脚本化启动命令。
请在项目根目录执行这些命令。

## 1. 标准脚本

| 类型 | 脚本 | 用途 |
| --- | --- | --- |
| 启动游戏 | `scripts/oneclick/run_pettingzoo_game.sh` | PettingZoo 的标准启动入口，支持 AI、Dummy、ws dummy 和 human record |
| 回放 | `scripts/oneclick/run_pettingzoo_replay.sh` | 通过 `run_id` 回放一局已完成对局 |
| 实时查看辅助脚本 | `scripts/oneclick/run_pettingzoo_ws_rgb_viewer.sh` | 适用于 `space_invaders_dummy_ws_rgb` 冒烟验证的专用 ws_rgb 辅助脚本 |
| 运行后立即回放 | `scripts/oneclick/run_game_replay_oneclick.sh` | 先跑一个代表性 PettingZoo 配置，再自动打开 replay |

## 2. 回放脚本

回放一局已完成对局：

```bash
bash scripts/oneclick/run_pettingzoo_replay.sh <run_id>
```

示例：

```bash
bash scripts/oneclick/run_pettingzoo_replay.sh pettingzoo_space_invaders_ai_20260303_120000
```

## 3. AI 启动命令

先设置 API Key：

```bash
export OPENAI_API_KEY="<YOUR_KEY>"
```

全部已接入 AI 游戏：

| Game ID | 脚本命令 |
| --- | --- |
| `basketball_pong` | `bash scripts/oneclick/run_pettingzoo_game.sh --game basketball_pong --mode ai` |
| `boxing` | `bash scripts/oneclick/run_pettingzoo_game.sh --game boxing --mode ai` |
| `combat_plane` | `bash scripts/oneclick/run_pettingzoo_game.sh --game combat_plane --mode ai` |
| `combat_tank` | `bash scripts/oneclick/run_pettingzoo_game.sh --game combat_tank --mode ai` |
| `double_dunk` | `bash scripts/oneclick/run_pettingzoo_game.sh --game double_dunk --mode ai` |
| `entombed_competitive` | `bash scripts/oneclick/run_pettingzoo_game.sh --game entombed_competitive --mode ai` |
| `entombed_cooperative` | `bash scripts/oneclick/run_pettingzoo_game.sh --game entombed_cooperative --mode ai` |
| `flag_capture` | `bash scripts/oneclick/run_pettingzoo_game.sh --game flag_capture --mode ai` |
| `foozpong` | `bash scripts/oneclick/run_pettingzoo_game.sh --game foozpong --mode ai` |
| `ice_hockey` | `bash scripts/oneclick/run_pettingzoo_game.sh --game ice_hockey --mode ai` |
| `joust` | `bash scripts/oneclick/run_pettingzoo_game.sh --game joust --mode ai` |
| `mario_bros` | `bash scripts/oneclick/run_pettingzoo_game.sh --game mario_bros --mode ai` |
| `maze_craze` | `bash scripts/oneclick/run_pettingzoo_game.sh --game maze_craze --mode ai` |
| `othello` | `bash scripts/oneclick/run_pettingzoo_game.sh --game othello --mode ai` |
| `pong` | `bash scripts/oneclick/run_pettingzoo_game.sh --game pong --mode ai` |
| `space_invaders` | `bash scripts/oneclick/run_pettingzoo_game.sh --game space_invaders --mode ai` |
| `space_war` | `bash scripts/oneclick/run_pettingzoo_game.sh --game space_war --mode ai` |
| `surround` | `bash scripts/oneclick/run_pettingzoo_game.sh --game surround --mode ai` |
| `tennis` | `bash scripts/oneclick/run_pettingzoo_game.sh --game tennis --mode ai` |
| `video_checkers` | `bash scripts/oneclick/run_pettingzoo_game.sh --game video_checkers --mode ai` |
| `volleyball_pong` | `bash scripts/oneclick/run_pettingzoo_game.sh --game volleyball_pong --mode ai` |
| `wizard_of_wor` | `bash scripts/oneclick/run_pettingzoo_game.sh --game wizard_of_wor --mode ai` |

## 4. Dummy 启动命令

全部已接入 Dummy 游戏：

| Game ID | 脚本命令 |
| --- | --- |
| `basketball_pong` | `bash scripts/oneclick/run_pettingzoo_game.sh --game basketball_pong --mode dummy` |
| `boxing` | `bash scripts/oneclick/run_pettingzoo_game.sh --game boxing --mode dummy` |
| `combat_plane` | `bash scripts/oneclick/run_pettingzoo_game.sh --game combat_plane --mode dummy` |
| `combat_tank` | `bash scripts/oneclick/run_pettingzoo_game.sh --game combat_tank --mode dummy` |
| `double_dunk` | `bash scripts/oneclick/run_pettingzoo_game.sh --game double_dunk --mode dummy` |
| `entombed_competitive` | `bash scripts/oneclick/run_pettingzoo_game.sh --game entombed_competitive --mode dummy` |
| `entombed_cooperative` | `bash scripts/oneclick/run_pettingzoo_game.sh --game entombed_cooperative --mode dummy` |
| `flag_capture` | `bash scripts/oneclick/run_pettingzoo_game.sh --game flag_capture --mode dummy` |
| `foozpong` | `bash scripts/oneclick/run_pettingzoo_game.sh --game foozpong --mode dummy` |
| `ice_hockey` | `bash scripts/oneclick/run_pettingzoo_game.sh --game ice_hockey --mode dummy` |
| `joust` | `bash scripts/oneclick/run_pettingzoo_game.sh --game joust --mode dummy` |
| `mario_bros` | `bash scripts/oneclick/run_pettingzoo_game.sh --game mario_bros --mode dummy` |
| `maze_craze` | `bash scripts/oneclick/run_pettingzoo_game.sh --game maze_craze --mode dummy` |
| `othello` | `bash scripts/oneclick/run_pettingzoo_game.sh --game othello --mode dummy` |
| `pong` | `bash scripts/oneclick/run_pettingzoo_game.sh --game pong --mode dummy` |
| `space_invaders` | `bash scripts/oneclick/run_pettingzoo_game.sh --game space_invaders --mode dummy` |
| `space_war` | `bash scripts/oneclick/run_pettingzoo_game.sh --game space_war --mode dummy` |
| `surround` | `bash scripts/oneclick/run_pettingzoo_game.sh --game surround --mode dummy` |
| `tennis` | `bash scripts/oneclick/run_pettingzoo_game.sh --game tennis --mode dummy` |
| `video_checkers` | `bash scripts/oneclick/run_pettingzoo_game.sh --game video_checkers --mode dummy` |
| `volleyball_pong` | `bash scripts/oneclick/run_pettingzoo_game.sh --game volleyball_pong --mode dummy` |
| `wizard_of_wor` | `bash scripts/oneclick/run_pettingzoo_game.sh --game wizard_of_wor --mode dummy` |

## 5. ws_rgb 与 Human Record 快捷模式

推荐的 ws_rgb 冒烟验证：

```bash
bash scripts/oneclick/run_pettingzoo_game.sh --game space_invaders --mode ws_dummy
```

Human vs Human record 模式：

```bash
bash scripts/oneclick/run_pettingzoo_game.sh --game space_invaders --mode human_record
```

## 6. 运行后立即回放的一键辅助

如果你想一条命令完成“运行 + 回放”，仍然可以用：

```bash
bash scripts/oneclick/run_game_replay_oneclick.sh --game pettingzoo --mode dummy
```

或者：

```bash
OPENAI_API_KEY="<YOUR_KEY>" \
bash scripts/oneclick/run_game_replay_oneclick.sh --game pettingzoo --mode ai
```

说明：

- 这个辅助脚本使用的是代表性 PettingZoo 配置，不是上面那套完整的按游戏矩阵。
- 如果你要按具体游戏启动，优先使用 `run_pettingzoo_game.sh`。
