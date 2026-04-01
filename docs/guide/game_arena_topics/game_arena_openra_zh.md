# OpenRA Game Arena 指南

[English](game_arena_openra.md) | 中文

这份文档说明当前仓库里的 OpenRA GameKit 接入形态。首版交付采用确定性的 stub backend，因此可以在不启动原生 OpenRA dedicated server 的前提下，完整验证 GameKit、`arena_visual`、human input 和 replay 主链。

## 1. Env 目录

| Env ID | 类型 | 参考内容 | 用途 |
| --- | --- | --- | --- |
| `ra_map01` | 兼容 env | 旧版 stub smoke env | 保持既有 human / llm / dummy 配置不回归 |
| `ra_skirmish_1v1` | RA skirmish | `mods/ra/maps/marigold-town.oramap` | Red Alert 1v1 RTS smoke |
| `cnc_mission_gdi01` | CnC mission | `mods/cnc/maps/gdi01` | 脚本任务 / mission smoke |
| `d2k_skirmish_1v1` | D2K skirmish | `mods/d2k/maps/chin-rock.oramap` | Dune 2000 RTS smoke |

## 2. 标准配置

| 类型 | 路径 | 用途 |
| --- | --- | --- |
| Dummy headless | `config/custom/openra/openra_dummy_gamekit.yaml` | 兼容 `ra_map01` 的后端冒烟验证 |
| Dummy visual | `config/custom/openra/openra_dummy_visual_gamekit.yaml` | 兼容 `ra_map01` 的浏览器验证 |
| RA dummy headless | `config/custom/openra/openra_ra_skirmish_dummy_gamekit.yaml` | `ra_skirmish_1v1` 后端冒烟 |
| RA dummy visual | `config/custom/openra/openra_ra_skirmish_dummy_visual_gamekit.yaml` | `ra_skirmish_1v1` 浏览器验证 |
| CnC dummy headless | `config/custom/openra/openra_cnc_dummy_gamekit.yaml` | `cnc_mission_gdi01` 后端冒烟 |
| CnC dummy visual | `config/custom/openra/openra_cnc_dummy_visual_gamekit.yaml` | `cnc_mission_gdi01` 浏览器验证 |
| D2K dummy headless | `config/custom/openra/openra_d2k_dummy_gamekit.yaml` | `d2k_skirmish_1v1` 后端冒烟 |
| D2K dummy visual | `config/custom/openra/openra_d2k_dummy_visual_gamekit.yaml` | `d2k_skirmish_1v1` 浏览器验证 |
| LLM headless | `config/custom/openra/openra_llm_headless_gamekit.yaml` | 一个 LLM 席位对战脚本化对手 |
| LLM visual | `config/custom/openra/openra_llm_visual_gamekit.yaml` | 带 `arena_visual` 的 LLM 运行路径 |
| Human visual | `config/custom/openra/openra_human_visual_gamekit.yaml` | 浏览器里的 human-vs-dummy 演示 |
| 样本数据 | `tests/data/Test_OpenRA.jsonl` | 上述配置默认使用的单样本 fixture |

## 3. 推荐验证顺序

1. 先跑 `openra_ra_skirmish_dummy_gamekit.yaml`，确认多 env 的 GameKit 主链可用。
2. 再跑 `openra_cnc_dummy_visual_gamekit.yaml` 和 `openra_d2k_dummy_visual_gamekit.yaml`，确认不同 env 的浏览器页、RTS scene projection 和媒体加载正确。
3. 最后跑 `openra_human_visual_gamekit.yaml`，确认浏览器输入、全屏切换和动作提交可用。

## 4. 启动命令

Headless 冒烟：

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src python run.py \
  --config config/custom/openra/openra_dummy_gamekit.yaml \
  --output-dir runs \
  --run-id openra_dummy_headless
```

RA skirmish 冒烟：

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src python run.py \
  --config config/custom/openra/openra_ra_skirmish_dummy_gamekit.yaml \
  --output-dir runs \
  --run-id openra_ra_skirmish_dummy
```

Visual 冒烟：

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src python run.py \
  --config config/custom/openra/openra_dummy_visual_gamekit.yaml \
  --output-dir runs \
  --run-id openra_dummy_visual
```

CnC visual 冒烟：

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src python run.py \
  --config config/custom/openra/openra_cnc_dummy_visual_gamekit.yaml \
  --output-dir runs \
  --run-id openra_cnc_dummy_visual
```

Human visual：

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src python run.py \
  --config config/custom/openra/openra_human_visual_gamekit.yaml \
  --output-dir runs \
  --run-id openra_human_visual
```

## 5. Human player 怎么玩

启动 `openra_human_visual_gamekit.yaml` 后：

1. 打开运行时打印出来的 `arena_visual` 页面。
2. 点击舞台右上角的 `Fullscreen`，把游戏内容区块放大；退出时点 `Exit fullscreen`。
3. 观察页面底部的 `Legal Actions` 卡片。每个 action chip 都对应一个结构化 OpenRA 动作。
4. 点击某个 chip，即可为 `player_0` 提交该动作。
5. 右侧面板会同步显示 credits、power、objectives、selection、units 和 production queues。

当前 human 交互是“schema-first”的首版：

- `Select units`：切换当前选中单位
- `Issue command`：发送预设战术指令
- `Queue production`：把单位加入当前生产队列
- `Camera pan`：提交观察者式镜头移动动作

也就是说，首版不是鼠标框选地图上的任意像素点，而是先验证“浏览器动作 -> InputMapper -> GameKit -> RTS 画面反馈”这条正式主链。

## 6. 当前限制

现阶段还没有启动原生 OpenRA dedicated server。`backend_mode: real` 先作为未来 bridge 的预留位；当前自带配置统一保持 `backend_mode: dummy`，以保证测试、自测和浏览器链路可重复、可稳定执行。

## 7. 自测命令

后端自测统一使用 `gage-eval` conda 环境：

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src pytest \
  tests/unit/game_kits/test_visualization_specs.py \
  tests/game_arena/game_kits/test_runtime_binding_resolver.py \
  tests/unit/role/arena/visualization/test_frame_projection.py \
  tests/unit/role/arena/test_input_mapping.py \
  tests/integration/runtime/test_gamekit_config_inventory.py \
  tests/integration/runtime/test_human_visual_gamekit_configs.py \
  tests/integration/runtime/test_non_human_gamekit_run_matrix.py \
  tests/integration/arena/test_visual_gamekit_browser_matrix.py \
  -k openra -v
```

如果本地没有启动 `http://127.0.0.1:1234/v1` 的 LLM 服务，建议先做 dummy / human 自测：

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src pytest \
  tests/game_arena/game_kits/test_runtime_binding_resolver.py \
  tests/integration/runtime/test_gamekit_config_inventory.py \
  tests/integration/runtime/test_human_visual_gamekit_configs.py \
  tests/integration/runtime/test_non_human_gamekit_run_matrix.py \
  tests/integration/arena/test_visual_gamekit_browser_matrix.py \
  -k "openra and not llm" -v
```
