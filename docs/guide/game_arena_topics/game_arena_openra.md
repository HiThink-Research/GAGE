# OpenRA Game Arena Guide

English | [中文](game_arena_openra_zh.md)

This guide covers the current OpenRA GameKit integration in `gage-eval-main`.
The first shipped version uses a deterministic stub backend so the full
GameKit, `arena_visual`, human input, and replay contracts can be verified
without a native OpenRA bridge process.

## 1. Env catalog

| Env ID | Type | Reference content | Purpose |
| --- | --- | --- | --- |
| `ra_map01` | Compatibility env | legacy stub smoke env | Keeps the original human / llm / dummy configs stable |
| `ra_skirmish_1v1` | RA skirmish | `mods/ra/maps/marigold-town.oramap` | Red Alert 1v1 RTS smoke |
| `cnc_mission_gdi01` | CnC mission | `mods/cnc/maps/gdi01` | Scripted mission smoke |
| `d2k_skirmish_1v1` | D2K skirmish | `mods/d2k/maps/chin-rock.oramap` | Dune 2000 RTS smoke |

## 2. Standard configs

| Type | Path | Purpose |
| --- | --- | --- |
| Dummy headless | `config/custom/openra/openra_dummy_gamekit.yaml` | Compatibility smoke test for `ra_map01` |
| Dummy visual | `config/custom/openra/openra_dummy_visual_gamekit.yaml` | Compatibility browser test for `ra_map01` |
| RA dummy headless | `config/custom/openra/openra_ra_skirmish_dummy_gamekit.yaml` | Headless smoke for `ra_skirmish_1v1` |
| RA dummy visual | `config/custom/openra/openra_ra_skirmish_dummy_visual_gamekit.yaml` | Browser smoke for `ra_skirmish_1v1` |
| CnC dummy headless | `config/custom/openra/openra_cnc_dummy_gamekit.yaml` | Headless smoke for `cnc_mission_gdi01` |
| CnC dummy visual | `config/custom/openra/openra_cnc_dummy_visual_gamekit.yaml` | Browser smoke for `cnc_mission_gdi01` |
| D2K dummy headless | `config/custom/openra/openra_d2k_dummy_gamekit.yaml` | Headless smoke for `d2k_skirmish_1v1` |
| D2K dummy visual | `config/custom/openra/openra_d2k_dummy_visual_gamekit.yaml` | Browser smoke for `d2k_skirmish_1v1` |
| LLM headless | `config/custom/openra/openra_llm_headless_gamekit.yaml` | One LLM seat against a scripted opponent |
| LLM visual | `config/custom/openra/openra_llm_visual_gamekit.yaml` | LLM runtime plus `arena_visual` |
| Human visual | `config/custom/openra/openra_human_visual_gamekit.yaml` | Browser-driven human-vs-dummy walkthrough |
| Sample dataset | `tests/data/Test_OpenRA.jsonl` | Single-sample fixture used by the shipped configs |

## 3. Recommended verification order

1. Run `openra_ra_skirmish_dummy_gamekit.yaml` to verify the multi-env GameKit path.
2. Run `openra_cnc_dummy_visual_gamekit.yaml` and `openra_d2k_dummy_visual_gamekit.yaml` to verify browser loading, RTS scene projection, and media streaming across env variants.
3. Run the human visual config to verify browser input, fullscreen, and action submission.

## 4. Commands

Headless smoke:

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src python run.py \
  --config config/custom/openra/openra_dummy_gamekit.yaml \
  --output-dir runs \
  --run-id openra_dummy_headless
```

RA skirmish smoke:

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src python run.py \
  --config config/custom/openra/openra_ra_skirmish_dummy_gamekit.yaml \
  --output-dir runs \
  --run-id openra_ra_skirmish_dummy
```

Visual smoke:

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src python run.py \
  --config config/custom/openra/openra_dummy_visual_gamekit.yaml \
  --output-dir runs \
  --run-id openra_dummy_visual
```

CnC visual smoke:

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src python run.py \
  --config config/custom/openra/openra_cnc_dummy_visual_gamekit.yaml \
  --output-dir runs \
  --run-id openra_cnc_dummy_visual
```

Human visual:

```bash
cd /Users/panke/AI-learning/eval-framework/new/gage-eval-main
/Users/panke/miniconda3/bin/conda run -p /Users/panke/miniconda3/envs/gage-eval \
  env PYTHONPATH=src python run.py \
  --config config/custom/openra/openra_human_visual_gamekit.yaml \
  --output-dir runs \
  --run-id openra_human_visual
```

## 5. How the human player plays

After the human visual config starts:

1. Open the `arena_visual` browser page printed by the runtime.
2. Use the `Fullscreen` button in the stage chrome to expand the game content area. Use `Exit fullscreen` to leave.
3. Review the `Legal Actions` card. Each chip corresponds to one structured OpenRA action.
4. Click a chip to submit that action for `player_0`.
5. Use the right-side panels to track credits, power, objectives, selection, units, and production queues.

Current human interaction is intentionally schema-first:

- `Select units` updates the current selection.
- `Issue command` sends a pre-baked tactical order.
- `Queue production` enqueues a unit for the current production building.
- `Camera pan` submits an observer-style camera action.

## 6. Current limitation

The shipped integration does not launch a native OpenRA dedicated server yet.
`backend_mode: real` is reserved for the future bridge; the current configs keep
`backend_mode: dummy` so tests and browser flows remain deterministic.

## 7. Self-check command

Backend verification is expected to run inside the `gage-eval` conda environment:

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

If the local LLM endpoint at `http://127.0.0.1:1234/v1` is not running, use the dummy / human subset first:

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
