# Oneclick Wrappers

`scripts/oneclick/` is retained for local shortcuts. The canonical launchers live
under `scripts/run/`; new documentation should link to
[`../run/README.md`](../run/README.md).

## Current Targets

Use the canonical command directly when documenting or debugging a launch.

| Shortcut family | Canonical target |
| --- | --- |
| PettingZoo game | `scripts/run/arenas/pettingzoo/run.sh` |
| Doudizhu game | `scripts/run/arenas/doudizhu/run.sh` |
| Mahjong game | `scripts/run/arenas/mahjong/run.sh` |
| Retro Mario game | `scripts/run/arenas/retro_mario/run.sh` |
| ViZDoom game | `scripts/run/arenas/vizdoom/run.sh` |
| ViZDoom comparison | `scripts/run/arenas/vizdoom/compare.sh` |
| Arena Visual artifact opener | `scripts/run/arenas/replay/run_and_open.sh` |

## Environment

All wrappers inherit the same local environment handling as `scripts/run/common/env.sh`:

- `GAGE_WORKSPACE_ROOT` selects the workspace root.
- `GAGE_RUNS_DIR` selects the output directory.
- `PYTHON_BIN` selects the Python interpreter.
- `VENV_PATH`, `VIRTUAL_ENV`, and `CONDA_PREFIX` are considered by the Python helper.

The wrapper names are local affordances, not the source of truth.
