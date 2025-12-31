# Contributing to gage-eval

Thanks for taking the time to contribute! This document describes how we collaborate on **gage-eval** and how to add new benchmarks, assets, and features while keeping the framework consistent and maintainable.

Quick links:

- Project home: `README.md` / `README_zh.md`
- Framework overview: `docs/guide/framework_overview.md` / `docs/guide/framework_overview_zh.md`
- Sample schema: `docs/guide/sample.md` / `docs/guide/sample_zh.md`
- Benchmark onboarding (support module): `docs/guide/support_cli.md` / `docs/guide/support_cli_zh.md`
- Testing guide: `TESTING.md`

## Before you start

- **Status**: this project is in internal validation; APIs/configs may evolve quickly. Prefer small, well-scoped PRs.
- **No secrets / private data**: do not commit API keys, customer data, or model weights. Use environment variables for credentials.
- **Be reproducible**: configs, datasets, and judging logic should be runnable by others with minimal setup.

## Design principles (how gage-eval stays scalable)

gage-eval is built around these core ideas:

- **Step chain** orchestrates the workflow: `support -> inference -> judge -> auto_eval` (with fixed `preprocess` + `report`).
- **RoleAdapter** is the glue layer: steps should not embed backend-specific logic; they bind by `adapter_id`.
- **Registry-first extensibility**: new datasets/backends/roles/metrics should be registered as assets and referenced by id from YAML.
- **Unified artifacts**: one run writes `events.jsonl`, `samples.jsonl`, `summary.json` to make debugging and regression easy.

When adding features, try to preserve these separations instead of coupling orchestration + implementation.

## Development setup

### Requirements

- Python 3.10+
- Optional: GPU environment (only required for GPU backends/tests)

### Install

From the project root (`gage-eval-main/`):

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run a minimal demo

```bash
python run.py \
  --config config/run_configs/demo_echo_run_1.yaml \
  --output-dir runs \
  --run-id demo_echo
```

## Where to contribute (common change types)

### Documentation

- Home pages: `README.md`, `README_zh.md`
- Guides: `docs/guide/` (`*_zh.md` for Chinese; English files have no `_zh` suffix)
- Diagrams/assets: `docs/assets/`

If you change user-facing behavior, update docs and keep links valid.

### Add a benchmark (recommended: use the support module)

gage-eval provides a workflow to onboard datasets with less custom glue code:

1) Inspect (sample + schema):

```bash
PYTHONPATH=src python -m gage_eval.support inspect <dataset_or_path> --max-samples 5
```

2) Generate `design.md` (single source of truth):

```bash
PYTHONPATH=src python -m gage_eval.support design <slug>
```

3) Implement (dry-run first, then write):

```bash
PYTHONPATH=src python -m gage_eval.support implement <slug>
PYTHONPATH=src python -m gage_eval.support implement <slug> --force
```

See `docs/guide/support_cli.md` for details and guardrails.

### Add / modify a dataset preprocessor

- Location: `src/gage_eval/assets/datasets/preprocessors/`
- Keep preprocessing aligned with the standardized Sample contract (see `docs/guide/sample.md`).
- Prefer adding a small unit test under `tests/unit/assets/preprocessors/`.

### Add / modify metrics

- Location: `src/gage_eval/metrics/builtin/`
- Add unit tests under `tests/unit/assets/metrics/` (or a relevant adjacent folder).

### Add / modify backends and role adapters

- Model backends: `src/gage_eval/role/model/backends/`
- Role adapters: `src/gage_eval/role/adapters/`

Keep orchestration logic inside steps and keep capability-specific wiring inside adapters/backends.

### Add / modify reporting logic

- Summary generators: `src/gage_eval/reporting/summary_generators/`
- Report step: `src/gage_eval/pipeline/steps/report.py`

Prefer adding new report logic via a generator instead of hardcoding it into the report step.

## Code style expectations

- Follow PEP 8 and keep imports clean.
- Add type hints for public APIs.
- **Comments/docstrings/TODOs must be written in English** (see `AGENTS.md` in `gage-eval-main/`).
- Prefer `loguru` for logging (do not print).

## Tests & validation

### Run tests

See `TESTING.md` for the full testing strategy. Common commands:

```bash
cd gage-eval-main
pytest -m "fast"
pytest -m "not gpu and not network"
```

### Validate configs

Validate builtin templates:

```bash
cd gage-eval-main
bash scripts/check_config.sh
```

Validate a specific YAML:

```bash
cd gage-eval-main
python -m gage_eval.tools.config_checker --config config/custom/<your_config>.yaml
```

### Keep `registry_manifest.yaml` up to date (when adding assets)

If you add new registry assets or change discovery behavior:

```bash
cd gage-eval-main
python scripts/build_registry_manifest.py
python scripts/validate_registry_manifest.py
```

## Pull request checklist

- [ ] Clear scope (one logical change per PR)
- [ ] Docs updated if behavior/config changed
- [ ] Tests added/updated for critical logic
- [ ] `pytest -m "not gpu and not network"` passes (or explain why it cannot)
- [ ] `bash scripts/check_config.sh` passes (if configs/templates changed)
- [ ] `registry_manifest.yaml` updated/validated (if assets/discovery changed)
