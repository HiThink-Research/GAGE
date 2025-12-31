# Repository Guidelines

## Code Commenting Standards

### Language Requirement

- **English only**: All comments, docstrings, and TODOs must be written in clear, concise English.

### The "STEP" Protocol (Core Logic)

For complex methods (for example: pipeline execution, main loops), use strictly formatted markers to
divide the logic into distinct stages.

- **Format**: `# STEP <N>: <Brief Description>`
- **Case**: Uppercase `STEP`
- **Placement**: Put the marker immediately above the code block for that step.

Example:

```python
# STEP 1: Load and preprocess the dataset
dataset = self.loader.load(config.data_path)
```

### Docstrings (Google Style)

Use Google-style docstrings for public functions, methods, and classes.

- **Summary line**: Use a concise, command-style description (for example: "Calculates the metric..."
  not "This function calculates...").
- **Optional details**: Add a short explanation if the logic is non-trivial.
- **Sections**: `Args`, `Returns`, and `Raises` when applicable.

Example:

```python
def generate_response(prompt: str, temperature: float = 0.7) -> str:
    """Generates a response from the LLM based on the prompt.

    Args:
        prompt: The input text string.
        temperature: Sampling temperature. Higher values mean more randomness.

    Returns:
        The generated text string.
    """
```

### Type Hints

- **Mandatory**: All public methods and classes must include Python type hints.
- **Goal**: Type hints serve as self-documentation and enable static analysis (mypy/pyright).

### Inline Comments

- **Explain "why"**: Describe intent and rationale, not obvious mechanics.
- **Spacing**: Start with `# ` (hash + space) and separate from code by at least 2 spaces.
- Avoid roadmap comments; track plans in `DEVLOG.md` or design docs instead.

### Special Tags

Use these tags:

- `# TODO(<owner>): <Description> - <Future improvements>` (example: `# TODO(user): ...`)
- `# FIXME: <Description> - <Broken code that needs immediate attention>`
- `# NOTE: <Description> - <Important context or caveats>`

### Golden Sample

```python
from __future__ import annotations

from typing import Any


class EvaluationPipeline:
    """Manages the end-to-end execution of the evaluation tasks."""

    def run(self, dataset_name: str, model_config: dict[str, Any]) -> dict[str, float]:
        """Executes the evaluation pipeline for a specific dataset.

        Args:
            dataset_name: The unique identifier for the dataset in the registry.
            model_config: Configuration dictionary containing model path and params.

        Returns:
            A dictionary mapping metric names to their computed scores.

        Raises:
            ValueError: If the dataset_name is not found in the registry.
        """

        # STEP 1: Initialize model and tokenizer
        # NOTE: Lazy loading is used here to minimize memory overhead during setup
        model = self.model_loader.load(model_config)

        # STEP 2: Load and preprocess dataset
        if dataset_name not in self.registry:
            raise ValueError(f"Dataset {dataset_name} not found.")

        dataset = self.registry.get(dataset_name)
        # We filter out empty samples to prevent model inference errors
        valid_samples = [s for s in dataset if s.content]

        # STEP 3: Batch inference
        predictions: list[str] = []
        for batch in self._create_batches(valid_samples):
            # TODO(panke): Implement async generation for better concurrency
            batch_output = model.generate(batch)
            predictions.extend(batch_output)

        # STEP 4: Compute metrics
        results = self.evaluator.compute(
            predictions=predictions,
            references=[s.label for s in valid_samples],
        )

        return results
```

## Engineering Principles

- Optimize for open-source readiness: readable code, explainable behavior, extensible interfaces.
- Protect two non-negotiables: reproducibility and diagnosability.
- Keep scripts/tests offline-first: no network/model downloads in default flows.
- Treat large/heavy dependencies as optional; keep CPU-only unit tests runnable.

## Architecture Map (Where to Start)

Core abstractions and canonical implementations:

- `PipelineConfig`: `src/gage_eval/config/pipeline_config.py`
- `ConfigRegistry`: `src/gage_eval/config/registry.py`
- `DataManager`: `src/gage_eval/assets/datasets/manager.py`
- `SampleLoop`: `src/gage_eval/evaluation/sample_loop.py`
- `TaskPlanner`: `src/gage_eval/evaluation/task_planner.py`
- `RoleManager`: `src/gage_eval/role/role_manager.py`
- `ObservabilityTrace`: `src/gage_eval/observability/trace.py`
- `EvalCache` (samples/summary writer): `src/gage_eval/evaluation/cache.py`

## Project Structure & Module Organization

- `src/gage_eval/`: framework code (pipeline/runtime, registry, role adapters, observability, support CLI).
- `src/gage_eval/assets/`: built-in datasets, preprocessors, and metrics.
- `config/`: runnable YAML configs.
  - `config/run_configs/`: stable “run” entrypoints.
  - `config/builtin_templates/`: versioned templates (`v1.yaml`, `v2.yaml`, …).
  - `config/custom/`: local experiments and generated configs.
- `tests/`: pytest suites (unit/integration/e2e) and fixtures.
- `docs/`: user/developer guides and schemas.
- `scripts/`: offline, idempotent toolchain scripts (validation/maintenance); do not put one-off scripts here.
- Generated (don’t commit): `runs/`, `.gage_cache/`, `.pytest_cache/`, `dev_docs/` (support workspace).
- `third_party/`: upstream snapshots (read-only). Do not edit; adapt/patch behavior in `src/` and document why.

## Build, Test, and Development Commands

- Install: `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
- Run a demo pipeline: `python run.py --config config/run_configs/demo_echo_run_1.yaml --output-dir runs --run-id demo_echo`
- Support CLI: `PYTHONPATH=src python -m gage_eval.support --help`
- Validate configs: `bash scripts/check_config.sh` (or `python -m gage_eval.tools.config_checker --config <yaml>`)
- Registry manifest:
  - Regenerate: `python scripts/build_registry_manifest.py`
  - Validate: `python scripts/validate_registry_manifest.py`

## Coding Style & Naming Conventions

- Python: 4-space indentation, `snake_case` for functions/files, `CapWords` for classes, `UPPER_SNAKE` for constants.
- New Python files should include `from __future__ import annotations`.
- Prefer built-in generics (`list[str]`, `dict[str, Any]`) and keep `Any` at JSON/registry boundaries only.
- Avoid heavy side effects at import time (no network, model downloads, or large initialization during module import).
- Import order: standard library, third-party, then local imports.
- Comments/docstrings: English only; Google-style docstrings for public APIs; keep complex logic organized with `# STEP <N>:` markers (see "Code Commenting Standards" above).
- Prefer type hints on public surfaces; avoid ad-hoc `sys.path` edits (tests set `pythonpath = src` in `pytest.ini`).

## Configuration & Contracts

- YAML filenames use `snake_case.yaml`; keep reusable templates in `config/builtin_templates/` and runnable entrypoints in `config/run_configs/`.
- IDs must be unique within a PipelineConfig: `dataset_id`, `backend_id`, `adapter_id`, `prompt_id`, `metric_id`.
- Override precedence: task/CLI override > YAML defaults > environment-variable fallback.
- For adapters that merge params (e.g., `context_provider`, `judge_extend`), prefer the standard precedence:
  `implementation_params` < `step.params` < `payload.params`.
- If you change a builtin template definition, re-run `bash scripts/check_config.sh` and ensure template digest checks still pass.
- If you add/change a registry asset, regenerate and validate `registry_manifest.yaml` (`python scripts/build_registry_manifest.py`, then `python scripts/validate_registry_manifest.py`).

## Observability & Run Artifacts

- Use `loguru` for human-readable logs and `ObservabilityTrace` for machine-readable events.
- Trace event names must be `snake_case`; payload must be JSON-serializable.
- When reporting failures, prefer a stable `failure_reason`/`error_type` field in `snake_case` to keep analysis/querying consistent.
- Include minimal context when available: `run_id`, `task_id`, `dataset_id`, `sample_id`, `adapter_id`.
- Large payloads should be written to a file under the run directory (and referenced by path in the payload) instead of inlining big blobs.
- Run artifacts live under `runs/<run_id>/` by default (override with `GAGE_EVAL_SAVE_DIR`):

```text
runs/<run_id>/
  events.jsonl
  samples.jsonl
  summary.json
  samples/
  logs/
```

## Data & Model Governance

- Prefer versioned identifiers (e.g., `dataset_id: foo_v1`) and keep schema validation enabled when possible.
- Use registry-driven preprocessors (`dataset_preprocessors`) to normalize raw rows into the standardized Sample envelope.
- For dataset onboarding, prefer the Support CLI workflow (`python -m gage_eval.support inspect/design/implement`) over ad-hoc scripts.
- Pin remote model revisions when applicable (e.g., HuggingFace `revision`) and never commit model weights or private paths.

## Testing Guidelines

- Run tests via `pytest` (do not run `python tests/...` directly).
- Markers currently registered: `fast`, `io`, `compat` (see `pytest.ini`).
- Common runs: `pytest -m fast`, `pytest -m "not io"`, `pytest` (see `TESTING.md` for tiers/markers).
- Name tests `test_*.py`; keep unit tests fast and mock external I/O/providers.
- Minimal local gate before PR:
  - `bash scripts/check_config.sh`
  - `python scripts/validate_registry_manifest.py`
  - `pytest -m "not io"`

## Commit & Pull Request Guidelines

- This snapshot may not include `.git`; if history is available, match existing subject style. Otherwise use Conventional Commits, e.g. `feat(config): add swebench smoke config`.
- PRs: describe intent + risk, link the config(s) used, include how to reproduce (`python run.py ...`), and add/adjust tests for new behavior.

### Branch Types and Naming Examples

| Branch Type | Example | Description | Base Branch |
| --- | --- | --- | --- |
| `main` | `main` | Stable trunk | N/A |
| `feat/<topic>` | `feat/swebench_timeout` | New features or large changes requiring collaboration | `main` |
| `fix/<topic>` | `fix/cache_buffer_deadlock` | Bug fixes or regressions | `main` |
| `docs/<topic>` | `docs/support_cli_update` | Documentation-only changes | `main` |
| `chore/<topic>` | `chore/registry_manifest_tooling` | Tooling and maintenance scripts | `main` |
| `user/<name>/<topic>` | `user/alice/swebench_trace` | Personal branches (not guaranteed to persist) | `feat/<topic>` |

Additional guidance: `feat/<topic>` is the default base branch; small changes may branch directly from `main`.

## Environment Variables (Quick Reference)

- Boolean env vars accept `1/true/yes/on` (case-insensitive).
- Common knobs: `GAGE_EVAL_SAVE_DIR`, `GAGE_EVAL_MAX_SAMPLES`, `GAGE_EVAL_THREADS`, `GAGE_EVAL_PREFETCH_FACTOR`, `GAGE_EVAL_MAX_INFLIGHT`, `GAGE_EVAL_SEQUENTIAL`.
- Observability knobs: `GAGE_EVAL_OBSERVABILITY`, `GAGE_EVAL_ENABLE_LOG_SINK`, `GAGE_EVAL_LOG_SINK_LEVEL`.
- External credentials (never commit): `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `HUGGINGFACEHUB_API_TOKEN`.
- Full list: `../engineering-norms-1223.md`.

## Security & Configuration Tips

- Never commit secrets; use env vars (e.g. `OPENAI_API_KEY`, `HUGGINGFACEHUB_API_TOKEN`/`HF_API_TOKEN`) or local `.env` files under `scripts/oneclick/` (kept out of VCS).
- Never commit run artifacts (`runs/`) or caches (`.gage_cache/`); keep them local or in a secure storage bucket.
- Do not modify `third_party/` in-place; use adapters or runtime patches in `src/` and document the upstream source.
