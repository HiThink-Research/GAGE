# Phase 1 8-Workflow Run And Extension Guide

This guide explains:

- how to run the phase 1 eight-workflow matrix,
- how to set up the local Codex proxy for installed-client workflows,
- how to verify run outputs,
- and how to add one new benchmark kit with new workflows without breaking the phase 1 layering model.

This document follows the phase 1 runtime design in:

- `site/documents/gage-agent-phase1-development-plan-0408.md`
- `docs/installed_client_service_contract.md`

## 1. What The 8 Workflows Are

The phase 1 matrix contains four benchmark kits and two scheduler surfaces:

- `terminal_bench` + `framework_loop`
- `terminal_bench` + `installed_client`
- `swebench` + `framework_loop`
- `swebench` + `installed_client`
- `tau2` + `framework_loop`
- `tau2` + `installed_client`
- `appworld` + `framework_loop`
- `appworld` + `installed_client`

The shared runner is:

- `scripts/run_phase1_8flows.sh`

## 2. Layering Rules

When running or extending this matrix, keep these boundaries stable:

- `runtime.py` owns benchmark lifecycle and runtime-owned state.
- `sub_workflows/framework_loop.py` owns framework-loop projection and normalization.
- `sub_workflows/installed_client.py` owns installed-client request projection and artifact capture.
- `units.py` owns reusable benchmark-specific prompt/input helpers.
- `artifacts.py` owns benchmark-specific artifact persistence and normalization.
- `judge_bridge.py` owns verifier resource binding.
- installed-client tool calling and memory are owned by the external client service, not by GAGE framework-loop code.

If a new behavior is benchmark-specific, prefer adding it under the benchmark kit instead of the shared runtime core.

## 3. Prerequisites

### 3.1 Base Environment

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The runner expects:

- `docker` in `PATH`
- `.venv/bin/python` to exist, or `PYTHON_BIN` to point to a Python executable
- local benchmark dependencies to be available for the selected kits

### 3.2 Benchmark Inputs

The script uses these defaults unless you override them:

```bash
export OLLAMA_BASE_URL=http://127.0.0.1:11434/v1
export OLLAMA_MODEL=qwen3-vl:2b-instruct
export OLLAMA_API_KEY=dummy
export SWEBENCH_LOCAL_PATH=/home/amyjx/work/GAGE/local-datasets/swebench_pro
export TAU2_DATA_DIR=/home/amyjx/work/tau2-bench/data
export TAU2_MAX_STEPS=12
export TAU2_MAX_ERRORS=4
export TAU2_MAX_TURNS=12
export TAU2_USER_MODEL=ollama_chat/qwen3-vl:2b-instruct
```

Adjust these if your local paths or model names differ.

## 4. Set Up The Local Codex Proxy

Installed-client workflows call an HTTP installed-client service. In the local phase 1 setup, that service is provided by the local Codex proxy repo outside GAGE:

- `/home/amyjx/work/gage-installed-client-stub`

### 4.1 Start The Proxy

```bash
cd /home/amyjx/work/gage-installed-client-stub
./run_local_codex_proxy.sh
```

If `codex` is not in `PATH`, point the proxy at it explicitly:

```bash
cd /home/amyjx/work/gage-installed-client-stub
CODEX_EXECUTABLE=/path/to/codex ./run_local_codex_proxy.sh
```

The proxy listens on:

```text
http://127.0.0.1:8787
```

### 4.2 Check That The Proxy Is Healthy

```bash
curl -fsS http://127.0.0.1:8787/healthz
```

Expected fields include:

- `status`
- `codex_available`
- `codex_executable`

### 4.3 How The Runner Finds The Proxy

The runner uses the first non-empty value among:

- `GAGE_CODEX_CLIENT_URL`
- `CODEX_CLIENT_URL`
- `GAGE_INSTALLED_CLIENT_URL`

If none is set, `scripts/run_phase1_8flows.sh` auto-probes:

```text
http://127.0.0.1:8787/healthz
```

If that probe succeeds, installed-client workflows are enabled automatically.

If you want to set it explicitly:

```bash
export GAGE_CODEX_CLIENT_URL=http://127.0.0.1:8787
```

## 5. Run The 8 Workflows

### 5.1 Recommended Command

From the GAGE repo root:

```bash
cd /home/amyjx/work/GAGE
RUN_PREFIX=phase1_$(date +%Y%m%d) ./scripts/run_phase1_8flows.sh
```

This script:

- runs all eight workflows,
- adds a timestamp suffix to each `run_id`,
- and writes the aggregate batch to a timestamped output directory.

The default output directory pattern is:

```text
runs/<RUN_PREFIX>_8flows_<HHMMSS>/
```

Each individual run inside that directory also gets the same suffix:

```text
<RUN_PREFIX>_<kit>_<scheduler>_<HHMMSS>
```

### 5.2 Useful Overrides

Run with a different model:

```bash
cd /home/amyjx/work/GAGE
OLLAMA_MODEL=qwen3-vl:2b-instruct \
RUN_PREFIX=phase1_$(date +%Y%m%d) \
./scripts/run_phase1_8flows.sh
```

Run a smoke batch with a custom Python executable:

```bash
cd /home/amyjx/work/GAGE
PYTHON_BIN=/path/to/python \
RUN_PREFIX=phase1_$(date +%Y%m%d) \
./scripts/run_phase1_8flows.sh
```

Disable the timestamp suffix only when you intentionally want a fixed directory:

```bash
cd /home/amyjx/work/GAGE
RUN_TIME_SUFFIX="" OUTPUT_DIR_EXACT=1 ./scripts/run_phase1_8flows.sh
```

That mode is not recommended for normal validation because it makes old and new results easier to confuse.

## 6. Where To Verify A Run

After a run finishes, inspect each workflow directory under the batch output directory.

Example shape:

```text
runs/<batch_dir>/
  <run_id>/
    events.jsonl
    samples.jsonl
    summary.json
    samples/
```

### 6.1 First Check: `summary.json`

Use `summary.json` to answer:

- did the run finish,
- what metrics were produced,
- and what the top-level benchmark outcome looked like.

Check:

- `metrics`
- `sample_count`
- `runtime_failure_code`
- benchmark-specific aggregates such as pass/fail or reward metrics

Minimum validation:

- `metrics` exists and is not empty
- metric names match the benchmark config you expected
- counts are consistent with the sample count you ran

### 6.2 Second Check: `samples.jsonl`

Use `samples.jsonl` to inspect sample-level outcomes and diagnostics.

For each sample, inspect:

- `predict_result[0].answer`
- `predict_result[0].agent_trace`
- `eval_result.failure_reason`
- `eval_result.diagnostic_reason`
- `eval_result.diagnostic_details`
- `eval_result.score`

This is the best single file to understand what the agent did and why the sample passed or failed.

### 6.3 Third Check: Per-Sample Runtime Files

Under each run:

```text
samples/runtime/<task_id>/<sample_id>/
```

Check these files:

- `runtime_metadata.json`
- `verifier/result.json`
- `artifacts/`
- `logs/raw_error.json` when a failure happened

Use them in this order:

1. `samples.jsonl` for the normalized final result
2. `verifier/result.json` for benchmark-native evidence
3. `runtime_metadata.json` for execution context
4. `artifacts/` for benchmark-specific outputs
5. `logs/raw_error.json` for raw failure envelopes

### 6.4 What To Expect In `artifacts/`

Each benchmark kit exports different artifact shapes.

`terminal_bench` typically exports:

- `tool_trace.json`
- `workspace_diff.json`
- `stdout.log`
- `stderr.log`

`swebench` typically exports:

- `submission.patch`
- `agent_trace.json`
- `final_response.txt`
- `swebench_diagnostics.json`

`tau2` typically exports:

- `tau2_state.json`
- `tau2_trajectory.json`
- `tau2_cost.json`

`appworld` typically exports:

- `appworld_save.json`
- `appworld_outputs.json`
- `appworld_tool_trace.json`
- `appworld_logs.json`

### 6.5 Result Verification Checklist

For each run, verify:

- `summary.json` has the expected benchmark metrics
- `samples.jsonl` has non-empty sample records
- `predict_result[0].agent_trace` exists and is the expected type
- the sample `artifacts/` directory exists
- benchmark-specific artifact files are present
- failures are visible through `failure_reason`, `diagnostic_reason`, or `diagnostic_details`

## 7. Add A New Benchmark Kit With New Workflows

This section describes the phase 1-compatible path for onboarding a new benchmark kit.

### 7.1 Create The Kit Directory

Create:

```text
src/gage_eval/agent_eval_kits/<new_kit>/
```

Recommended files:

- `kit.py`
- `runtime.py`
- `resources.py`
- `units.py`
- `artifacts.py`
- `judge_bridge.py`
- `trace_mapping.py`
- `legacy_support_migration.py`
- `sub_workflows/framework_loop.py`
- `sub_workflows/installed_client.py`

Not every kit needs the same amount of code, but this is the intended phase 1 shape.

### 7.2 Define Runtime Ownership In `runtime.py`

Put benchmark lifecycle here:

- initialize benchmark-owned state
- bootstrap runtime-owned context
- expose state needed by sub-workflows
- save or finalize benchmark-owned state when needed

Do not put scheduler-specific payload shaping here.

### 7.3 Define Scheduler-Specific Projection In `sub_workflows/*`

`sub_workflows/framework_loop.py` should own:

- framework-loop prompt/input projection
- framework-loop artifact capture
- framework-loop result normalization

`sub_workflows/installed_client.py` should own:

- installed-client request payload shaping
- installed-client environment projection
- installed-client artifact capture
- installed-client result normalization

Do not put the full benchmark lifecycle here.

### 7.4 Put Shared Benchmark Helpers In `units.py` And `artifacts.py`

Use `units.py` for:

- reusable prompt builders
- shared input projection helpers
- small benchmark-specific normalization helpers

Use `artifacts.py` for:

- benchmark-specific file export
- artifact naming
- fallback collection logic
- diagnostic artifact generation

### 7.5 Register The Kit In `kit.py`

Your `load_kit()` should return one `BenchmarkKitEntry` that wires:

- the runtime entry,
- supported schedulers,
- workflow resolver,
- verifier resource resolver,
- trace mapper,
- compat shim resolver

The workflow resolver must map scheduler type to the correct workflow bundle.

### 7.6 Add Configs Under `config/custom/<new_kit>/`

At minimum, add one config for each scheduler surface you want to validate:

- `<new_kit>_framework_loop_*.yaml`
- `<new_kit>_installed_client_*.yaml`

Installed-client configs should describe the installed-client runtime path. They should not rely on a framework-loop backend path for the main execution loop.

### 7.7 Add Tests Before Adding The Kit To The 8-Workflow Script

Recommended minimum test coverage:

- unit tests for `runtime.py`
- unit tests for `sub_workflows/framework_loop.py`
- unit tests for `sub_workflows/installed_client.py`
- unit tests for `artifacts.py`
- one integration test that exercises one sample end to end

### 7.8 Add The New Workflow To The Shared Runner

Only after the kit is runnable on its own:

1. add the new config path to `scripts/run_phase1_8flows.sh` or a sibling matrix script
2. keep the same naming pattern for `run_id`
3. keep benchmark-specific logic in the kit, not in the script

## 8. Installed-Client Contract For New Kits

Installed-client mode expects the benchmark workflow to hand off two things:

- `request`
- `environment`

The detailed HTTP schema is documented in:

- `docs/installed_client_service_contract.md`

In practice:

- `request` should contain benchmark-owned instruction and stable client metadata
- `environment` should contain runtime-owned context the client may need to act against
- the installed-client service owns tool calling, memory, and the agent loop itself

If a new kit needs more installed-client fields, prefer extending the kit-local request projection first and only widen the shared service contract when the new field is broadly useful.

## 9. Recommended Validation Flow

When bringing up a new kit or debugging the 8-workflow matrix, use this order:

1. start the local Codex proxy and confirm `/healthz`
2. run one workflow directly with `python run.py --config ...`
3. inspect `summary.json`
4. inspect `samples.jsonl`
5. inspect `verifier/result.json`
6. inspect `artifacts/`
7. only then move to the shared 8-workflow runner

This keeps failures local and makes it easier to tell whether the problem is:

- benchmark lifecycle,
- scheduler projection,
- installed-client service wiring,
- verifier wiring,
- or artifact capture.

## 10. Quick Commands

Start the local Codex proxy:

```bash
cd /home/amyjx/work/gage-installed-client-stub
./run_local_codex_proxy.sh
```

Run the phase 1 matrix:

```bash
cd /home/amyjx/work/GAGE
RUN_PREFIX=phase1_$(date +%Y%m%d) ./scripts/run_phase1_8flows.sh
```

Run one workflow directly:

```bash
cd /home/amyjx/work/GAGE
.venv/bin/python run.py \
  --config config/custom/appworld/appworld_agent_demo_runtime_ollama.yaml \
  --run-id appworld_framework_smoke \
  --output-dir runs \
  --max-samples 1
```

Check the installed-client proxy:

```bash
curl -fsS http://127.0.0.1:8787/healthz
```

Check the final batch directory:

```bash
ls runs
```

Find the main verification files for one run:

```bash
find runs/<batch_dir>/<run_id> -maxdepth 3 \\( -name summary.json -o -name samples.jsonl -o -name result.json -o -path '*/artifacts/*' \\)
```
