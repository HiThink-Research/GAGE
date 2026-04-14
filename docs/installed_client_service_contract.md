# Installed-Client Service Contract

This document defines the minimal HTTP contract expected by GAGE installed-client workflows.

## Summary

Installed-client mode assumes:

1. GAGE owns benchmark bootstrap, verifier, artifacts, and run metadata.
2. The external client owns the full agent loop, including tool calling and memory.
3. GAGE calls one HTTP service instead of executing a benchmark-local CLI binary.

## Request

GAGE sends:

```json
{
  "request": {
    "instruction": "task instruction",
    "cwd": "/workspace/repo",
    "env": {},
    "metadata": {
      "submission_contract": "submission.patch"
    },
    "payload": {
      "...": "benchmark-specific fields"
    }
  },
  "environment": {
    "session_id": "session-123",
    "run_id": "phase1_run",
    "task_id": "appworld_agent_demo_installed_client",
    "sample_id": "appworld_demo_0001",
    "benchmark_kit_id": "appworld",
    "scheduler_type": "installed_client",
    "client_id": "codex",
    "artifact_layout": {
      "...": "runtime-owned local paths"
    },
    "runtime_context": {
      "...": "benchmark-owned runtime state"
    },
    "prompt_context": {
      "...": "prompt-level context"
    },
    "scheduler_state": {
      "...": "scheduler state"
    }
  }
}
```

Notes:

- `request` is the benchmark-to-client handoff.
- `environment` is JSON-safe runtime state only.
- Runtime-only handles such as `sandbox_provider` are not sent to the service.

## Response

The service should return either:

```json
{
  "result": {
    "exit_code": 0,
    "stdout": "final answer",
    "stderr": "",
    "answer": "final answer",
    "status": "completed",
    "patch_content": "diff --git ...",
    "agent_trace": [],
    "metadata": {},
    "trajectory_text": "..."
  }
}
```

or the same fields at the top level.

## Required Behavior

- `exit_code` must be stable and numeric.
- `status` should be one of `completed`, `failed`, or `aborted`.
- `agent_trace` should contain the client-owned tool/memory execution evidence when available.
- `patch_content` should be returned for patch-delivery benchmarks such as SWE-bench when available.
- The service may ignore `artifact_layout`; GAGE persists artifacts locally after the response.

## Authentication

If the service requires bearer auth, GAGE sends:

- `Authorization: Bearer <token>`

when one of the following is set:

- `GAGE_CODEX_CLIENT_TOKEN`
- `CODEX_CLIENT_TOKEN`
- `GAGE_INSTALLED_CLIENT_TOKEN`

## Run Prerequisites

Installed-client workflows need:

1. A reachable service URL in one of:
   - `GAGE_CODEX_CLIENT_URL`
   - `CODEX_CLIENT_URL`
   - `GAGE_INSTALLED_CLIENT_URL`
2. Benchmark dependencies such as Docker / local datasets / Tau2 data.
3. Optional bearer token env vars if the service is protected.

They do not require:

- a local `codex` binary in the host PATH
- a benchmark container image with `codex` preinstalled
