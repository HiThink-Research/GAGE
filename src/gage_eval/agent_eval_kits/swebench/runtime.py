from __future__ import annotations

from gage_eval.agent_eval_kits.swebench.units import build_swebench_runtime_context


class SwebenchRuntime:
    """Owns the SWE-bench runtime bootstrap."""

    benchmark_kit_id = "swebench"
    runtime_version = "phase1"
    supported_schedulers = ("installed_client", "framework_loop")
    verifier_kind = "judge_adapter"
    resource_requirements = {"resource_kind": "docker"}
    lifecycle_policy = {"initialize": "noop", "save": "noop", "teardown": "provider_managed"}
    state_schema_keys = ("runtime_context", "prompt_context", "benchmark_state", "scheduler_state")
    compat_mode = "legacy_backend"

    def bootstrap(self, *, session, sample, payload, sandbox_provider=None):
        """Bootstrap SWE-bench runtime context."""

        runtime_context = build_swebench_runtime_context(sample)
        prompt_context = {
            "repo": runtime_context.get("repo"),
            "base_commit": runtime_context.get("base_commit"),
            "test_command": runtime_context.get("test_command"),
        }
        return {
            "runtime_context": runtime_context,
            "prompt_context": prompt_context,
            "benchmark_state": {},
            "scheduler_state": {},
        }
