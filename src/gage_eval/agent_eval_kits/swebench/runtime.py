from __future__ import annotations

from gage_eval.agent_eval_kits.swebench.tools import build_swebench_runtime_context


class SwebenchRuntime:
    """Owns the SWE-bench runtime bootstrap."""

    benchmark_kit_id = "swebench"
    supported_schedulers = ("installed_client", "framework_loop")

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
