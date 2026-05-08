from __future__ import annotations

from typing import Any

from gage_eval.agent_eval_kits.terminal_bench.artifacts import capture_terminal_workspace_state
from gage_eval.agent_eval_kits.terminal_bench.units import build_terminal_runtime_context


class TerminalBenchRuntime:
    """Owns the terminal benchmark lifecycle and state bootstrap."""

    benchmark_kit_id = "terminal_bench"
    supported_schedulers = ("installed_client", "framework_loop")

    def bootstrap(self, *, session, sample: dict[str, Any], payload: dict[str, Any], sandbox_provider=None) -> dict[str, Any]:
        """Bootstrap terminal benchmark state for one sample."""

        runtime_context = build_terminal_runtime_context(sample)
        workspace_state = capture_terminal_workspace_state(
            sandbox_provider,
            cwd=str(runtime_context.get("cwd") or "/workspace"),
        )
        return {
            "runtime_context": runtime_context,
            "prompt_context": runtime_context,
            "benchmark_state": {
                "cwd": runtime_context["cwd"],
                "workspace_state": workspace_state,
            },
            "scheduler_state": {},
        }
