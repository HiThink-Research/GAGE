from __future__ import annotations

from typing import Any

from gage_eval.agent_eval_kits.tau2.units import build_tau2_prompt_context


class Tau2RuntimeEntry:
    """Owns Tau2 initialize_task bootstrap."""

    benchmark_kit_id = "tau2"
    runtime_version = "phase1"
    supported_schedulers = ("installed_client", "framework_loop")
    verifier_kind = "judge_adapter"
    resource_requirements = {"resource_kind": "local_process"}
    lifecycle_policy = {"initialize": "initialize_task", "save": "state_snapshot", "teardown": "provider_managed"}
    state_schema_keys = ("runtime_context", "prompt_context", "benchmark_state", "scheduler_state")

    def bootstrap(self, *, session, sample, payload, sandbox_provider=None):
        """Bootstrap Tau2 runtime state through initialize_task()."""

        if sandbox_provider is None:
            raise ValueError("tau2 runtime requires sandbox_provider")
        handle = sandbox_provider.get_handle()
        runtime = handle.sandbox if handle is not None else None
        initializer = getattr(runtime, "initialize_task", None)
        if not callable(initializer):
            raise RuntimeError("tau2 initialize_task is unavailable")
        _configure_user_simulator(runtime, session)
        initialize_result = initializer(sample)
        prompt_context = build_tau2_prompt_context(sample, initialize_result)
        return {
            "runtime_context": prompt_context,
            "prompt_context": prompt_context,
            "benchmark_state": {"initialize_result": initialize_result},
            "scheduler_state": {},
        }


def _configure_user_simulator(runtime: Any, session: Any) -> None:
    benchmark_config = getattr(session, "benchmark_config", None)
    if not isinstance(benchmark_config, dict):
        return
    user_simulator = benchmark_config.get("user_simulator")
    if user_simulator is None:
        return
    configure = getattr(runtime, "configure_user_simulator", None)
    if not callable(configure):
        return
    configure(user_simulator)
