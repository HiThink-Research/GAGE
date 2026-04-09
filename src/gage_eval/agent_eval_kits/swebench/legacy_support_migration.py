from __future__ import annotations

from gage_eval.agent_eval_kits.common import build_noop_compat_shim


def resolve_compat_shim():
    """Return the SWE-bench compat shim."""

    return build_noop_compat_shim(
        shim_id="swebench.legacy_backend",
        legacy_source="swebench_context_provider",
        target_runtime_id="swebench_runtime",
        target_benchmark_kit_id="swebench",
    )
