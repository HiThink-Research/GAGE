from __future__ import annotations

from gage_eval.agent_eval_kits.common import build_noop_compat_shim


def resolve_compat_shim():
    """Return the terminal benchmark compat shim."""

    return build_noop_compat_shim(
        shim_id="terminal_bench.noop_compat",
        legacy_source="terminal_bench_legacy",
        target_runtime_id="terminal_bench_runtime",
        target_benchmark_kit_id="terminal_bench",
    )
