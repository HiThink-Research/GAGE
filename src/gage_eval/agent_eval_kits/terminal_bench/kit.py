"""Terminal benchmark kit definition."""

from __future__ import annotations

from gage_eval.agent_eval_kits.terminal_bench.contracts import (
    BenchmarkKitDefinition,
    TERMINAL_BENCH_KIT_ID,
    TERMINAL_BENCH_REQUIRED_SURFACES,
)


def build_kit() -> BenchmarkKitDefinition:
    """Return the terminal benchmark kit definition."""

    return BenchmarkKitDefinition(
        kit_id=TERMINAL_BENCH_KIT_ID,
        verifier_kind="native",
        required_surfaces=TERMINAL_BENCH_REQUIRED_SURFACES,
    )
