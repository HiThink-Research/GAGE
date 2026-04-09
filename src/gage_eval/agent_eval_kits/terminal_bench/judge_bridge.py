from __future__ import annotations

from gage_eval.agent_runtime.verifier.adapters import NativeVerifierAdapter


def resolve_verifier_resources() -> dict[str, object]:
    """Resolve terminal benchmark verifier resources."""

    return {
        "adapter": NativeVerifierAdapter("terminal_bench.native_verifier"),
    }
