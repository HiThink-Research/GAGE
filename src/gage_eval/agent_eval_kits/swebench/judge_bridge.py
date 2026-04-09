from __future__ import annotations

from gage_eval.agent_runtime.verifier.adapters import SwebenchVerifierAdapter


def resolve_verifier_resources() -> dict[str, object]:
    """Resolve SWE-bench verifier resources."""

    return {
        "adapter": SwebenchVerifierAdapter(),
    }
