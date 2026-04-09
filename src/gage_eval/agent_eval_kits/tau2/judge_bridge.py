from __future__ import annotations

from gage_eval.agent_runtime.verifier.adapters import Tau2VerifierAdapter


def resolve_verifier_resources() -> dict[str, object]:
    """Resolve Tau2 verifier resources."""

    return {
        "adapter": Tau2VerifierAdapter(),
    }
