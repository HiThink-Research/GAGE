from __future__ import annotations

from gage_eval.agent_runtime.verifier.adapters import AppWorldVerifierAdapter


def resolve_verifier_resources() -> dict[str, object]:
    """Resolve AppWorld verifier resources."""

    return {
        "adapter": AppWorldVerifierAdapter(),
    }
