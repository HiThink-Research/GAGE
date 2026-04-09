from __future__ import annotations

from typing import Any

__all__ = [
    "AppWorldVerifierAdapter",
    "BaseVerifierAdapter",
    "JudgeBinding",
    "JudgeVerifierAdapter",
    "NativeVerifierAdapter",
    "RuntimeJudgeOutcome",
    "SwebenchVerifierAdapter",
    "Tau2VerifierAdapter",
    "VerifierInput",
    "VerifierResult",
]


def __getattr__(name: str) -> Any:
    """Lazily expose verifier runtime symbols."""

    if name == "JudgeBinding":
        from gage_eval.agent_runtime.verifier.binding import JudgeBinding

        return JudgeBinding
    if name in {"RuntimeJudgeOutcome", "VerifierInput", "VerifierResult"}:
        from gage_eval.agent_runtime.verifier.contracts import (
            RuntimeJudgeOutcome,
            VerifierInput,
            VerifierResult,
        )

        return {
            "RuntimeJudgeOutcome": RuntimeJudgeOutcome,
            "VerifierInput": VerifierInput,
            "VerifierResult": VerifierResult,
        }[name]
    if name in {
        "AppWorldVerifierAdapter",
        "BaseVerifierAdapter",
        "JudgeVerifierAdapter",
        "NativeVerifierAdapter",
        "SwebenchVerifierAdapter",
        "Tau2VerifierAdapter",
    }:
        from gage_eval.agent_runtime.verifier.adapters import (
            AppWorldVerifierAdapter,
            BaseVerifierAdapter,
            JudgeVerifierAdapter,
            NativeVerifierAdapter,
            SwebenchVerifierAdapter,
            Tau2VerifierAdapter,
        )

        return {
            "AppWorldVerifierAdapter": AppWorldVerifierAdapter,
            "BaseVerifierAdapter": BaseVerifierAdapter,
            "JudgeVerifierAdapter": JudgeVerifierAdapter,
            "NativeVerifierAdapter": NativeVerifierAdapter,
            "SwebenchVerifierAdapter": SwebenchVerifierAdapter,
            "Tau2VerifierAdapter": Tau2VerifierAdapter,
        }[name]
    raise AttributeError(name)
