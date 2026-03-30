"""Verifier types and adapters for agent runtime runs."""

from __future__ import annotations

from gage_eval.agent_runtime.verifier.base import Verifier, VerifierInput, VerifierResult
from gage_eval.agent_runtime.verifier.judge_adapter import JudgeVerifierAdapter

__all__ = [
    "JudgeVerifierAdapter",
    "Verifier",
    "VerifierInput",
    "VerifierResult",
]

