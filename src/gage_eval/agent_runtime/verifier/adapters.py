from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from gage_eval.agent_runtime.contracts.failure import FailureEnvelope
from gage_eval.agent_runtime.verifier.contracts import VerifierInput, VerifierResult


@dataclass
class BaseVerifierAdapter:
    """Base runtime-owned verifier adapter contract."""

    judge_source: str

    def run(self, verifier_input: VerifierInput) -> VerifierResult:
        raise NotImplementedError


class NativeVerifierAdapter(BaseVerifierAdapter):
    """Runs a simple local-native verifier based on sample expectations."""

    def run(self, verifier_input: VerifierInput) -> VerifierResult:
        sample = verifier_input.sample
        agent_output = verifier_input.scheduler_result.get("agent_output") or {}
        answer = str(agent_output.get("answer") or "").strip()
        expected = (
            sample.get("expected_answer")
            or sample.get("answer")
            or verifier_input.runtime_context.get("expected_answer")
        )
        resolved = True
        if expected not in (None, ""):
            resolved = str(expected).strip() == answer
        payload = {
            "status": "completed",
            "resolved": resolved,
            "score": 1.0 if resolved else 0.0,
            "summary": "native verifier completed",
            "answer": answer,
            "expected_answer": expected,
        }
        return VerifierResult(status="completed", payload=payload)


class JudgeVerifierAdapter(BaseVerifierAdapter):
    """Bridges runtime verifier input into a legacy judge implementation."""

    def __init__(self, judge_source: str, judge_impl: Any) -> None:
        super().__init__(judge_source=judge_source)
        self._judge_impl = judge_impl

    def run(self, verifier_input: VerifierInput) -> VerifierResult:
        payload = {
            "sample": verifier_input.sample,
            "model_output": verifier_input.scheduler_result.get("agent_output") or {},
            "runtime_handle": verifier_input.runtime_context.get("runtime_handle") or {},
            "params": verifier_input.verifier_resources,
        }
        sandbox_provider = verifier_input.runtime_context.get("sandbox_provider")
        if sandbox_provider is not None:
            payload["sandbox_provider"] = sandbox_provider
        result = self._judge_impl.invoke(payload)
        if isinstance(result, dict):
            return VerifierResult(status="completed", payload=result)
        return VerifierResult(status="completed", payload={"result": result})


class AppWorldVerifierAdapter(JudgeVerifierAdapter):
    """Runtime verifier wrapper for AppWorld."""

    def __init__(self) -> None:
        from gage_eval.role.judge.appworld_evaluate import AppWorldEvaluate

        super().__init__("appworld.verifier_adapter.run", AppWorldEvaluate())


class Tau2VerifierAdapter(JudgeVerifierAdapter):
    """Runtime verifier wrapper for Tau2."""

    def __init__(self) -> None:
        from gage_eval.role.judge.tau2_eval import Tau2Evaluate

        super().__init__("tau2.verifier_adapter.run", Tau2Evaluate())


class SwebenchVerifierAdapter(JudgeVerifierAdapter):
    """Runtime verifier wrapper for SWE-bench-style patch evaluation."""

    def __init__(self) -> None:
        from gage_eval.role.judge.swebench_docker import SwebenchDocker

        super().__init__("swebench.verifier_adapter.run", SwebenchDocker())


def build_failure_result(
    *,
    judge_source: str,
    failure: FailureEnvelope,
) -> VerifierResult:
    """Build a stable verifier result for already-normalized failures."""

    return VerifierResult(
        status="failed",
        payload={
            "status": "failed",
            "resolved": False,
            "score": 0.0,
            "summary": failure.summary,
            "failure_reason": failure.failure_code,
            "failure_domain": failure.failure_domain,
            "judge_source": judge_source,
            "failure": failure.to_dict(),
        },
    )
