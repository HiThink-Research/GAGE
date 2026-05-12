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
        resolved, failure_reason = _resolve_native_verdict(
            benchmark_kit_id=verifier_input.benchmark_kit_id,
            answer=answer,
            expected=expected,
        )
        payload = {
            "status": "completed",
            "resolved": resolved,
            "score": 1.0 if resolved else 0.0,
            "summary": "native verifier completed",
            "answer": answer,
            "expected_answer": expected,
            "failure_reason": failure_reason,
        }
        return VerifierResult(status="completed", payload=payload)


class JudgeVerifierAdapter(BaseVerifierAdapter):
    """Bridges runtime verifier input into a judge implementation."""

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
        environment_lease = verifier_input.runtime_context.get("environment_lease")
        if environment_lease is not None:
            payload["environment_lease"] = environment_lease
        result = self._judge_impl.invoke(payload)
        if isinstance(result, dict):
            return VerifierResult(status="completed", payload=result)
        return VerifierResult(status="completed", payload={"result": result})


class AppWorldVerifierAdapter(JudgeVerifierAdapter):
    """Runtime verifier wrapper for AppWorld."""

    def __init__(self) -> None:
        from gage_eval.agent_eval_kits.appworld.judge.adapters import (
            AppWorldVerifierAdapter as KitAppWorldVerifierAdapter,
        )

        BaseVerifierAdapter.__init__(self, judge_source="appworld.verifier_adapter.run")
        self._kit_adapter = KitAppWorldVerifierAdapter()

    def run(self, verifier_input: VerifierInput) -> VerifierResult:
        return self._kit_adapter.run(verifier_input)


class Tau2VerifierAdapter(JudgeVerifierAdapter):
    """Runtime verifier wrapper for Tau2."""

    def __init__(self) -> None:
        from gage_eval.agent_eval_kits.tau2.judge.adapters import (
            Tau2VerifierAdapter as KitTau2VerifierAdapter,
        )

        BaseVerifierAdapter.__init__(self, judge_source="tau2.verifier_adapter.run")
        self._kit_adapter = KitTau2VerifierAdapter()

    def run(self, verifier_input: VerifierInput) -> VerifierResult:
        return self._kit_adapter.run(verifier_input)


class SwebenchVerifierAdapter(JudgeVerifierAdapter):
    """Runtime verifier wrapper for SWE-bench-style patch evaluation."""

    def __init__(self) -> None:
        from gage_eval.agent_eval_kits.swebench.judge.adapters import (
            SwebenchVerifierAdapter as KitSwebenchVerifierAdapter,
        )

        BaseVerifierAdapter.__init__(self, judge_source="swebench.verifier_adapter.run")
        self._kit_adapter = KitSwebenchVerifierAdapter()

    def run(self, verifier_input: VerifierInput) -> VerifierResult:
        return self._kit_adapter.run(verifier_input)


def _resolve_native_verdict(
    *,
    benchmark_kit_id: str,
    answer: str,
    expected: Any,
) -> tuple[bool, str | None]:
    """Resolve a stable native-verifier verdict for local benchmark adapters."""

    expected_text = _normalize_expected_text(expected)
    if expected_text is not None:
        resolved = expected_text == answer
        return resolved, None if resolved else "answer_mismatch"

    return False, "missing_expected_answer"


def _normalize_expected_text(expected: Any) -> str | None:
    """Normalize the expected native-verifier answer when available."""

    if expected in (None, ""):
        return None
    return str(expected).strip()


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
