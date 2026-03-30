"""Judge verifier adapter for SWE-bench style benchmarks."""

from __future__ import annotations

from typing import Any, Dict, Optional

from gage_eval.agent_runtime.verifier.base import Verifier, VerifierInput, VerifierResult


class JudgeVerifierAdapter:
    """Adapt a judge implementation to the verifier protocol."""

    def __init__(self, judge: Optional[Any] = None, **implementation_params: Any) -> None:
        self._judge = judge
        self._implementation_params = dict(implementation_params)

    def verify(self, verifier_input: VerifierInput) -> VerifierResult:
        """Run the wrapped judge and normalize its output."""
        judge = self._judge or self._build_judge()
        payload = dict(verifier_input.payload or {})
        payload.setdefault("sample", {"id": verifier_input.sample_id, "metadata": dict(verifier_input.metadata)})
        payload.setdefault("params", {})
        if verifier_input.artifact_paths:
            payload.setdefault("artifact_paths", dict(verifier_input.artifact_paths))
        if verifier_input.metadata:
            payload.setdefault("metadata", dict(verifier_input.metadata))
        try:
            result = judge.invoke(payload)
        except Exception as exc:  # pragma: no cover - defensive
            return VerifierResult(status="error", summary=str(exc), raw_output={"error": str(exc)})
        if not isinstance(result, dict):
            return VerifierResult(status="error", summary="judge_returned_non_mapping", raw_output={"value": result})
        resolved = bool(result.get("resolved"))
        status = "pass" if resolved else "fail"
        score = 1.0 if resolved else 0.0
        summary = result.get("failure_reason") or ("resolved" if resolved else "unresolved")
        return VerifierResult(status=status, score=score, summary=summary, raw_output=result)

    def _build_judge(self) -> Any:
        """Instantiate the default SWE-bench judge lazily."""
        from gage_eval.role.judge.swebench_docker import SwebenchDocker

        return SwebenchDocker(**self._implementation_params)

