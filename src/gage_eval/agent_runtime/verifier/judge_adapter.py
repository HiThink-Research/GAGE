"""Judge verifier adapter for SWE-bench style benchmarks."""

from __future__ import annotations

from pathlib import Path
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
        mode = str(self._implementation_params.get("mode") or "").strip().lower()
        if mode == "patch_presence":
            return _PatchPresenceJudge(**self._implementation_params)
        from gage_eval.role.judge.swebench_docker import SwebenchDocker

        return SwebenchDocker(**self._implementation_params)


class _PatchPresenceJudge:
    """Smoke-path judge that validates the produced patch content."""

    def __init__(self, **params: Any) -> None:
        self._params = dict(params)

    def invoke(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        metadata = _coerce_mapping(payload.get("metadata"))
        sample = _coerce_mapping(payload.get("sample"))
        sample_metadata = _coerce_mapping(sample.get("metadata"))
        expected_terms = _coerce_terms(
            self._params.get("expected_patch_contains")
            or metadata.get("expected_patch_contains")
            or sample_metadata.get("expected_patch_contains")
        )
        patch_text = _resolve_patch_text(payload)
        if not patch_text.strip():
            return {
                "resolved": False,
                "failure_reason": "missing_patch_content",
                "matched_terms": [],
            }
        missing_terms = [term for term in expected_terms if term not in patch_text]
        if missing_terms:
            return {
                "resolved": False,
                "failure_reason": "patch_missing_expected_terms",
                "matched_terms": [term for term in expected_terms if term not in missing_terms],
                "missing_terms": missing_terms,
            }
        if "diff --git" not in patch_text:
            return {
                "resolved": False,
                "failure_reason": "patch_missing_diff_header",
                "matched_terms": list(expected_terms),
            }
        return {
            "resolved": True,
            "failure_reason": None,
            "matched_terms": list(expected_terms),
        }


def _resolve_patch_text(payload: Dict[str, Any]) -> str:
    for candidate in (
        payload.get("patch_content"),
        _coerce_mapping(payload.get("scheduler_result")).get("patch_content"),
        _coerce_mapping(_coerce_mapping(payload.get("scheduler_result")).get("raw_output")).get("patch_content"),
        payload.get("model_output"),
    ):
        if isinstance(candidate, str) and candidate.strip():
            return candidate
    artifact_paths = _coerce_mapping(payload.get("artifact_paths"))
    for key in ("patch_path", "patch_file"):
        path = artifact_paths.get(key) or payload.get(key)
        if isinstance(path, str) and path.strip():
            try:
                return Path(path).read_text(encoding="utf-8")
            except OSError:
                continue
    return ""


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _coerce_terms(value: Any) -> tuple[str, ...]:
    if isinstance(value, str) and value.strip():
        return (value.strip(),)
    if isinstance(value, (list, tuple)):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return ()
