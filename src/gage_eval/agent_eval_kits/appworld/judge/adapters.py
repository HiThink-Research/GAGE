"""Runtime verifier adapter for AppWorld."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from gage_eval.agent_runtime.verifier.contracts import VerifierInput, VerifierResult

from .scoring import build_appworld_diagnostics, normalize_appworld_payload


@dataclass
class AppWorldVerifierAdapter:
    """Evaluate AppWorld from the kit-owned save/scheduler payloads."""

    judge_source: str = "appworld.verifier_adapter.run"

    def run(self, verifier_input: VerifierInput) -> VerifierResult:
        appworld_payload = normalize_appworld_payload(
            sample=verifier_input.sample,
            scheduler_result=verifier_input.scheduler_result,
            runtime_context=verifier_input.runtime_context,
        )
        diagnostic_reason, diagnostic_details = build_appworld_diagnostics(appworld_payload)
        failure_reason = appworld_payload.get("failure_reason")
        status = "failed" if failure_reason else "completed"
        payload: dict[str, Any] = {
            "status": status,
            "appworld": appworld_payload,
            "diagnostic_reason": diagnostic_reason,
            "diagnostic_details": diagnostic_details,
        }
        metric = _metric(appworld_payload, failure_reason=failure_reason)
        payload.update(metric)
        return VerifierResult(status=status, payload=payload)


def _metric(appworld_payload: Mapping[str, Any], *, failure_reason: Any) -> dict[str, Any]:
    tgc = _coerce_float(appworld_payload.get("tgc"))
    if tgc is not None:
        return {
            "score": tgc,
            "resolved": tgc >= (1.0 - 1e-6),
            "failure_reason": None if tgc >= (1.0 - 1e-6) else str(failure_reason or "task_incomplete"),
        }
    tests = appworld_payload.get("tests") if isinstance(appworld_payload.get("tests"), Mapping) else {}
    fails = tests.get("fails")
    passes = tests.get("passes")
    fail_count = len(fails) if isinstance(fails, list) else 0
    pass_count = len(passes) if isinstance(passes, list) else 0
    if fail_count or pass_count:
        resolved = fail_count == 0 and pass_count > 0
        return {
            "score": 1.0 if resolved else 0.0,
            "resolved": resolved,
            "failure_reason": None if resolved else str(failure_reason or "assertion_failed"),
        }
    return {
        "score": 0.0,
        "resolved": False,
        "failure_reason": str(failure_reason or "missing_appworld_success_signal"),
    }


def _coerce_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
