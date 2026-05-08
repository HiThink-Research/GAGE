"""Tau2 verifier executor."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from .scoring import evaluate_tau2_sample


@dataclass(frozen=True)
class Tau2ExecutionRequest:
    sample_id: str
    sample: dict[str, Any]
    runtime_state: dict[str, Any]
    scheduler_result: dict[str, Any] = field(default_factory=dict)
    timeout_s: int | float | None = None
    trajectory_ref: str | None = None
    runtime_state_ref: str | None = None
    trace_ref: str | None = None
    tool_trace_summary: dict[str, Any] = field(default_factory=dict)


def execute_tau2_verifier(
    request: Tau2ExecutionRequest,
    *,
    evaluator: Callable[[Tau2ExecutionRequest], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Execute Tau2 verification and normalize executor failures."""

    try:
        result = (evaluator or _evaluate)(request)
    except TimeoutError:
        return {
            "status": "failed",
            "resolved": False,
            "score": 0.0,
            "failure_code": "verifier.executor.timeout",
            "failure_reason": "verifier_timeout",
            "summary": "Tau2 verifier timed out",
        }
    except Exception as exc:
        return {
            "status": "failed",
            "resolved": False,
            "score": 0.0,
            "failure_code": "verifier.executor.failed",
            "failure_reason": f"{type(exc).__name__}: {exc}",
            "summary": "Tau2 verifier crashed during evaluation",
        }

    payload = dict(result or {})
    payload.setdefault("status", "completed")
    if "tau2" in payload:
        reward = payload.get("tau2", {}).get("reward") if isinstance(payload.get("tau2"), dict) else None
        try:
            payload.setdefault("score", float(reward))
        except (TypeError, ValueError):
            payload.setdefault("score", 0.0)
        payload.setdefault("resolved", float(payload.get("score") or 0.0) >= 1.0)
    return payload


def _evaluate(request: Tau2ExecutionRequest) -> dict[str, Any]:
    return evaluate_tau2_sample(sample=request.sample, runtime_state=request.runtime_state)
