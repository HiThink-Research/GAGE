"""Kit-owned Tau2 verifier adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from gage_eval.agent_runtime.verifier.contracts import VerifierInput, VerifierResult

from .bridges import build_tau2_verifier_request
from .executor import Tau2ExecutionRequest, execute_tau2_verifier
from .scoring import tau2_metric


@dataclass
class Tau2VerifierAdapter:
    """Runtime verifier adapter for Tau2."""

    timeout_s: int = 300
    judge_source: str = "tau2.verifier_adapter.run"

    def run(self, verifier_input: VerifierInput) -> VerifierResult:
        request_payload = build_tau2_verifier_request(
            sample_id=verifier_input.sample_id,
            sample=verifier_input.sample,
            scheduler_result=verifier_input.scheduler_result,
            runtime_context=verifier_input.runtime_context,
        )
        timeout_s = verifier_input.verifier_resources.get("timeout_s", self.timeout_s)
        result = execute_tau2_verifier(
            Tau2ExecutionRequest(
                sample_id=str(request_payload["sample_id"]),
                sample=dict(request_payload["sample"]),
                runtime_state=_merge_runtime_metadata(
                    request_payload.get("runtime_state"),
                    verifier_input.sample,
                ),
                scheduler_result=dict(request_payload["scheduler_result"]),
                timeout_s=timeout_s,
                trajectory_ref=request_payload.get("trajectory_ref"),
                runtime_state_ref=request_payload.get("runtime_state_ref"),
                trace_ref=request_payload.get("trace_ref"),
                tool_trace_summary=dict(request_payload.get("tool_trace_summary") or {}),
            )
        )
        result.setdefault("metric", tau2_metric(result))
        result.setdefault("artifact_refs", _artifact_refs(result))
        _emit_verifier_result(verifier_input.runtime_context.get("trace"), verifier_input, result)
        return VerifierResult(
            status="completed" if result.get("status") == "completed" else "failed",
            payload=result,
        )


def _merge_runtime_metadata(runtime_state: Any, sample: Mapping[str, Any]) -> dict[str, Any]:
    state = dict(runtime_state) if isinstance(runtime_state, Mapping) else {}
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), Mapping) else {}
    tau2_meta = metadata.get("tau2") if isinstance(metadata.get("tau2"), Mapping) else {}
    for key in ("trial", "seed"):
        if key not in state and key in tau2_meta:
            state[key] = tau2_meta[key]
    return state


def _artifact_refs(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    refs = result.get("artifact_refs")
    if isinstance(refs, list) and refs:
        return [dict(ref) for ref in refs if isinstance(ref, Mapping)]
    return [{"owner": "verifier", "name": "verifier_result.json", "path": "verifier/result.json"}]


def _emit_verifier_result(trace: Any, verifier_input: VerifierInput, result: Mapping[str, Any]) -> None:
    emit = getattr(trace, "emit", None)
    if not callable(emit):
        return
    emit(
        "verifier.result",
        {
            "metric": dict(result.get("metric") or tau2_metric(result)),
            "verifier_result": dict(result),
            "artifact_refs": _artifact_refs(result),
        },
        sample_id=verifier_input.sample_id,
    )
