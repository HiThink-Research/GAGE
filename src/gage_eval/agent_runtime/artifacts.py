from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from gage_eval.agent_runtime.contracts.failure import FailureEnvelope
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.verifier.contracts import RuntimeJudgeOutcome
from gage_eval.observability.trace import ObservabilityTrace


class RuntimeArtifactSink:
    """Writes runtime metadata, verifier results, and raw error artifacts."""

    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = Path(base_dir or os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs")).expanduser()

    def build_layout(self, *, run_id: str, task_id: str, sample_id: str) -> dict[str, str]:
        """Build the canonical sample-scoped artifact layout."""

        sample_root = self._base_dir / run_id / "samples" / "runtime" / task_id / sample_id
        artifacts_dir = sample_root / "artifacts"
        verifier_dir = sample_root / "verifier"
        logs_dir = sample_root / "logs"
        sample_root.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        verifier_dir.mkdir(parents=True, exist_ok=True)
        logs_dir.mkdir(parents=True, exist_ok=True)
        return {
            "sample_root": str(sample_root),
            "artifacts_dir": str(artifacts_dir),
            "verifier_result": str(verifier_dir / "result.json"),
            "runtime_metadata": str(sample_root / "runtime_metadata.json"),
            "raw_error": str(logs_dir / "raw_error.json"),
        }

    def persist_runtime_metadata(
        self,
        *,
        session: AgentRuntimeSession,
        scheduler_result: SchedulerResult | None = None,
        failure: FailureEnvelope | None = None,
    ) -> str:
        """Write sample-scoped runtime metadata."""

        target = Path(session.artifact_layout["runtime_metadata"])
        payload = {
            "session_id": session.session_id,
            "run_id": session.run_id,
            "task_id": session.task_id,
            "sample_id": session.sample_id,
            "benchmark_kit_id": session.benchmark_kit_id,
            "scheduler_type": session.scheduler_type,
            "client_id": session.client_id,
            "resource_lease": session.resource_lease.to_dict() if session.resource_lease is not None else None,
            "runtime_context": dict(session.runtime_context or {}),
            "prompt_context": dict(session.prompt_context or {}),
            "benchmark_state": dict(session.benchmark_state or {}),
            "scheduler_state": dict(session.scheduler_state or {}),
            "scheduler_result": scheduler_result.to_dict() if scheduler_result is not None else None,
            "failure": failure.to_dict() if failure is not None else None,
        }
        target.write_text(
            json.dumps(to_json_compatible(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(target)

    def persist_verifier_result(self, outcome: RuntimeJudgeOutcome) -> str:
        """Write the normalized verifier output."""

        target = Path(outcome.persisted_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(to_json_compatible(outcome.to_dict()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(target)

    def persist_raw_error(self, *, session: AgentRuntimeSession, error: BaseException) -> str:
        """Write the raw error payload."""

        target = Path(session.artifact_layout["raw_error"])
        payload = {
            "error_type": error.__class__.__name__,
            "error": str(error),
        }
        target.write_text(
            json.dumps(to_json_compatible(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(target)


class RuntimeTraceEmitter:
    """Projects runtime execution lifecycle into the existing trace bus."""

    def emit_session_start(self, trace: ObservabilityTrace | None, session: AgentRuntimeSession) -> None:
        if trace is None:
            return
        trace.emit(
            "runtime_session_start",
            {
                "session_id": session.session_id,
                "task_id": session.task_id,
                "scheduler_type": session.scheduler_type,
                "benchmark_kit_id": session.benchmark_kit_id,
            },
            sample_id=session.sample_id,
        )

    def emit_session_end(
        self,
        trace: ObservabilityTrace | None,
        session: AgentRuntimeSession,
        *,
        scheduler_result: SchedulerResult | None = None,
    ) -> None:
        if trace is None:
            return
        trace.emit(
            "runtime_session_end",
            {
                "session_id": session.session_id,
                "scheduler_type": session.scheduler_type,
                "status": scheduler_result.status if scheduler_result is not None else "completed",
            },
            sample_id=session.sample_id,
        )

    def emit_failure(
        self,
        trace: ObservabilityTrace | None,
        session: AgentRuntimeSession,
        *,
        failure: FailureEnvelope,
    ) -> None:
        if trace is None:
            return
        payload = failure.to_dict()
        payload.update(
            {
                "session_id": session.session_id,
                "run_id": session.run_id,
                "task_id": session.task_id,
                "sample_id": session.sample_id,
                "scheduler_type": session.scheduler_type,
                "benchmark_kit_id": session.benchmark_kit_id,
            }
        )
        trace.emit("runtime_failure", payload, sample_id=session.sample_id)
