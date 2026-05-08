from __future__ import annotations

import json
import os
import hashlib
import mimetypes
import re
from pathlib import Path
from typing import Any

from gage_eval.agent_runtime.contracts.failure import FailureEnvelope
from gage_eval.agent_runtime.contracts.scheduler import SchedulerResult
from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.agent_runtime.session import AgentRuntimeSession
from gage_eval.agent_runtime.trace_schema import ArtifactRef, SampleRecord, TraceEvent, TrialResult
from gage_eval.agent_runtime.verifier.contracts import RuntimeJudgeOutcome
from gage_eval.observability.trace import ObservabilityTrace


_SAMPLE_LEVEL_INFRA_ARTIFACTS = {"effective_config.json", "sample_record.json", "trial_aggregate.json"}
_TRIAL_OWNERS = {"agent", "infra", "verifier"}
_SECRET_KEYNAME_PATTERN = re.compile(
    r"(?i)(?:^|[_\-\s])("
    r"api[_-]?key|api[_-]?secret|api[_-]?token|"
    r"client[_-]?secret|access[_-]?token|bearer[_-]?token|"
    r"refresh[_-]?token|id[_-]?token|auth[_-]?token|jwt[_-]?token|"
    r"csrf[_-]?token|session[_-]?token|secret|token|password|"
    r"passphrase|credential|credentials|private[_-]?key|signing[_-]?key|"
    r"authorization|auth"
    r")(?:$|[_\-\s])"
)
_USAGE_KEYNAME_ALLOWLIST = {
    "agent_total_tokens",
    "cached_tokens",
    "completion_tokens",
    "completion_tokens_details",
    "input_tokens",
    "input_tokens_details",
    "output_tokens",
    "output_tokens_details",
    "prompt_tokens",
    "prompt_tokens_details",
    "reasoning_tokens",
    "total_tokens",
    "user_total_tokens",
}


class ArtifactWriteError(RuntimeError):
    """Stable failure wrapper for artifact persistence errors."""

    code = "persistence.artifact.write_failed"

    def __init__(self, *, path: Path, error: BaseException) -> None:
        self.path = str(path)
        self.failure = {
            "failure_domain": "persistence",
            "failure_stage": "persist_outputs",
            "failure_code": self.code,
            "component_kind": "artifact_sink",
            "component_id": "runtime_artifact_sink",
            "owner": "runtime",
            "retryable": False,
            "summary": f"Failed to write artifact: {path}",
            "first_bad_step": "write_artifact",
            "suspect_files": [str(path)],
            "details": {"error_type": error.__class__.__name__, "error": str(error)},
        }
        super().__init__(self.failure["summary"])


class RuntimeArtifactSink:
    """Writes runtime metadata, verifier results, and raw error artifacts."""

    def __init__(self, base_dir: str | None = None) -> None:
        self._base_dir = Path(base_dir or os.environ.get("GAGE_EVAL_SAVE_DIR", "./runs")).expanduser()

    def build_layout(self, *, run_id: str, task_id: str, sample_id: str) -> dict[str, str]:
        """Build the canonical sample-scoped artifact layout."""

        artifacts_root = self._base_dir / run_id / "artifacts" / task_id / sample_id
        sample_infra_dir = artifacts_root / "infra"
        trial_root = artifacts_root / "trials" / "trial_0001"
        sample_infra_dir.mkdir(parents=True, exist_ok=True)
        return {
            "artifacts_root": str(artifacts_root),
            "sample_infra_dir": str(sample_infra_dir),
            "effective_config": str(sample_infra_dir / "effective_config.json"),
            "sample_record": str(sample_infra_dir / "sample_record.json"),
            "trial_aggregate": str(sample_infra_dir / "trial_aggregate.json"),
            "raw_error": str(sample_infra_dir / "raw_error.json"),
            "sample_root": str(artifacts_root),
            "artifacts_dir": str(artifacts_root),
            "verifier_result": str(trial_root / "verifier" / "verifier_result.json"),
            "runtime_metadata": str(sample_infra_dir / "sample_record.json"),
            "legacy_runtime_metadata": str(sample_infra_dir / "runtime_metadata.json"),
        }

    def persist_runtime_metadata(
        self,
        *,
        session: AgentRuntimeSession,
        scheduler_result: SchedulerResult | None = None,
        failure: FailureEnvelope | None = None,
    ) -> str:
        """Write sample-scoped runtime metadata."""

        target = Path(session.artifact_layout.get("legacy_runtime_metadata") or session.artifact_layout["runtime_metadata"])
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
        payload = _raw_error_payload(error)
        target.write_text(
            json.dumps(to_json_compatible(payload), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return str(target)

    def write_artifact(
        self,
        *,
        run_id: str | None = None,
        task_id: str | None = None,
        sample_id: str | None = None,
        trial_id: str | None = None,
        owner: str,
        name: str,
        content: bytes | str | dict[str, Any] | list[Any],
        metadata: dict[str, Any] | None = None,
        mime_type: str | None = None,
        secret_values: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    ) -> ArtifactRef | dict[str, Any]:
        """Write either a v2 Agent runtime artifact or a legacy lease artifact."""

        secret_replacements = _normalize_secret_replacements(secret_values)
        if run_id is None and task_id is None and sample_id is None and trial_id is None:
            return self._write_lease_artifact(
                owner=owner,
                name=name,
                content=content,
                metadata=metadata or {},
                mime_type=mime_type,
                secret_replacements=secret_replacements,
            )

        if run_id is None or task_id is None or sample_id is None:
            raise ValueError("run_id, task_id, and sample_id are required for v2 artifacts")

        run_segment = _safe_segment(run_id)
        task_segment = _safe_segment(task_id)
        sample_segment = _safe_segment(sample_id)
        owner_segment = _safe_segment(owner)
        name_segment = _safe_segment(name)
        if owner_segment not in _TRIAL_OWNERS:
            raise ValueError(f"owner must be one of {sorted(_TRIAL_OWNERS)}")

        if trial_id is None:
            if name_segment not in _SAMPLE_LEVEL_INFRA_ARTIFACTS:
                raise ValueError("trial_id is required for non sample-level artifacts")
            if owner_segment != "infra":
                raise ValueError("sample-level artifacts must use owner='infra'")
            relative_path = Path("artifacts") / task_segment / sample_segment / "infra" / name_segment
        else:
            trial_segment = _safe_segment(trial_id)
            relative_path = (
                Path("artifacts") / task_segment / sample_segment / "trials" / trial_segment / owner_segment / name_segment
            )

        if name_segment == "sample_record.json" and trial_id is None:
            content = _validated_sample_record_payload(content, secret_replacements)

        target = self._base_dir / run_segment / relative_path
        data, detected_mime_type = _serialize_content(
            content,
            name=name_segment,
            mime_type=mime_type,
            secret_replacements=secret_replacements,
        )
        self._write_bytes(target, data)
        return ArtifactRef(
            owner=owner_segment,
            name=name_segment,
            path=relative_path.as_posix(),
            mime_type=detected_mime_type,
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        )

    def write_effective_config(
        self,
        *,
        run_id: str,
        task_id: str,
        sample_id: str,
        final_config: dict[str, Any],
        source_layers: list[dict[str, Any]],
        secret_values: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    ) -> ArtifactRef:
        """Write redacted effective_config.json with source layer override evidence."""

        secret_replacements = _normalize_secret_replacements(secret_values)
        payload = {
            "final_config": _redact_secrets(to_json_compatible(final_config), secret_replacements),
            "source_layers": _redact_secrets(to_json_compatible(source_layers), secret_replacements),
            "override_chain": _redact_secrets(_build_override_chain(source_layers), secret_replacements),
        }
        ref = self.write_artifact(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            owner="infra",
            name="effective_config.json",
            content=payload,
            mime_type="application/json",
            secret_values=secret_replacements,
        )
        if not isinstance(ref, ArtifactRef):  # pragma: no cover - guarded by run/task/sample inputs
            raise TypeError("write_effective_config expected a v2 ArtifactRef")
        return ref

    def append_trace_event(
        self,
        *,
        run_id: str,
        task_id: str,
        sample_id: str,
        trial_id: str,
        actor: str,
        event_type: str,
        payload: dict[str, Any],
        sequence_no: int | None = None,
        timestamp: str | None = None,
        artifact_refs: list[ArtifactRef | dict[str, Any]] | None = None,
        secret_values: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    ) -> ArtifactRef:
        """Validate and append a trace event to the trial trace.jsonl file."""

        from datetime import datetime, timezone

        secret_replacements = _normalize_secret_replacements(secret_values)
        relative_path = (
            Path("artifacts")
            / _safe_segment(task_id)
            / _safe_segment(sample_id)
            / "trials"
            / _safe_segment(trial_id)
            / "infra"
            / "trace.jsonl"
        )
        target = self._base_dir / _safe_segment(run_id) / relative_path
        normalized_payload = _redact_secrets(to_json_compatible(payload), secret_replacements)
        event_payload = {
            "run_id": run_id,
            "task_id": task_id,
            "sample_id": sample_id,
            "trial_id": trial_id,
            "sequence_no": sequence_no,
            "timestamp": timestamp or datetime.now(timezone.utc).isoformat(),
            "actor": actor,
            "event_type": event_type,
            "payload": normalized_payload,
            "artifact_refs": [
                ref.model_dump(mode="python") if isinstance(ref, ArtifactRef) else dict(ref)
                for ref in (artifact_refs or [])
            ],
        }
        try:
            if event_payload["sequence_no"] is None:
                event_payload["sequence_no"] = self._next_trace_sequence(target)
            event = TraceEvent.model_validate(event_payload, context={"secret_values": secret_replacements})
            data = (
                json.dumps(event.model_dump(mode="python"), ensure_ascii=False, separators=(",", ":")).encode("utf-8")
                + b"\n"
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("ab") as handle:
                handle.write(data)
            size_bytes = target.stat().st_size
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
        except OSError as exc:
            raise ArtifactWriteError(path=target, error=exc) from exc
        return ArtifactRef(
            owner="infra",
            name="trace.jsonl",
            path=relative_path.as_posix(),
            mime_type="application/jsonl",
            size_bytes=size_bytes,
            sha256=digest,
        )

    def write_trial_record(
        self,
        *,
        run_id: str,
        task_id: str,
        sample_id: str,
        trial_id: str,
        trial_result: TrialResult | dict[str, Any],
        secret_values: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    ) -> ArtifactRef:
        """Write a validated per-trial TrialResult record."""

        result = (
            trial_result
            if isinstance(trial_result, TrialResult)
            else TrialResult.model_validate(to_json_compatible(trial_result))
        )
        ref = self.write_artifact(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            trial_id=trial_id,
            owner="infra",
            name="trial_result.json",
            content=result.model_dump(mode="python"),
            mime_type="application/json",
            secret_values=secret_values,
        )
        if not isinstance(ref, ArtifactRef):  # pragma: no cover - guarded by v2 inputs
            raise TypeError("write_trial_record expected a v2 ArtifactRef")
        return ref

    def write_trial_aggregate(
        self,
        *,
        run_id: str,
        task_id: str,
        sample_id: str,
        aggregate: Any,
        secret_values: dict[str, str] | list[str] | tuple[str, ...] | None = None,
    ) -> ArtifactRef:
        """Write the sample-level TrialAggregate payload."""

        content = aggregate.to_dict() if hasattr(aggregate, "to_dict") else to_json_compatible(aggregate)
        ref = self.write_artifact(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            owner="infra",
            name="trial_aggregate.json",
            content=content,
            mime_type="application/json",
            secret_values=secret_values,
        )
        if not isinstance(ref, ArtifactRef):  # pragma: no cover - guarded by v2 inputs
            raise TypeError("write_trial_aggregate expected a v2 ArtifactRef")
        return ref

    def _write_lease_artifact(
        self,
        *,
        owner: str,
        name: str,
        content: bytes | str | dict[str, Any] | list[Any],
        metadata: dict[str, Any],
        mime_type: str | None,
        secret_replacements: dict[str, str],
    ) -> dict[str, Any]:
        owner_segment = _safe_segment(owner)
        name_segment = _safe_segment(name)
        relative_path = Path("artifacts") / owner_segment / name_segment
        data, detected_mime_type = _serialize_content(
            content,
            name=name_segment,
            mime_type=mime_type,
            secret_replacements=secret_replacements,
        )
        target = self._base_dir / relative_path
        self._write_bytes(target, data)
        return {
            "owner": owner_segment,
            "name": name_segment,
            "path": relative_path.as_posix(),
            "mime_type": detected_mime_type,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
            "metadata": dict(metadata),
        }

    def _write_bytes(self, target: Path, data: bytes) -> None:
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        except OSError as exc:
            raise ArtifactWriteError(path=target, error=exc) from exc

    def _next_trace_sequence(self, target: Path) -> int:
        if not target.exists():
            return 1
        return sum(1 for _ in target.open("rb")) + 1


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


def _safe_segment(value: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("unsafe empty path segment")
    if "/" in value or "\\" in value:
        raise ValueError(f"unsafe path segment: {value!r}")
    if value in {".", ".."} or ".." in value or Path(value).is_absolute():
        raise ValueError(f"unsafe path segment: {value!r}")
    return value


def _serialize_content(
    content: bytes | str | dict[str, Any] | list[Any],
    *,
    name: str,
    mime_type: str | None,
    secret_replacements: dict[str, str],
) -> tuple[bytes, str]:
    detected_mime_type = mime_type or _guess_mime_type(name, content)
    if isinstance(content, bytes):
        data = content
        if secret_replacements:
            data = _redact_text(content.decode("utf-8", errors="ignore"), secret_replacements).encode("utf-8")
        return data, detected_mime_type
    if isinstance(content, str):
        return _redact_text(content, secret_replacements).encode("utf-8"), detected_mime_type
    payload = _redact_secrets(to_json_compatible(content), secret_replacements)
    text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    return text.encode("utf-8"), detected_mime_type


def _raw_error_payload(error: BaseException) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "error_type": error.__class__.__name__,
        "error": str(error),
    }
    failure = getattr(error, "failure", None)
    if hasattr(failure, "to_dict"):
        payload["failure"] = failure.to_dict()
    manager_payload = _call_to_failure_payload(error)
    if manager_payload:
        payload["manager_failure"] = manager_payload
    validation_errors = _validation_errors_from_chain(error)
    if validation_errors:
        payload["validation_errors"] = validation_errors
    cause = getattr(error, "__cause__", None)
    if cause is not None and cause is not error:
        payload["cause"] = _raw_error_payload(cause)
    return payload


def _call_to_failure_payload(error: BaseException) -> dict[str, Any] | None:
    to_failure_payload = getattr(error, "to_failure_payload", None)
    if not callable(to_failure_payload):
        return None
    try:
        payload = to_failure_payload()
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _validation_errors_from_chain(error: BaseException) -> list[Any] | None:
    current: BaseException | None = error
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        errors = getattr(current, "errors", None)
        if callable(errors):
            try:
                resolved = errors()
            except Exception:
                resolved = None
            if resolved:
                return resolved
        current = getattr(current, "__cause__", None)
    return None


def _guess_mime_type(name: str, content: bytes | str | dict[str, Any] | list[Any]) -> str:
    if isinstance(content, dict | list):
        return "application/json"
    if name.endswith(".jsonl"):
        return "application/jsonl"
    guessed, _ = mimetypes.guess_type(name)
    return guessed or ("application/octet-stream" if isinstance(content, bytes) else "text/plain")


def _redact_secrets(value: Any, secret_replacements: dict[str, str], *, key_name: str | None = None) -> Any:
    if key_name is not None and _is_secret_keyname(key_name):
        redacted = _redact_sensitive_key_value(
            value,
            key_name=key_name,
            secret_replacements=secret_replacements,
        )
        if redacted is not _NO_KEYNAME_REDACTION:
            return redacted
    if isinstance(value, str):
        return _redact_text(value, secret_replacements)
    if isinstance(value, dict):
        return {key: _redact_secrets(child, secret_replacements, key_name=str(key)) for key, child in value.items()}
    if isinstance(value, list):
        return [_redact_secrets(child, secret_replacements) for child in value]
    return value


_NO_KEYNAME_REDACTION = object()


def _is_secret_keyname(key_name: str) -> bool:
    if str(key_name or "").lower() in _USAGE_KEYNAME_ALLOWLIST:
        return False
    return bool(_SECRET_KEYNAME_PATTERN.search(str(key_name or "")))


def _redact_sensitive_key_value(
    value: Any,
    *,
    key_name: str,
    secret_replacements: dict[str, str],
) -> Any:
    if value in (None, ""):
        return value
    if isinstance(value, str):
        redacted_text = _redact_text(value, secret_replacements)
        if redacted_text != value or redacted_text.startswith("<redacted") or value.startswith("${"):
            return redacted_text
        if not value.strip():
            return value
    return f"<redacted:keyname:{key_name}>"


def _redact_text(value: str, secret_replacements: dict[str, str]) -> str:
    redacted = value
    for secret, replacement in sorted(secret_replacements.items(), key=lambda item: len(item[0]), reverse=True):
        if secret:
            redacted = redacted.replace(secret, replacement)
    return redacted


def _normalize_secret_replacements(
    secret_values: dict[str, str] | list[str] | tuple[str, ...] | None,
) -> dict[str, str]:
    if not secret_values:
        return {}
    if isinstance(secret_values, dict):
        return {str(secret): str(replacement) for secret, replacement in secret_values.items() if secret}
    return {str(secret): f"<redacted:reference:secret-{index}>" for index, secret in enumerate(secret_values, start=1) if secret}


def _validated_sample_record_payload(content: Any, secret_replacements: dict[str, str]) -> dict[str, Any]:
    if hasattr(content, "model_dump"):
        payload = content.model_dump(mode="python")
    elif isinstance(content, dict):
        payload = to_json_compatible(content)
    else:
        raise ValueError("sample_record.json content must be a mapping or SampleRecord")
    redacted = _redact_secrets(payload, secret_replacements)
    record = SampleRecord.model_validate(redacted, context={"secret_values": secret_replacements})
    return record.model_dump(mode="python")


def _build_override_chain(source_layers: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    chains: dict[str, list[dict[str, Any]]] = {}
    for layer in source_layers:
        layer_name = str(layer.get("name", "unknown"))
        values = layer.get("values", {})
        if not isinstance(values, dict):
            continue
        for path, value in _flatten_mapping(values).items():
            chains.setdefault(path, []).append({"layer": layer_name, "value": value})
    return chains


def _flatten_mapping(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            flattened.update(_flatten_mapping(child, path))
        else:
            flattened[path] = child
    return flattened
