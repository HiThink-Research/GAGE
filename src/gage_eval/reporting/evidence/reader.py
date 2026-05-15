from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from gage_eval.reporting.contracts import EvidenceRef
from gage_eval.reporting.evidence.diagnostics import EvidenceDiagnostics
from gage_eval.reporting.privacy.secret_filter import SecretFilter


@dataclass
class RunEvidenceIndex:
    """Read-only index over existing run artifacts used by reporting."""

    run_dir: str | Path
    summary: dict[str, Any] = field(default_factory=dict)
    samples: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: dict[str, EvidenceRef] = field(default_factory=dict)
    diagnostics: EvidenceDiagnostics = field(default_factory=EvidenceDiagnostics)


class ReportEvidenceReader:
    """Builds a report evidence index from existing run output files."""

    def __init__(self, *, preview_bytes: int = 16_384, secret_filter: SecretFilter | None = None) -> None:
        self._preview_bytes = preview_bytes
        self._secret_filter = secret_filter or SecretFilter()

    def build_index(self, run_dir: str | Path) -> RunEvidenceIndex:
        root = Path(run_dir)
        diagnostics = EvidenceDiagnostics()
        summary = self._read_json(root / "summary.json", diagnostics, required=False)
        samples = self._read_samples(root / "samples.jsonl", diagnostics)
        refs = self._build_evidence_refs(root, samples, diagnostics)
        return RunEvidenceIndex(run_dir=root, summary=summary, samples=samples, evidence_refs=refs, diagnostics=diagnostics)

    def _read_json(self, path: Path, diagnostics: EvidenceDiagnostics, *, required: bool) -> dict[str, Any]:
        if not path.exists():
            if required:
                diagnostics.warn("report_pack.file_missing", path=path.name)
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            diagnostics.error("report_pack.json_unreadable", path=path.name, message=str(exc))
            return {}
        if isinstance(data, dict):
            return data
        diagnostics.warn("report_pack.json_not_object", path=path.name)
        return {}

    def _read_samples(self, path: Path, diagnostics: EvidenceDiagnostics) -> list[dict[str, Any]]:
        if not path.exists():
            diagnostics.warn("report_pack.file_missing", path=path.name)
            return []
        samples: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                diagnostics.error(
                    "report_pack.samples_jsonl_unreadable",
                    path=path.name,
                    line=line_number,
                    message=str(exc),
                )
                continue
            if isinstance(record, dict):
                samples.append(record)
            else:
                diagnostics.warn("report_pack.sample_not_object", path=path.name, line=line_number)
        return samples

    def _build_evidence_refs(
        self,
        root: Path,
        samples: Iterable[dict[str, Any]],
        diagnostics: EvidenceDiagnostics,
    ) -> dict[str, EvidenceRef]:
        refs: dict[str, EvidenceRef] = {}
        for sample in samples:
            task_id = _coerce_str(sample.get("task_id") or _nested(sample, "sample", "task_id"))
            sample_id = _coerce_str(sample.get("sample_id") or _nested(sample, "sample", "id"))
            for raw_ref in _iter_artifact_refs(sample):
                ref = self._artifact_ref(root, raw_ref, task_id=task_id, sample_id=sample_id, diagnostics=diagnostics)
                if ref is not None and ref.ref_id:
                    refs[ref.ref_id] = ref
        return dict(sorted(refs.items(), key=lambda item: item[0]))

    def _artifact_ref(
        self,
        root: Path,
        raw_ref: dict[str, Any],
        *,
        task_id: str | None,
        sample_id: str | None,
        diagnostics: EvidenceDiagnostics,
    ) -> EvidenceRef | None:
        path_value = _coerce_str(raw_ref.get("path"))
        if not path_value:
            diagnostics.warn("report_pack.artifact_ref_missing_path", sample_id=sample_id)
            return None
        if _is_absolute_or_escaping(path_value):
            diagnostics.warn("report_pack.artifact_ref_path_not_relative", path=path_value, sample_id=sample_id)
            return None

        artifact_path = root / path_value
        preview = self._preview_artifact(artifact_path, diagnostics, ref_path=path_value)
        mime_type = _coerce_str(raw_ref.get("mime_type")) or mimetypes.guess_type(path_value)[0] or "application/octet-stream"
        size_bytes = _coerce_int(raw_ref.get("size_bytes"))
        sha256 = _coerce_str(raw_ref.get("sha256"))
        timestamp_iso = _coerce_str(raw_ref.get("timestamp_iso"))
        if artifact_path.exists():
            stat = artifact_path.stat()
            size_bytes = size_bytes if size_bytes is not None else stat.st_size
            sha256 = sha256 or _sha256(artifact_path)
            timestamp_iso = timestamp_iso or _timestamp_iso(stat.st_mtime)
        else:
            diagnostics.warn("report_pack.artifact_missing", path=path_value, sample_id=sample_id)
            return None

        ref_id = _ref_id("artifact", path_value)
        return EvidenceRef(
            ref_id=ref_id,
            kind="artifact",
            path=path_value,
            artifact_role=_coerce_str(raw_ref.get("owner") or raw_ref.get("artifact_role")),
            sample_id=sample_id,
            task_id=task_id,
            trial_id=_trial_id_from_path(path_value),
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            timestamp_iso=timestamp_iso,
            preview=preview,
        )

    def _preview_artifact(
        self,
        path: Path,
        diagnostics: EvidenceDiagnostics,
        *,
        ref_path: str,
    ) -> dict[str, Any] | None:
        if not path.exists() or not path.is_file():
            return None
        try:
            raw = path.read_bytes()[: self._preview_bytes]
        except Exception as exc:
            diagnostics.warn("report_pack.artifact_preview_unreadable", path=ref_path, message=str(exc))
            return None
        text = raw.decode("utf-8", errors="replace")
        value: Any
        try:
            value = json.loads(text)
        except Exception:
            value = text
        redacted = self._secret_filter.redact(value)
        if redacted.redacted:
            diagnostics.warn("report_pack.secret_leak_detected", path=ref_path)
        return {"value": redacted.value, "truncated": path.stat().st_size > len(raw)}


def _iter_artifact_refs(sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
    candidates = [
        sample.get("artifact_refs"),
        _nested(sample, "sample", "artifact_refs"),
        _nested(sample, "judge_output", "artifact_refs"),
        _nested(sample, "model_output", "artifact_refs"),
        _nested(sample, "model_output", "agent_eval", "trial_aggregate", "trial_result_refs"),
    ]
    for trial in sample.get("trial_results") or []:
        if isinstance(trial, dict):
            candidates.append(trial.get("artifact_refs"))
            trace_ref = trial.get("trace_ref")
            if isinstance(trace_ref, dict):
                yield trace_ref
    for trial in _nested(sample, "model_output", "agent_eval", "trial_results") or []:
        if isinstance(trial, dict):
            candidates.append(trial.get("artifact_refs"))
            trace_ref = trial.get("trace_ref")
            if isinstance(trace_ref, dict):
                yield trace_ref
    for refs in candidates:
        if isinstance(refs, list):
            for ref in refs:
                if isinstance(ref, dict):
                    yield ref


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _coerce_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _coerce_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_absolute_or_escaping(value: str) -> bool:
    if value.startswith(("/", "~")):
        return True
    return any(part == ".." for part in PurePosixPath(value).parts)


def _ref_id(kind: str, path: str) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return f"evidence://{kind}/{digest}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp_iso(timestamp_s: float) -> str:
    return datetime.fromtimestamp(timestamp_s, timezone.utc).isoformat().replace("+00:00", "Z")


def _trial_id_from_path(path: str) -> str | None:
    parts = PurePosixPath(path).parts
    if "trials" not in parts:
        return None
    index = parts.index("trials")
    if index + 1 >= len(parts):
        return None
    return parts[index + 1]
