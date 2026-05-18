from __future__ import annotations

import hashlib
import json
import mimetypes
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable
from urllib.parse import urlparse

from gage_eval.reporting.contracts import EvidenceRef
from gage_eval.reporting.evidence.diagnostics import EvidenceDiagnostics
from gage_eval.reporting.game_artifacts import iter_game_artifact_refs
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
            for media_url in _iter_media_urls(sample):
                ref = self._media_ref(media_url, task_id=task_id, sample_id=sample_id, diagnostics=diagnostics)
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
        ref_path = _normalize_ref_path(
            root,
            path_value,
            allow_absolute_under_root=raw_ref.get("_allow_absolute_under_root") is True,
        )
        if ref_path is None or _is_absolute_or_escaping(ref_path):
            diagnostics.warn("report_pack.artifact_ref_path_not_relative", path=path_value, sample_id=sample_id)
            return None

        artifact_path = root / ref_path
        if not _path_is_under_root(artifact_path, root):
            diagnostics.warn("report_pack.artifact_ref_path_escapes_root", path=ref_path, sample_id=sample_id)
            return None
        preview = self._preview_artifact(artifact_path, diagnostics, ref_path=ref_path)
        mime_type = (
            _coerce_str(raw_ref.get("mime_type"))
            or mimetypes.guess_type(ref_path)[0]
            or "application/octet-stream"
        )
        size_bytes = _coerce_int(raw_ref.get("size_bytes"))
        sha256 = _coerce_str(raw_ref.get("sha256"))
        timestamp_iso = _coerce_str(raw_ref.get("timestamp_iso"))
        if artifact_path.exists():
            stat = artifact_path.stat()
            size_bytes = size_bytes if size_bytes is not None else stat.st_size
            sha256 = sha256 or _sha256(artifact_path)
            timestamp_iso = timestamp_iso or _timestamp_iso(stat.st_mtime)
        else:
            diagnostics.warn("report_pack.artifact_missing", path=ref_path, sample_id=sample_id)
            return None

        ref_id = _ref_id("artifact", ref_path)
        return EvidenceRef(
            ref_id=ref_id,
            kind="artifact",
            path=ref_path,
            scenario_kind=_coerce_str(raw_ref.get("scenario_kind")),
            artifact_role=_coerce_str(raw_ref.get("owner") or raw_ref.get("artifact_role")),
            sample_id=sample_id,
            task_id=task_id,
            trial_id=_trial_id_from_path(ref_path),
            mime_type=mime_type,
            size_bytes=size_bytes,
            sha256=sha256,
            timestamp_iso=timestamp_iso,
            preview=preview,
        )

    def _media_ref(
        self,
        url: str,
        *,
        task_id: str | None,
        sample_id: str | None,
        diagnostics: EvidenceDiagnostics,
    ) -> EvidenceRef | None:
        if not _is_remote_http_url(url):
            return None
        digest = hashlib.sha256(url.encode("utf-8")).hexdigest()
        preview = self._preview_media_url(url, diagnostics, sample_id=sample_id)
        return EvidenceRef(
            ref_id=_media_ref_id(digest, task_id=task_id, sample_id=sample_id),
            kind="media",
            path=f"external://sha256/{digest}",
            sample_id=sample_id,
            task_id=task_id,
            mime_type=_mime_type_from_url(url),
            sha256=digest,
            preview=preview,
        )

    def _preview_media_url(
        self,
        url: str,
        diagnostics: EvidenceDiagnostics,
        *,
        sample_id: str | None,
    ) -> dict[str, Any]:
        parsed = urlparse(url)
        source = f"{parsed.netloc}{parsed.path}" if parsed.netloc else parsed.path
        if len(source) > 240:
            source = f"{source[:117]}...{source[-120:]}"
        preview = {"source": source}
        redacted = self._secret_filter.redact(preview)
        if redacted.redacted:
            diagnostics.warn("report_pack.secret_leak_detected", path="media_url", sample_id=sample_id)
        return redacted.value

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
    yield from _iter_game_artifact_refs(sample)


def _iter_media_urls(sample: dict[str, Any]) -> Iterable[str]:
    seen: set[str] = set()
    for value in _iter_media_values(sample):
        url = _media_url_from_value(value)
        if url is None or url in seen:
            continue
        seen.add(url)
        yield url


def _iter_media_values(sample: dict[str, Any]) -> Iterable[Any]:
    for message in sample.get("messages") or []:
        yield from _iter_message_media_values(message)
    for message in _nested(sample, "sample", "messages") or []:
        yield from _iter_message_media_values(message)
    yield from _iter_media_field_values(_nested(sample, "inputs", "multi_modal_data", "image"))
    yield from _iter_media_field_values(_nested(sample, "sample", "inputs", "multi_modal_data", "image"))


def _iter_media_field_values(value: Any) -> Iterable[Any]:
    if isinstance(value, (str, dict)):
        yield value
        return
    if isinstance(value, list):
        for item in value:
            if isinstance(item, (str, dict)):
                yield item


def _iter_message_media_values(message: Any) -> Iterable[Any]:
    if not isinstance(message, dict):
        return
    content = message.get("content")
    if not isinstance(content, list):
        return
    for item in content:
        if not isinstance(item, dict):
            continue
        image_url = item.get("image_url")
        if image_url is not None:
            yield image_url


def _media_url_from_value(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return _coerce_str(value.get("url") or value.get("path"))
    return None


def _iter_game_artifact_refs(sample: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for ref in iter_game_artifact_refs(sample):
        yield _game_artifact_ref(ref.path, role=ref.role, name=ref.name)


def _game_artifact_ref(path: str, *, role: str, name: str) -> dict[str, Any]:
    return {
        "name": name,
        "path": path,
        "artifact_role": role,
        "scenario_kind": "game",
        "_allow_absolute_under_root": True,
    }


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


def _normalize_ref_path(root: Path, value: str, *, allow_absolute_under_root: bool) -> str | None:
    if value.startswith("~"):
        if not allow_absolute_under_root:
            return None
        expanded = Path(value).expanduser()
        return _relative_to_root(root, expanded)
    if value.startswith("/"):
        if not allow_absolute_under_root:
            return None
        return _relative_to_root(root, Path(value))

    if any(part == ".." for part in PurePosixPath(value).parts):
        return None
    if (root / value).exists():
        return value

    parts = PurePosixPath(value).parts
    for index, part in enumerate(parts):
        if part != root.name:
            continue
        candidate = PurePosixPath(*parts[index + 1 :]).as_posix()
        if candidate and not _is_absolute_or_escaping(candidate) and (root / candidate).exists():
            return candidate
    return value


def _relative_to_root(root: Path, path: Path) -> str | None:
    try:
        return path.expanduser().resolve(strict=False).relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return None


def _path_is_under_root(path: Path, root: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve(strict=False))
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def _ref_id(kind: str, path: str) -> str:
    digest = hashlib.sha1(path.encode("utf-8")).hexdigest()[:12]
    return f"evidence://{kind}/{digest}"


def _media_ref_id(url_digest: str, *, task_id: str | None, sample_id: str | None) -> str:
    identity = hashlib.sha1(f"{task_id or ''}\0{sample_id or ''}\0{url_digest}".encode("utf-8")).hexdigest()
    return f"evidence://media/{url_digest[:12]}-{identity[:12]}"


def _is_remote_http_url(value: str) -> bool:
    parsed = urlparse(value)
    return parsed.scheme in {"http", "https"} and bool(parsed.netloc)


def _mime_type_from_url(url: str) -> str:
    parsed = urlparse(url)
    return mimetypes.guess_type(parsed.path)[0] or "image/*"


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
