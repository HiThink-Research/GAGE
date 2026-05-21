from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gage_eval.reporting.evidence.diagnostics import EvidenceDiagnostics


class RunLayoutConsistencyChecker:
    """Reports gaps in the persisted run layout without raising for bad input."""

    def check(self, run_dir: str | Path) -> EvidenceDiagnostics:
        root = Path(run_dir)
        diagnostics = EvidenceDiagnostics()
        try:
            summary = self._read_summary(root, diagnostics)
            samples = self._read_samples(root, diagnostics)
            self._check_sample_count(summary, samples, diagnostics)
            self._check_sample_details(root, samples, diagnostics)
            self._check_artifact_refs(root, samples, diagnostics)
        except Exception as exc:
            diagnostics.error("report_pack.layout_check_failed", message=str(exc))
        return diagnostics

    def _read_summary(self, root: Path, diagnostics: EvidenceDiagnostics) -> dict[str, Any]:
        path = root / "summary.json"
        if not path.exists():
            diagnostics.warn("report_pack.summary_missing", path="summary.json")
            return {}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            diagnostics.error("report_pack.summary_unreadable", path="summary.json", message=str(exc))
            return {}
        return data if isinstance(data, dict) else {}

    def _read_samples(self, root: Path, diagnostics: EvidenceDiagnostics) -> list[dict[str, Any]]:
        path = root / "samples.jsonl"
        if not path.exists():
            diagnostics.warn("report_pack.samples_missing", path="samples.jsonl")
            return []
        samples: list[dict[str, Any]] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except Exception as exc:
            diagnostics.error("report_pack.samples_unreadable", path="samples.jsonl", message=str(exc))
            return []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception as exc:
                diagnostics.error("report_pack.sample_record_unreadable", line=line_number, message=str(exc))
                continue
            if isinstance(record, dict):
                samples.append(record)
        return samples

    def _check_sample_count(
        self,
        summary: dict[str, Any],
        samples: list[dict[str, Any]],
        diagnostics: EvidenceDiagnostics,
    ) -> None:
        expected = summary.get("sample_count")
        if expected is None:
            return
        try:
            expected_count = int(expected)
        except (TypeError, ValueError):
            diagnostics.warn("report_pack.sample_count_invalid", value=expected)
            return
        if expected_count != len(samples):
            diagnostics.warn(
                "report_pack.sample_count_mismatch",
                expected=expected_count,
                actual=len(samples),
            )

    def _check_sample_details(
        self,
        root: Path,
        samples: list[dict[str, Any]],
        diagnostics: EvidenceDiagnostics,
    ) -> None:
        detail_paths = sorted((root / "samples").glob("*/*.json")) if (root / "samples").exists() else []
        diagnostics.derived_detail_count = len(detail_paths)
        known_sample_ids = {_sample_id(sample) for sample in samples}
        known_sample_ids.discard(None)
        for detail in detail_paths:
            if detail.stem not in known_sample_ids:
                diagnostics.warn(
                    "report_pack.derived_sample_detail_without_journal_record",
                    path=str(detail.relative_to(root)),
                )

    def _check_artifact_refs(
        self,
        root: Path,
        samples: list[dict[str, Any]],
        diagnostics: EvidenceDiagnostics,
    ) -> None:
        for sample in samples:
            for ref in _artifact_refs(sample):
                path_value = ref.get("path")
                if not isinstance(path_value, str) or not path_value:
                    diagnostics.warn("report_pack.artifact_ref_missing_path", sample_id=_sample_id(sample))
                    continue
                if path_value.startswith(("/", "~")) or ".." in Path(path_value).parts:
                    diagnostics.warn("report_pack.artifact_ref_path_not_relative", path=path_value)
                    continue
                if not (root / path_value).exists():
                    diagnostics.warn("report_pack.artifact_missing", path=path_value, sample_id=_sample_id(sample))


def _sample_id(sample: dict[str, Any]) -> str | None:
    value = sample.get("sample_id")
    if value is None and isinstance(sample.get("sample"), dict):
        value = sample["sample"].get("id")
    return str(value) if value is not None else None


def _artifact_refs(sample: dict[str, Any]) -> list[dict[str, Any]]:
    refs = sample.get("artifact_refs")
    return [ref for ref in refs if isinstance(ref, dict)] if isinstance(refs, list) else []
