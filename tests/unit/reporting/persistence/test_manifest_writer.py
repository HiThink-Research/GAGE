from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import EvidenceRef
from gage_eval.reporting.persistence.manifest_writer import ManifestWriter


@pytest.mark.io
def test_manifest_writer_requires_integrity_fields(tmp_path) -> None:
    ref = EvidenceRef(
        ref_id="evidence://sample/task/sample",
        kind="sample_record",
        path="samples.jsonl",
        mime_type="application/jsonl",
        size_bytes=1,
        sha256="0" * 64,
        timestamp_iso="2026-05-14T00:00:00+00:00",
    )

    path = ManifestWriter().write(tmp_path, run_id="run", evidence_refs=[ref])

    text = path.read_text(encoding="utf-8")
    assert "sha256" in text
    assert "size_bytes" in text
    assert "timestamp_iso" in text
