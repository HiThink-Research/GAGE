from __future__ import annotations

import pytest

from gage_eval.reporting.evidence.manifest import SampleEvidenceManifest


@pytest.mark.fast
def test_sample_evidence_manifest_serializes_redaction_status() -> None:
    manifest = SampleEvidenceManifest(
        sample_key="task/sample",
        task_id="task",
        sample_id="sample",
        root_journal_ref_id="evidence://sample/task/sample",
        redaction_status="redacted",
    )

    payload = manifest.to_dict()

    assert payload["namespace_detail_ref_ids"] == []
    assert payload["artifact_ref_ids"] == []
    assert payload["trial_ref_ids"] == []
    assert payload["redaction_status"] == "redacted"
