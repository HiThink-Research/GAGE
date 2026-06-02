from __future__ import annotations

import pytest

from gage_eval.reporting.evidence.external_harness.harbor import HarborHarnessAdapter
from gage_eval.reporting.evidence.reader import RunEvidenceIndex


@pytest.mark.fast
def test_harbor_adapter_detects_harbor_records_without_leaking_source_fields() -> None:
    index = RunEvidenceIndex(
        run_dir="run",
        samples=[{"sample": {"task_type": "external_harness.harbor", "metadata": {"_harness": {"harbor_task_key": "x"}}}}],
    )

    adapter = HarborHarnessAdapter()
    job = adapter.normalize_job(index).to_dict()

    assert adapter.detect(index) is True
    assert job["harness_id"] == "harbor"
    assert "harbor_task_key" not in job
