from __future__ import annotations

import pytest

from gage_eval.reporting.evidence.external_harness.base import ExternalHarnessAdapter, ExternalHarnessJob


class _Adapter(ExternalHarnessAdapter):
    harness_id = "demo"
    adapter_id = "external_harness.demo"

    def detect(self, evidence) -> bool:
        return True

    def normalize_job(self, evidence):
        return ExternalHarnessJob(harness_id="demo", job_id="job", status="completed", raw_ref_ids=[])

    def normalize_trials(self, evidence):
        return []

    def project_metrics(self, evidence):
        return []


@pytest.mark.fast
def test_external_harness_adapter_contract_normalizes_job() -> None:
    job = _Adapter().normalize_job(object())

    assert job.to_dict()["harness_id"] == "demo"
