from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.methodology_builder import MethodologyBuilder


@pytest.mark.fast
def test_methodology_builder_redacts_config_and_records_caveats() -> None:
    methodology = MethodologyBuilder().build(
        run_metadata={
            "config_digest": "abc",
            "config": {
                "api_key": "sk-abcdefghijklmnopqrstuvwxyz1234567890",
                "password": "hunter2",
                "owner": "alice@example.com",
                "endpoint": "http://127.0.0.1/admin?token=raw",
            },
        },
        metrics=[{"metric_id": "reward_mean", "scope": "run"}],
        runtime_health={"sample_count": 3},
        diagnostics={"report_pack_status": "degraded", "warnings": [{"code": "report_pack.artifact_missing"}]},
    )

    serialized = str(methodology)
    assert "sk-abcdefghijklmnopqrstuvwxyz1234567890" not in serialized
    assert "hunter2" not in serialized
    assert "alice@example.com" not in serialized
    assert "127.0.0.1" not in serialized
    assert "sample_size.small" in serialized
    assert "report_pack.degraded" in serialized
