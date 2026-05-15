from __future__ import annotations

import pytest

from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.run_metadata import RunMetadata, RuntimeStats, ValidationSummary


@pytest.mark.fast
def test_run_metadata_rejects_secret_config_digest() -> None:
    with pytest.raises(ValueError, match="config_digest"):
        RunMetadata(
            run_id="run",
            run_dir="runs/run",
            config_digest="api_key=plain-token",
        )


@pytest.mark.fast
def test_runtime_stats_covers_required_runtime_health_fields() -> None:
    stats = RuntimeStats(
        sample_count=3,
        completed_count=1,
        failed_count=1,
        aborted_count=1,
        scheduler_failed_count=1,
        verifier_skipped_count=1,
        observability_health={"events_emitted_total": 2},
    )

    assert stats.to_summary_payload() == {
        "runtime_health": {
            "sample_count": 3,
            "completed_count": 1,
            "failed_count": 1,
            "aborted_count": 1,
            "scheduler_failed_count": 1,
            "verifier_skipped_count": 1,
        },
        "observability_health": {"events_emitted_total": 2},
    }


@pytest.mark.fast
def test_validation_summary_redacts_errors() -> None:
    summary = ValidationSummary.from_dict(
        {
            "samples_total": 1,
            "samples_valid": 0,
            "samples_dropped": 1,
            "errors": [{"message": "Authorization: Bearer abc123"}],
        }
    )

    payload = summary.to_metadata_value()

    assert "Bearer abc123" not in str(payload)
    assert "<redacted:auth>" in str(payload)


@pytest.mark.fast
def test_eval_cache_records_typed_validation_summary(tmp_path) -> None:
    cache = EvalCache(base_dir=tmp_path, run_id="typed-metadata")

    cache.record_validation_summary(
        ValidationSummary.from_dict(
            {
                "samples_total": 1,
                "samples_valid": 0,
                "samples_dropped": 1,
                "errors": [{"message": "alice@example.com"}],
            }
        )
    )

    payload = cache.get_metadata("validation_summary")
    assert payload["samples_total"] == 1
    assert "alice@example.com" not in str(payload)
