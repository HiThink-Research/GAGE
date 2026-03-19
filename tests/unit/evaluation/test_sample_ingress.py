from __future__ import annotations

import pytest

from gage_eval.assets.datasets.validation import ValidationFailure
from gage_eval.evaluation.sample_ingress import (
    SampleIdentityResolver,
    SampleIngressCoordinator,
    SampleIngressPolicy,
    SampleValidationGateError,
    ValidationLedger,
    resolve_runtime_sample_id,
)


@pytest.mark.fast
def test_sample_identity_resolver_builds_synthetic_id_from_task_and_source_index() -> None:
    resolver = SampleIdentityResolver()

    prepared = resolver.prepare_sample(
        {"messages": []},
        dataset_id="dataset-alpha",
        source_index=7,
        task_id="task-alpha",
    )

    assert prepared["id"] == "task-alpha:7"
    assert prepared["sample_id"] == "task-alpha:7"
    assert prepared["_gage_sample_id_source"] == "synthetic_source_index"
    assert prepared["_gage_source_index"] == 7


@pytest.mark.fast
def test_resolve_runtime_sample_id_prefers_prepared_identity() -> None:
    sample = {
        "id": "task-alpha:3",
        "sample_id": "task-alpha:3",
        "_gage_sample_id_source": "synthetic_source_index",
        "_gage_source_index": 3,
        "_gage_task_id": "task-alpha",
    }

    assert resolve_runtime_sample_id(sample) == "task-alpha:3"


@pytest.mark.fast
def test_sample_ingress_coordinator_rejects_missing_sample_id_in_strict_mode() -> None:
    coordinator = SampleIngressCoordinator(
        dataset_id="dataset-alpha",
        validator=None,
        policy=SampleIngressPolicy(strict_sample_id=True),
        task_id="task-alpha",
    )

    with pytest.raises(SampleValidationGateError, match="sample id is required"):
        list(
            coordinator.prepare(
                [{"messages": [], "_gage_source_index": 0, "_gage_dataset_id": "dataset-alpha"}]
            )
        )


@pytest.mark.fast
def test_validation_ledger_tracks_counts_and_gate_payload() -> None:
    ledger = ValidationLedger(drop_reasons_limit=2)

    ledger.record_seen()
    ledger.record_failure("raw_record:invalid_record_type")
    ledger.record_seen()
    ledger.record_valid()
    ledger.mark_gate_failure(
        code="drop_ratio_exceeded",
        message="samples drop ratio exceeded configured threshold",
    )

    snapshot = ledger.snapshot()

    assert snapshot["samples_total"] == 2
    assert snapshot["samples_valid"] == 1
    assert snapshot["samples_dropped"] == 1
    assert snapshot["samples_drop_ratio"] == 0.5
    assert snapshot["drop_reasons_top"] == [
        {"reason": "raw_record:invalid_record_type", "count": 1}
    ]
    assert snapshot["validation_gate_triggered"] is True
    assert snapshot["validation_gate_error_code"] == "drop_ratio_exceeded"


@pytest.mark.fast
def test_sample_ingress_coordinator_rejects_when_drop_ratio_exceeds_threshold() -> None:
    aggregate_ledger = ValidationLedger()
    coordinator = SampleIngressCoordinator(
        dataset_id="dataset-alpha",
        validator=None,
        policy=SampleIngressPolicy(max_drop_ratio=0.4),
        task_id="task-alpha",
        aggregate_ledger=aggregate_ledger,
    )

    coordinator.record_seen(0)
    coordinator.record_failure(
        ValidationFailure(
            step="raw_record",
            dataset_id="dataset-alpha",
            message="invalid record",
            reason="raw_record:invalid_record_type",
            index=0,
        )
    )
    coordinator.record_seen(1)

    with pytest.raises(SampleValidationGateError, match="max_drop_ratio=0.4000"):
        list(
            coordinator.prepare(
                [
                    {
                        "id": "explicit-1",
                        "messages": [],
                        "_gage_source_index": 1,
                        "_gage_dataset_id": "dataset-alpha",
                    }
                ]
            )
        )

    snapshot = aggregate_ledger.snapshot()
    assert snapshot["samples_total"] == 2
    assert snapshot["samples_valid"] == 1
    assert snapshot["samples_dropped"] == 1
    assert snapshot["validation_gate_triggered"] is True
    assert snapshot["validation_gate_error_code"] == "drop_ratio_exceeded"


@pytest.mark.fast
def test_sample_ingress_coordinator_rejects_when_valid_count_below_threshold() -> None:
    aggregate_ledger = ValidationLedger()
    coordinator = SampleIngressCoordinator(
        dataset_id="dataset-alpha",
        validator=None,
        policy=SampleIngressPolicy(min_valid_samples=2),
        task_id="task-alpha",
        aggregate_ledger=aggregate_ledger,
    )

    coordinator.record_seen(0)

    with pytest.raises(SampleValidationGateError, match="min_valid_samples=2"):
        list(
            coordinator.prepare(
                [
                    {
                        "id": "explicit-1",
                        "messages": [],
                        "_gage_source_index": 0,
                        "_gage_dataset_id": "dataset-alpha",
                    }
                ]
            )
        )

    snapshot = aggregate_ledger.snapshot()
    assert snapshot["samples_total"] == 1
    assert snapshot["samples_valid"] == 1
    assert snapshot["samples_dropped"] == 0
    assert snapshot["validation_gate_triggered"] is True
    assert snapshot["validation_gate_error_code"] == "min_valid_samples_violation"
