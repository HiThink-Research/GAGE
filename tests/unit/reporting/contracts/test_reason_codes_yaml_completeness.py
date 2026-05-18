from __future__ import annotations

import pytest

from gage_eval.reporting.contracts.reason_codes import ReasonCodeRegistry


@pytest.mark.fast
def test_builtin_reason_codes_are_registered() -> None:
    registry = ReasonCodeRegistry.load_default()

    for code in [
        "scheduler.failed",
        "verifier.skipped",
        "runtime.error",
        "timeout",
        "score.low",
        "outlier.latency",
        "outlier.token",
        "artifact.missing",
        "observability.degraded",
        "expected.failure",
        "appworld.missing_success_signal",
        "verifier.run_verifier.appworld.verifier_failed",
        "verifier.skipped_due_to_scheduler_failure",
        "client_execution.tool_retry_budget_exhausted",
    ]:
        entry = registry.get(code)
        assert entry["impact_default"]
        assert entry["actionability_default"]
        assert entry["human_readable_zh"]
        assert entry["human_readable_en"]


@pytest.mark.fast
def test_registry_detects_declared_but_unregistered_codes() -> None:
    registry = ReasonCodeRegistry.from_dict(
        {
            "schema_version": "gage.reason_codes.v1",
            "reason_codes": {
                "scheduler.failed": {
                    "impact_default": "high",
                    "actionability_default": "high",
                    "human_readable_zh": "调度失败",
                    "human_readable_en": "Scheduler failed",
                }
            },
        }
    )

    missing = registry.validate_completeness({"scheduler.failed", "runtime.error"})

    assert missing == ["runtime.error"]
