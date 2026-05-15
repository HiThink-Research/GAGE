from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import ReasonCodeRegistry


pytestmark = pytest.mark.fast


REQUIRED_REASON_CODES = {
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
}


def test_builtin_reason_codes_include_required_defaults() -> None:
    registry = ReasonCodeRegistry.load_builtin()

    assert REQUIRED_REASON_CODES.issubset(set(registry.codes))
    for code in REQUIRED_REASON_CODES:
        entry = registry.get(code)
        assert entry.impact_default in {"critical", "high", "medium", "low", "unknown"}
        assert entry.actionability_default in {"high", "medium", "low", "unknown"}
        assert entry.human_readable_zh
        assert entry.human_readable_en


def test_reason_code_registry_detects_declared_codes_missing_from_yaml() -> None:
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

    diagnostics = registry.validate_completeness(
        declared_codes=["scheduler.failed", "runtime.error"]
    )

    assert diagnostics == [
        {
            "code": "reason_code.unregistered",
            "path": "reason_codes.runtime.error",
            "message": "Reason code is declared but not registered: runtime.error",
        }
    ]
