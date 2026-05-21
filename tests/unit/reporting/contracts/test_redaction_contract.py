from __future__ import annotations

import pytest

from gage_eval.reporting.contracts import RedactionFinding, RedactionResult, SecretPattern


pytestmark = pytest.mark.fast


def test_redaction_contract_round_trips_findings_without_secret_value() -> None:
    finding = RedactionFinding(kind="token", start=4, end=18, pattern_name="openai_key")
    result = RedactionResult(
        value="key=<redacted:token>",
        findings=[finding],
        redacted=True,
    )

    reloaded = RedactionResult.from_dict(result.to_dict())

    assert reloaded.to_dict() == {
        "value": "key=<redacted:token>",
        "findings": [
            {"kind": "token", "start": 4, "end": 18, "pattern_name": "openai_key"}
        ],
        "redacted": True,
    }


def test_secret_pattern_placeholder_uses_redacted_kind_shape() -> None:
    pattern = SecretPattern(kind="auth", name="bearer", regex=r"Bearer\s+\S+")

    assert pattern.placeholder == "<redacted:auth>"
    assert pattern.to_dict() == {
        "kind": "auth",
        "name": "bearer",
        "regex": r"Bearer\s+\S+",
    }
