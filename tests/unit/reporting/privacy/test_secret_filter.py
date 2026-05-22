from __future__ import annotations

import copy

import pytest

from gage_eval.reporting.contracts import RedactionFinding
from gage_eval.reporting.privacy import SecretFilter


pytestmark = pytest.mark.fast


def test_secret_filter_detects_default_secret_patterns() -> None:
    text = (
        "Authorization: Bearer abc.def\n"
        "api_key=sk-testsecret1234567890\n"
        "password=hunter2\n"
        "Cookie: session_id=abc123\n"
        "email alice@example.com\n"
        "callback=http://localhost:8080/callback?token=secret\n"
    )

    findings = SecretFilter().detect(text)

    assert all(isinstance(finding, RedactionFinding) for finding in findings)
    assert {finding.kind for finding in findings} >= {
        "auth",
        "token",
        "secret",
        "session",
        "email",
        "private_url",
    }


def test_secret_filter_redacts_nested_values_without_mutating_input() -> None:
    payload = {
        "headers": {"Authorization": "Bearer abc.def"},
        "messages": [
            "contact alice@example.com",
            {"password": "hunter2", "url": "http://127.0.0.1:8080/a?token=secret"},
        ],
    }
    original = copy.deepcopy(payload)

    result = SecretFilter().redact(payload)

    assert payload == original
    assert result.redacted is True
    assert result.value == {
        "headers": {"Authorization": "<redacted:auth>"},
        "messages": [
            "contact <redacted:email>",
            {"password": "<redacted:secret>", "url": "<redacted:private_url>"},
        ],
    }


def test_secret_filter_redacts_json_like_secret_fields_inside_strings() -> None:
    payload = {
        "arguments_json": '{"username":"user@example.com","password":"password123"}'
    }

    result = SecretFilter().redact(payload)

    text = result.value["arguments_json"]
    assert "user@example.com" not in text
    assert "password123" not in text
    assert "<redacted:email>" in text
    assert "<redacted:secret>" in text


def test_secret_filter_assert_safe_raises_without_exposing_secret() -> None:
    with pytest.raises(ValueError) as exc_info:
        SecretFilter().assert_safe("Authorization: Bearer abc.def")

    message = str(exc_info.value)
    assert "auth" in message
    assert "abc.def" not in message
