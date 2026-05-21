from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.case_details_builder import CaseDetailsBuilder


@pytest.mark.fast
def test_case_details_builder_truncates_and_redacts_preview() -> None:
    details = CaseDetailsBuilder(max_messages=1, max_tool_calls=1, max_preview_bytes=256).build(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Authorization: Bearer abc123 password=hunter2 "
                        "alice@example.com http://192.168.1.10/admin?token=raw"
                    ),
                },
                {"role": "assistant", "content": "second"},
            ],
            "tool_calls": [{"tool": "a"}, {"tool": "b"}],
            "scoring_breakdown": {"reward": 0, "api_key": "plain-token"},
        },
    )

    payload = details.to_dict()
    assert "Bearer abc123" not in str(payload)
    assert "hunter2" not in str(payload)
    assert "alice@example.com" not in str(payload)
    assert "192.168.1.10" not in str(payload)
    assert "plain-token" not in str(payload)
    assert payload["truncated"] is True
    assert len(payload["message_history_preview"]) == 1


@pytest.mark.fast
def test_case_details_builder_does_not_embed_case_id() -> None:
    details = CaseDetailsBuilder().build(
        {
            "evidence_ref_ids": ["evidence://sample/task/sample"],
            "artifact_preview_ref_ids": ["evidence://sample/task/sample"],
        },
    )

    payload = details.to_dict()
    assert "case_id" not in payload
