from __future__ import annotations

from dataclasses import dataclass

import pytest

from gage_eval.role.common.backend_utils import finalize_backend_result


@dataclass
class FakeCompletion:
    text: str
    token_ids: list[int]


class FakeVLLMOutput:
    def __init__(self) -> None:
        self.request_id = "req-1"
        self.prompt = "hello"
        self.outputs = [FakeCompletion(text="tool text", token_ids=[1, 2, 3])]


@pytest.mark.fast
def test_finalize_backend_result_records_json_safe_raw_response() -> None:
    result = finalize_backend_result(
        {"sample": {"id": "s1"}},
        ["tool text"],
        sample_n=1,
        batch_path="native_single",
        backend_tag="vllm_backend",
        raw_outputs=[FakeVLLMOutput()],
    )

    assert result["answer"] == "tool text"
    assert result["raw_response"]["request_id"] == "req-1"
    assert result["raw_response"]["prompt"] == "hello"
    assert result["raw_response"]["outputs"][0]["text"] == "tool text"
    assert result["raw_response"]["outputs"][0]["token_ids"] == [1, 2, 3]
    assert result["raw_response"]["object_type"].endswith("FakeVLLMOutput")


@pytest.mark.fast
def test_finalize_backend_result_records_multiple_raw_responses() -> None:
    result = finalize_backend_result(
        {},
        ["a", "b"],
        sample_n=2,
        batch_path="native_single",
        backend_tag="vllm_backend",
        raw_outputs=[{"outputs": [{"text": "a"}]}, {"outputs": [{"text": "b"}]}],
    )

    assert result["answer"] == ["a", "b"]
    assert result["raw_response"] == [
        {"outputs": [{"text": "a"}]},
        {"outputs": [{"text": "b"}]},
    ]
