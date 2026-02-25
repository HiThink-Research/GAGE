from __future__ import annotations

import pytest

from gage_eval.assets.datasets.preprocessors.inverse_ifeval_preprocessor import InverseIFEvalPreprocessor


@pytest.mark.fast
def test_inverse_ifeval_preprocessor_maps_fields() -> None:
    preprocessor = InverseIFEvalPreprocessor()
    record = {
        "id": "sample-1",
        "prompt": "Return only YES.",
        "response_reference": "YES",
        "language": "chinese",
        "judge_prompt_template": "<Question>: {prompt}\\n<Reference>: {response_reference}\\n<Response>: {response}",
        "judge_system_prompt": "You are a strict grader.",
        "constraints": [
            {"id": "c1", "type": "must_contain", "value": "yes"},
            {"id": "c2", "type": "max_length", "value": 5},
        ],
        "instruction_id_list": ["contains"],
        "kwargs": {"contains": "yes"},
        "schema": {"version": "v1"},
        "answer": "YES",
    }

    sample = preprocessor.to_sample(record)

    assert sample.id == "sample-1"
    assert sample.messages[0].role == "user"
    assert sample.messages[0].content[0].text == "Return only YES."
    assert sample.references == ["YES"]
    assert sample.label == "YES"
    assert sample.metadata is not None
    assert sample.metadata["constraints"][0]["id"] == "c1"
    assert sample.metadata["instruction_id_list"] == ["contains"]
    assert sample.metadata["kwargs"]["contains"] == "yes"
    assert sample.metadata["raw_schema_fragment"]["version"] == "v1"
    assert sample.metadata["prompt_text"] == "Return only YES."
    assert sample.metadata["response_reference"] == "YES"
    assert sample.metadata["language"] == "chinese"
    assert sample.metadata["judge_prompt_template"].startswith("<Question>")
    assert sample.metadata["judge_system_prompt"] == "You are a strict grader."


@pytest.mark.fast
def test_inverse_ifeval_preprocessor_id_fallback_and_prompt_fallback() -> None:
    preprocessor = InverseIFEvalPreprocessor()
    record = {
        "question": "Only output abc.",
        "constraints": ["abc"],
    }

    sample = preprocessor.to_sample(record)

    assert sample.id.startswith("inverse_ifeval_")
    assert sample.messages[0].content[0].text == "Only output abc."


@pytest.mark.fast
def test_inverse_ifeval_preprocessor_empty_references_when_missing() -> None:
    preprocessor = InverseIFEvalPreprocessor()
    record = {
        "sample_id": "s-2",
        "input": "Reply with one token.",
        "instruction_ids": ["max_length"],
        "instruction_kwargs": {"max_length": 1},
    }

    sample = preprocessor.to_sample(record)

    assert sample.id == "s-2"
    assert sample.references == []
    assert sample.label is None
    assert sample.metadata is not None
    assert sample.metadata["instruction_id_list"] == ["max_length"]
    assert sample.metadata["kwargs"]["max_length"] == 1


@pytest.mark.fast
def test_inverse_ifeval_preprocessor_uses_response_reference_as_reference() -> None:
    preprocessor = InverseIFEvalPreprocessor()
    record = {
        "sample_id": "s-3",
        "input": "Output exactly the keyword.",
        "response_reference": "keyword",
    }

    sample = preprocessor.to_sample(record)

    assert sample.references == ["keyword"]
    assert sample.label == "keyword"
    assert sample.metadata is not None
    assert sample.metadata["response_reference"] == "keyword"
