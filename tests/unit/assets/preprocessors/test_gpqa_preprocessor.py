import pytest

from gage_eval.assets.datasets.preprocessors.gpqa_preprocessor import (
    GpqaPreprocessor,
    GpqaStructOnlyPreprocessor,
)


@pytest.mark.fast
def test_gpqa_preprocess_deterministic_shuffle(monkeypatch):
    """Multi-choice: verifies correct_choice/option_map/render flags after shuffling."""

    # Force a deterministic shuffle: reverse the list.
    monkeypatch.setattr("gage_eval.assets.datasets.preprocessors.gpqa_preprocessor.random.shuffle", lambda x: x.reverse())

    pre = GpqaPreprocessor()
    sample = {
        "Question": "Which is correct?",
        "Correct Answer": "True",
        "Incorrect Answer 1": "False1",
        "Incorrect Answer 2": "False2",
        "Incorrect Answer 3": "False3",
    }
    out = pre.to_sample(sample)

    # Choices should be re-ordered and labeled.
    labels = [c["label"] for c in out["choices"]]
    assert labels == ["A", "B", "C", "D"]
    option_texts = [c["message"]["content"][0]["text"] for c in out["choices"]]
    assert option_texts == ["False3", "False2", "False1", "True"]  # after reverse, True is last

    meta = out["metadata"]
    assert meta["option_map"]["D"] == "True"
    assert meta["correct_choice"] == "D"
    assert out["chat_template_mode"] == "preprocess"
    assert out["cache_suffix"] == "-converted"


@pytest.mark.fast
def test_gpqa_struct_only_strips_prompt():
    """Struct-only keeps structured fields and removes render flags."""

    pre = GpqaStructOnlyPreprocessor()
    sample = {
        "Question": "Q?",
        "Correct Answer": "Yes",
        "Incorrect Answer 1": "No",
    }
    out = pre.to_sample(sample)

    assert out.get("prompt") is None
    assert out.get("messages") == []
    assert out.get("inputs") == {}
    for key in ("chat_template_mode", "template_source", "rendered_by", "cache_suffix"):
        assert key not in out
    # choices/metadata should be kept.
    assert out["choices"]
    assert out["metadata"]["correct_choice"] in {"A", "B", "C", "D"}


@pytest.mark.fast
def test_gpqa_preprocess_applies_tokenizer_chat_template(monkeypatch):
    """Uses tokenizer chat_template rendering when a tokenizer is provided."""

    # Force deterministic shuffle for stable assertions.
    monkeypatch.setattr("gage_eval.assets.datasets.preprocessors.gpqa_preprocessor.random.shuffle", lambda x: x.reverse())

    class DummyTokenizer:
        def __init__(self):
            self.calls = 0

        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            return "templated_prompt"

        def encode(self, text):
            return [1, 2, 3]

    tok = DummyTokenizer()
    pre = GpqaPreprocessor(tokenizer=tok, tokenizer_path="tok-path")
    sample = {
        "Question": "Which is correct?",
        "Correct Answer": "True",
        "Incorrect Answer 1": "False1",
        "Incorrect Answer 2": "False2",
    }

    out = pre.to_sample(sample)
    # The rendered prompt should populate prompt/inputs and set template flags.
    assert out["prompt"] == "templated_prompt"
    assert out["inputs"]["prompt"] == "templated_prompt"
    assert out["inputs"]["input_ids"] == [1, 2, 3]
    assert out.get("template_source") == "model"
    assert out.get("cache_suffix") == "-chat_template"
    assert out.get("_tokenizer_path") == "tok-path"
    assert tok.calls == 1
