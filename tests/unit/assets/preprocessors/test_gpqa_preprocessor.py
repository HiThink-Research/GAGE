import pytest

from gage_eval.assets.datasets.preprocessors.gpqa_preprocessor import (
    GpqaPreprocessor,
    GpqaStructOnlyPreprocessor,
)


@pytest.mark.fast
def test_gpqa_preprocess_deterministic_shuffle(monkeypatch):
    """多选题：校验选项重排后 correct_choice/option_map/标记。"""

    # 固定 shuffle 顺序：反转列表
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

    # 选项应被重排并带标签
    labels = [c["label"] for c in out["choices"]]
    assert labels == ["A", "B", "C", "D"]
    option_texts = [c["message"]["content"][0]["text"] for c in out["choices"]]
    assert option_texts == ["False3", "False2", "False1", "True"]  # reverse 后 True 在末尾

    meta = out["metadata"]
    assert meta["option_map"]["D"] == "True"
    assert meta["correct_choice"] == "D"
    assert out["chat_template_mode"] == "preprocess"
    assert out["cache_suffix"] == "-converted"


@pytest.mark.fast
def test_gpqa_struct_only_strips_prompt():
    """Struct-only 仅保留结构化字段，移除渲染标记。"""

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
    # choices/metadata 应保留
    assert out["choices"]
    assert out["metadata"]["correct_choice"] in {"A", "B", "C", "D"}


@pytest.mark.fast
def test_gpqa_preprocess_applies_tokenizer_chat_template(monkeypatch):
    """有 tokenizer 时应使用 chat_template 渲染 prompt。"""

    # 固定 shuffle 顺序，便于断言内容
    monkeypatch.setattr("gage_eval.assets.datasets.preprocessors.gpqa_preprocessor.random.shuffle", lambda x: x.reverse())

    class DummyTokenizer:
        def __init__(self):
            self.calls = 0

        def apply_chat_template(self, messages, **kwargs):
            self.calls += 1
            return "templated_prompt"

    tok = DummyTokenizer()
    pre = GpqaPreprocessor(tokenizer=tok, tokenizer_path="tok-path")
    sample = {
        "Question": "Which is correct?",
        "Correct Answer": "True",
        "Incorrect Answer 1": "False1",
        "Incorrect Answer 2": "False2",
    }

    out = pre.to_sample(sample)
    # chat_template 结果应覆盖 prompt/inputs，并打 chat_template 标记
    assert out["prompt"] == "templated_prompt"
    assert out["inputs"] == {"prompt": "templated_prompt"}
    assert out.get("template_source") == "model"
    assert out.get("cache_suffix") == "-chat_template"
    assert out.get("_tokenizer_path") == "tok-path"
    assert tok.calls == 1
