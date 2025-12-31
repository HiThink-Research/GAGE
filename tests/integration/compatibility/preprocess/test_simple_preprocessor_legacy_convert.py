from gage_eval.assets.datasets.preprocessors.default_preprocessor import DefaultPreprocessor


def test_simple_preprocessor_legacy_prompt_to_standard_sample():
    pre = DefaultPreprocessor()
    legacy = {
        "question": "legacy question",
        "choices": ["A. blue", "B. red"],
        "answer": 0,
        "prompt": "legacy prompt",
    }
    inputs = pre.transform(legacy)
    # transform returns inputs while mutating the original dict into a Sample-like structure.
    assert inputs["prompt"] == "legacy prompt"
    assert legacy["messages"][0]["content"][0]["text"] == "legacy question"
    assert legacy["choices"][0]["message"]["content"][0]["text"] == "A. blue"
    assert legacy["metadata"]["correct_choice"] == "A"
