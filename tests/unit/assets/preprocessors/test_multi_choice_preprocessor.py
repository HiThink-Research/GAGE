from gage_eval.assets.datasets.preprocessors.multi_choice_preprocessor import MultiChoicePreprocessor


def test_multi_choice_preprocessor_generates_id_when_source_has_no_id() -> None:
    sample = MultiChoicePreprocessor(on_error="raise").transform(
        {
            "question": "Which option is correct?",
            "subject": "demo",
            "choices": ["wrong", "right", "also wrong", "still wrong"],
            "answer": 1,
        },
        dataset_id="mmlu_business_ethics",
    )

    assert sample.id.startswith("mmlu_business_ethics:")
    assert sample.references == ["B"]
    assert sample.label == "B"
    assert sample.metadata["correct_choice"] == "B"
