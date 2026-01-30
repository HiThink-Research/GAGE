from gage_eval.assets.datasets.preprocessors.appworld_preprocessor import AppWorldPreprocessor


def test_appworld_preprocessor_builds_sample() -> None:
    preprocessor = AppWorldPreprocessor(
        subset="dev",
        ground_truth_mode="full",
        experiment_name="appworld_dev",
    )
    record = {
        "task_id": "calendar_001",
        "instruction": "Create an event",
        "metadata": {"appworld": {"allowed_apps": ["calendar"]}},
    }
    sample = preprocessor.transform(record)

    assert sample.id == "calendar_001"
    assert sample.metadata["appworld"]["task_id"] == "calendar_001"
    assert sample.metadata["appworld"]["subset"] == "dev"
    assert sample.metadata["appworld"]["ground_truth_mode"] == "full"
    allowed_apps = sample.metadata["appworld"]["allowed_apps"]
    assert "calendar" in allowed_apps
    assert "api_docs" in allowed_apps
    assert "supervisor" in allowed_apps
    assert sample.metadata["appworld"]["experiment_name"] == "appworld_dev"
    assert sample.messages[0].content[0].text == "Create an event"


def test_appworld_preprocessor_forces_minimal_on_test() -> None:
    preprocessor = AppWorldPreprocessor(ground_truth_mode="full")
    record = {
        "task_id": "calendar_002",
        "instruction": "Schedule a meeting",
        "metadata": {"appworld": {"subset": "test_normal"}},
    }
    sample = preprocessor.transform(record)

    assert sample.metadata["appworld"]["ground_truth_mode"] == "minimal"


def test_appworld_preprocessor_generates_experiment_name() -> None:
    preprocessor = AppWorldPreprocessor()
    record_one = {"task_id": "calendar_003", "instruction": "Cancel the event"}
    record_two = {"task_id": "calendar_004", "instruction": "Update the event"}

    sample_one = preprocessor.transform(record_one)
    sample_two = preprocessor.transform(record_two)

    name_one = sample_one.metadata["appworld"]["experiment_name"]
    name_two = sample_two.metadata["appworld"]["experiment_name"]

    assert name_one.startswith("appworld-experiment-")
    assert name_one == name_two
