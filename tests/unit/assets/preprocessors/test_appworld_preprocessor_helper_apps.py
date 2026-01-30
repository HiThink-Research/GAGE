from __future__ import annotations

from gage_eval.assets.datasets.preprocessors.appworld_preprocessor import AppWorldPreprocessor


def test_appworld_preprocessor_injects_helper_apps() -> None:
    preprocessor = AppWorldPreprocessor()
    record = {
        "task_id": "calendar_helper_001",
        "instruction": "Create an event",
        "metadata": {"appworld": {"allowed_apps": ["calendar"]}},
    }

    sample = preprocessor.transform(record)

    assert sample.metadata["appworld"]["allowed_apps"] == ["calendar", "api_docs", "supervisor"]


def test_appworld_preprocessor_skips_allowed_apps_when_missing() -> None:
    preprocessor = AppWorldPreprocessor()
    record = {
        "task_id": "calendar_helper_002",
        "instruction": "Cancel the event",
    }

    sample = preprocessor.transform(record)

    assert "allowed_apps" not in sample.metadata["appworld"]
