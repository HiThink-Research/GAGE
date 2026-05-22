from __future__ import annotations

from typing import Any

from gage_eval.assets.datasets.preprocessors import base as base_module
from gage_eval.assets.datasets.preprocessors.base import BasePreprocessor


class _MissingIdPreprocessor(BasePreprocessor):
    def to_sample(self, record: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": "0.0.1",
            "messages": [{"role": "user", "content": "Which option is correct?"}],
            "metadata": {"source_question": record["question"]},
        }


class _BrokenPreprocessor(BasePreprocessor):
    def to_sample(self, record: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        return {
            "schema_version": "0.0.1",
            "id": "",
            "messages": [],
        }


def test_base_preprocessor_generates_stable_id_when_structured_sample_has_no_id() -> None:
    record = {"question": "Which option is correct?", "choices": ["A", "B"]}
    preprocessor = _MissingIdPreprocessor(on_error="raise")

    first = preprocessor.transform(dict(record), dataset_id="vision_demo")
    second = preprocessor.transform(dict(record), dataset_id="vision_demo")

    assert first.id == second.id
    assert first.id.startswith("vision_demo:")
    assert len(first.id.split(":", 1)[1]) == 16


def test_base_preprocessor_warns_when_skip_policy_drops_record(monkeypatch) -> None:
    warnings: list[str] = []
    monkeypatch.setattr(base_module.logger, "warning", lambda message, *args: warnings.append(message.format(*args)))

    result = _BrokenPreprocessor(on_error="skip").transform(
        {"question": "bad record"},
        dataset_id="broken_demo",
    )

    assert result is None
    assert any("Preprocess skipped sample" in message for message in warnings)
