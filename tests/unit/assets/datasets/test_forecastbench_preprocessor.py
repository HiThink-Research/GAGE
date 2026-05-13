"""Tests for ForecastBench preprocessor."""

from __future__ import annotations

from gage_eval.assets.datasets.preprocessors.forecastbench.forecastbench_preprocessor import (
    ForecastBenchPreprocessor,
)


def _minimal_record() -> dict:
    return {
        "id": "0xabc",
        "source": "polymarket",
        "question": "Will X happen?",
        "resolution_criteria": "Resolve per official rules.",
        "background": "Context here.",
        "freeze_datetime": "2025-02-20T00:00:00+00:00",
        "freeze_datetime_value": 0.001,
        "forecast_due_date": "2025-03-02",
        "question_set": "2025-03-02-llm.json",
        "resolved": True,
        "resolved_to": 0.0,
    }


def test_forecastbench_preprocessor_stable_id_and_schema() -> None:
    pre = ForecastBenchPreprocessor()
    rec = _minimal_record()
    sample = pre.transform(rec)
    assert sample is not None
    assert sample.id == "forecastbench:2025-03-02-llm:polymarket:0xabc"
    assert sample.task_type == "probability-forecast"
    assert sample.references == [0.0]
    assert sample.label == "0.0"
    assert sample.metadata["benchmark"] == "forecastbench"
    assert sample.metadata["question_id"] == "0xabc"
    assert sample.data_tag["source"] == "polymarket"
    assert sample.data_tag["question_type"] == "market"


def test_forecastbench_preprocessor_prompt_contract() -> None:
    pre = ForecastBenchPreprocessor()
    sample = pre.transform(_minimal_record())
    assert sample is not None
    text = sample.messages[0].content[0].text
    assert "Do not use web search" in text
    assert '{"forecast": <number between 0 and 1>, "reasoning":' in text


def test_forecastbench_preprocessor_metadata_without_freeze_value() -> None:
    pre = ForecastBenchPreprocessor()
    rec = _minimal_record()
    del rec["freeze_datetime_value"]
    sample = pre.transform(rec)
    assert sample is not None
    assert "freeze_datetime_value" not in (sample.metadata or {})
