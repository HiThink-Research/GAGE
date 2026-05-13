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
    assert text.startswith("You are an expert superforecaster")
    assert "Question: Will X happen?" in text
    assert "Question Background: Context here." in text
    assert "Resolution Criteria: Resolve per official rules." in text
    assert "Market value on 2025-02-20T00:00:00+00:00: 0.001" in text
    assert "Today's Date: 2025-03-02" in text
    assert "Output your answer (a number between 0 and 1) with an asterisk" in text
    assert "Do not output anything else." in text
    assert "Return JSON" not in text
    assert sample.metadata["freeze_datetime_value"] == 0.001


def test_forecastbench_preprocessor_can_omit_market_baseline_for_ablation() -> None:
    pre = ForecastBenchPreprocessor()
    sample = pre.transform(_minimal_record(), include_market_baseline_in_prompt=False)
    assert sample is not None
    text = sample.messages[0].content[0].text
    assert "Market value on" not in text
    assert sample.metadata["freeze_datetime_value"] == 0.001


def test_forecastbench_preprocessor_metadata_without_freeze_value() -> None:
    pre = ForecastBenchPreprocessor()
    rec = _minimal_record()
    del rec["freeze_datetime_value"]
    sample = pre.transform(rec)
    assert sample is not None
    assert "freeze_datetime_value" not in (sample.metadata or {})
