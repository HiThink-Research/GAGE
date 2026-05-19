"""Tests for ForecastBench preprocessor."""

from __future__ import annotations

import hashlib

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
    expected_hash = hashlib.sha1(b"0xabc").hexdigest()[:8]
    assert sample.id == f"fb:2025-03-02:pm:0xabc_{expected_hash}"
    assert sample.task_type == "probability-forecast"
    assert sample.references == [0.0]
    assert sample.label == "0.0"
    assert sample.metadata["benchmark"] == "forecastbench"
    # Full original identifiers preserved in metadata so the legacy
    # ``forecastbench:<stem>:<source>:<id>`` form is always reconstructable.
    assert sample.metadata["question_id"] == "0xabc"
    assert sample.metadata["question_set"] == "2025-03-02-llm.json"
    assert sample.metadata["source"] == "polymarket"
    assert sample.data_tag["source"] == "polymarket"
    assert sample.data_tag["question_type"] == "market"


def test_forecastbench_preprocessor_short_sample_id_fits_filename_budget() -> None:
    """Even with a full-length polymarket condition_id, the sample id stays short."""

    pre = ForecastBenchPreprocessor()
    rec = _minimal_record()
    rec["id"] = "0x" + "ab" * 32  # 66-char polymarket-style hex id
    sample = pre.transform(rec)
    assert sample is not None
    # Format `fb:<10>:<<=4>:<<=10>_<8>` -> ~34 chars + small margin.
    assert len(sample.id.encode("utf-8")) <= 64
    # ``0x`` + first 8 ``ab`` characters = 10-char readable prefix.
    assert sample.id.startswith("fb:2025-03-02:pm:0xabababab_")
    # Full id stays available downstream.
    assert sample.metadata["question_id"] == rec["id"]


def test_forecastbench_preprocessor_unknown_source_falls_back_to_short_token() -> None:
    pre = ForecastBenchPreprocessor()
    rec = _minimal_record()
    rec["source"] = "metaculus"
    sample = pre.transform(rec)
    assert sample is not None
    # Unknown source -> first 4 chars of normalized name.
    assert sample.id.startswith("fb:2025-03-02:meta:")
    # Full original source preserved for downstream filtering / data_tag.
    assert sample.metadata["source"] == "metaculus"
    assert sample.data_tag["source"] == "metaculus"


def test_forecastbench_preprocessor_unknown_source_fallback_is_slugged() -> None:
    pre = ForecastBenchPreprocessor()
    rec = _minimal_record()
    rec["source"] = "my/source"
    sample = pre.transform(rec)
    assert sample is not None
    assert sample.id.startswith("fb:2025-03-02:myso:")
    assert "/" not in sample.id
    assert sample.metadata["source"] == "my/source"
    assert sample.data_tag["source"] == "my/source"


def test_forecastbench_preprocessor_non_date_question_set_keeps_id_stable() -> None:
    """Question sets without a date prefix still produce a deterministic short id."""

    pre = ForecastBenchPreprocessor()
    rec = _minimal_record()
    rec["question_set"] = "latest-llm.json"
    sample = pre.transform(rec)
    assert sample is not None
    assert sample.id.startswith("fb:latest-llm:pm:")
    assert sample.metadata["question_set"] == "latest-llm.json"


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
