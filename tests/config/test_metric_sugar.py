import pytest

from gage_eval.config.pipeline_config import _normalize_metric_entry


def test_normalize_metric_entry_string():
    entry = _normalize_metric_entry("exact_match")
    assert entry["metric_id"] == "exact_match"
    assert entry["implementation"] == "exact_match"
    assert entry["aggregation"] is None
    assert entry["params"] == {}


def test_normalize_metric_entry_kv_shortcut():
    entry = _normalize_metric_entry({"regex_match": {"pattern": "\\d+", "aggregation": "mean"}})
    assert entry["metric_id"] == "regex_match"
    assert entry["implementation"] == "regex_match"
    assert entry["aggregation"] == "mean"
    assert entry["params"]["pattern"] == "\\d+"


def test_normalize_metric_entry_full_dict():
    payload = {
        "metric_id": "custom",
        "implementation": "pkg.mod:CustomMetric",
        "aggregation": "identity",
        "params": {"k": 1},
    }
    entry = _normalize_metric_entry(payload)
    assert entry == payload


def test_normalize_metric_entry_invalid():
    with pytest.raises(ValueError):
        _normalize_metric_entry(123)
