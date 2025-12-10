import pytest

from gage_eval.config.pipeline_config import _normalize_metric_entry


def test_fnstyle_metric_parses_params_and_aggregation():
    entry = _normalize_metric_entry("regex_match(pattern=\\d+,aggregation=mean,ignore_case=false)")
    assert entry["metric_id"] == "regex_match"
    assert entry["implementation"] == "regex_match"
    assert entry["aggregation"] == "mean"
    assert entry["params"]["pattern"] == "\\d+"
    assert entry["params"]["ignore_case"] is False


def test_fnstyle_metric_supports_numbers_and_quotes():
    entry = _normalize_metric_entry("numeric_match(tolerance=0.1,label_field='gold')")
    assert entry["params"]["tolerance"] == 0.1
    assert entry["params"]["label_field"] == "gold"


def test_fnstyle_metric_invalid_token_raises():
    with pytest.raises(ValueError):
        _normalize_metric_entry("exact_match(badtoken)")


def test_fnstyle_metric_missing_name_raises():
    with pytest.raises(ValueError):
        _normalize_metric_entry("(a=1)")
