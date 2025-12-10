import pytest

from gage_eval.metrics.utils import (
    normalize_text_advanced,
    ensure_list_of_strings,
    strip_thought_tags,
    flatten_numeric_list,
)


@pytest.mark.parametrize(
    "raw,kwargs,expected",
    [
        ("  Hello World  ", {"case_sensitive": False, "strip": True}, "hello world"),
        ("A   B", {"collapse_whitespace": True}, "a b"),
        ("MixedCase", {"case_sensitive": True}, "MixedCase"),
        (None, {}, None),
    ],
)
def test_normalize_text_advanced(raw, kwargs, expected):
    assert normalize_text_advanced(raw, **kwargs) == expected


def test_ensure_list_of_strings():
    assert ensure_list_of_strings("a") == ["a"]
    assert ensure_list_of_strings(["a", None]) == ["a"]
    assert ensure_list_of_strings(["a", None], ignore_none=False) == ["a", "None"]
    assert ensure_list_of_strings(None) == []
    assert ensure_list_of_strings(123) == ["123"]


def test_strip_thought_tags():
    text = "<think>reasoning</think>Answer"
    assert strip_thought_tags(text) == "Answer"
    assert strip_thought_tags(None) == ""


def test_flatten_numeric_list():
    nested = [[-1.0, -2.0], [-3], None, "4"]
    assert flatten_numeric_list(nested) == [-1.0, -2.0, -3.0, 4.0]
    assert flatten_numeric_list(None) == []
