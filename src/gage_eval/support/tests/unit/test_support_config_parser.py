from __future__ import annotations

import pytest

from gage_eval.support.utils import parse_support_config


def test_parse_support_config_success() -> None:
    md = """
hello
```yaml support_config
dataset_id: d1
preprocess_name: p1
fields:
  question_field: q
  answers_field: a
```
"""
    cfg = parse_support_config(md)
    assert cfg["dataset_id"] == "d1"


def test_parse_support_config_requires_unique_block() -> None:
    md = "```yaml support_config\ndataset_id: d\npreprocess_name: p\nfields: {}\n```\n" * 2
    with pytest.raises(ValueError):
        parse_support_config(md)

