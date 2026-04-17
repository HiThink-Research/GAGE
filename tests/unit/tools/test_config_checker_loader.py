from __future__ import annotations

from pathlib import Path

import pytest

from gage_eval.tools.config_checker import validate_config

ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.io
def test_config_checker_accepts_short_static_config() -> None:
    path = ROOT / "tests" / "fixtures" / "static_eval" / "aime24_short.yaml"

    validate_config(path, materialize_runtime=False)
