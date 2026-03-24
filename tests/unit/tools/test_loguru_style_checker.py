from __future__ import annotations

from pathlib import Path

import pytest

from gage_eval.tools.loguru_style_checker import find_violations


@pytest.mark.fast
def test_loguru_style_checker_detects_percent_placeholders(tmp_path: Path) -> None:
    target = tmp_path / "bad_loguru.py"
    target.write_text(
        "from loguru import logger\nlogger.warning(\"bad %s\", 'x')\n",
        encoding="utf-8",
    )

    violations = find_violations([target])

    assert len(violations) == 1
    assert "brace-style formatting" in violations[0].message


@pytest.mark.fast
def test_loguru_style_checker_ignores_standard_logging_percent_style(tmp_path: Path) -> None:
    target = tmp_path / "standard_logging.py"
    target.write_text(
        "import logging\nlogger = logging.getLogger(__name__)\nlogger.warning(\"ok %s\", 'x')\n",
        encoding="utf-8",
    )

    violations = find_violations([target])

    assert violations == []


@pytest.mark.fast
def test_loguru_style_checker_allows_brace_style(tmp_path: Path) -> None:
    target = tmp_path / "good_loguru.py"
    target.write_text(
        "from loguru import logger\nlogger.warning(\"good {}\", 'x')\n",
        encoding="utf-8",
    )

    violations = find_violations([target])

    assert violations == []
