from __future__ import annotations

import pytest

from gage_eval.observability.log_sink import _should_skip_observability


@pytest.mark.fast
def test_log_sink_respects_skip_observability_flag() -> None:
    assert _should_skip_observability({"extra": {"skip_observability": True}}) is True
    assert _should_skip_observability({"extra": {"skip_observability": False}}) is False
    assert _should_skip_observability({"extra": {}}) is False
