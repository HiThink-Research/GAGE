from __future__ import annotations

import pytest

from gage_eval.support.config import SupportConfig
from gage_eval.support.utils import guard_commands


def test_guard_rejects_injection() -> None:
    cfg = SupportConfig()
    with pytest.raises(RuntimeError):
        guard_commands(["python -c 'print(1)'; rm -rf /"], cfg, confirm=False)


def test_guard_allowlist_filters() -> None:
    cfg = SupportConfig()
    cfg.execution.command_allowlist = ["python"]
    cmds = guard_commands(["python -c 'print(1)'", "bash -lc echo hi"], cfg, confirm=False)
    assert cmds == ["python -c 'print(1)'"]

