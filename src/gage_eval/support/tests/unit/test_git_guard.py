from __future__ import annotations

from types import SimpleNamespace

import pytest

from gage_eval.support.utils import ensure_git_clean


def test_git_guard_blocks_dirty(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=" M foo.py\n", stderr="")

    monkeypatch.setattr("gage_eval.support.utils.subprocess.run", fake_run)
    with pytest.raises(RuntimeError):
        ensure_git_clean(force=False)


def test_git_guard_skips_when_force(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(*args, **kwargs):
        return SimpleNamespace(returncode=0, stdout=" M foo.py\n", stderr="")

    monkeypatch.setattr("gage_eval.support.utils.subprocess.run", fake_run)
    ensure_git_clean(force=True)

