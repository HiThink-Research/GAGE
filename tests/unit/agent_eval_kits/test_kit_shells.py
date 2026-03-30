from __future__ import annotations

import pytest

from gage_eval.agent_eval_kits.appworld.kit import build_kit as build_appworld_kit
from gage_eval.agent_eval_kits.tau2.kit import build_kit as build_tau2_kit


@pytest.mark.fast
def test_appworld_kit_importable() -> None:
    assert callable(build_appworld_kit)


@pytest.mark.fast
def test_appworld_build_kit_raises() -> None:
    with pytest.raises(NotImplementedError):
        build_appworld_kit()


@pytest.mark.fast
def test_tau2_kit_importable() -> None:
    assert callable(build_tau2_kit)


@pytest.mark.fast
def test_tau2_build_kit_raises() -> None:
    with pytest.raises(NotImplementedError):
        build_tau2_kit()
