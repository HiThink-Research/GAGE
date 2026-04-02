from __future__ import annotations

from gage_eval.agent_eval_kits.appworld.kit import build_kit as build_appworld_kit
from gage_eval.agent_eval_kits.tau2.kit import build_kit as build_tau2_kit


def test_appworld_kit_importable() -> None:
    assert callable(build_appworld_kit)


def test_appworld_build_kit_returns_definition() -> None:
    kit = build_appworld_kit()

    assert kit.kit_id == "appworld"
    assert kit.verifier_kind == "judge_adapter"


def test_tau2_kit_importable() -> None:
    assert callable(build_tau2_kit)


def test_tau2_build_kit_returns_definition() -> None:
    kit = build_tau2_kit()

    assert kit.kit_id == "tau2"
    assert kit.verifier_kind == "judge_adapter"
