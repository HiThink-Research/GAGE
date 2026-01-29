from __future__ import annotations

from typing import Any
from unittest.mock import Mock

from gage_eval.role.arena.games.retro.temporal.data_contract_v15 import (
    build_action_dict,
    build_observation_dict,
    build_result_dict,
)


def _build_mock_mapping(data: dict[str, Any]) -> Mock:
    mapping = Mock()
    mapping.__iter__.return_value = iter(data)
    mapping.__len__.return_value = len(data)
    mapping.__getitem__.side_effect = data.__getitem__
    return mapping


def test_temporal_observation_dict_allows_extra_override() -> None:
    info_data = {"score": 5}
    info = _build_mock_mapping(info_data)
    payload = build_observation_dict(
        view_text="state",
        legal_actions=["noop"],
        active_player="player_0",
        tick=2,
        step=3,
        info=info,
        extra={"info": {"override": True}, "note": "ok"},
    )

    assert payload["extra"]["info"] == {"override": True}
    assert payload["extra"]["note"] == "ok"
    assert info.__iter__.called


def test_temporal_action_dict_omits_hold_ticks_when_none() -> None:
    payload = build_action_dict(player="p0", move="noop", raw="noop", hold_ticks=None)
    assert "hold_ticks" not in payload


def test_temporal_result_dict_handles_mocked_mappings() -> None:
    scores = _build_mock_mapping({"p0": 1.25})
    metrics = _build_mock_mapping({"energy": 42})
    payload = build_result_dict(
        status="win",
        reason=None,
        winner="p0",
        replay_path="replay.json",
        scores=scores,
        metrics=metrics,
    )

    assert payload["reason"] == ""
    assert payload["scores"]["p0"] == 1.25
    assert payload["metrics"]["energy"] == 42
    assert scores.__iter__.called
    assert metrics.__iter__.called


def test_temporal_result_dict_skips_empty_scores_and_metrics() -> None:
    payload = build_result_dict(
        status="draw",
        reason="timeout",
        winner=None,
        replay_path=None,
        scores={},
        metrics={},
    )
    assert "scores" not in payload
    assert "metrics" not in payload
