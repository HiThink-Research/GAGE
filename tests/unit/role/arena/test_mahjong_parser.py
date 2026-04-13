from __future__ import annotations

from gage_eval.game_kits.phase_card_game.mahjong.kit import build_mahjong_game_kit
from gage_eval.game_kits.phase_card_game.mahjong.parsers.mahjong import (
    StandardMahjongParser,
)


def test_mahjong_parser_build_rethink_prompt_keeps_json_schema_literal() -> None:
    parser = StandardMahjongParser()

    prompt = parser.build_rethink_prompt(
        last_output="bad output",
        reason="illegal_move",
        legal_moves=["B1", "C2"],
    )

    assert '{"action": "<action>", "chat": "<short line>"}' in prompt
    assert "Legal moves: B1, C2." in prompt


def test_mahjong_parser_parses_json_action_with_chat() -> None:
    parser = StandardMahjongParser()

    result = parser.parse(
        '{"action":"B1","chat":"tile talk"}',
        legal_moves=["B1", "C2"],
    )

    assert result.action_text == "B1"
    assert result.chat_text == "tile talk"
    assert result.error is None


def test_mahjong_gamekit_owns_parser_and_renderer_refs() -> None:
    game_kit = build_mahjong_game_kit()

    assert game_kit.parser == "mahjong_v1"
    assert game_kit.renderer == "mahjong_replay_v1"
    assert [env_spec.parser for env_spec in game_kit.env_catalog] == [
        "mahjong_v1",
        "mahjong_v1",
    ]
    assert [env_spec.renderer for env_spec in game_kit.env_catalog] == [
        "mahjong_replay_v1",
        "mahjong_replay_v1",
    ]
