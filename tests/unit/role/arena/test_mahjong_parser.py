from __future__ import annotations

from gage_eval.role.arena.games.mahjong.parsers.mahjong import StandardMahjongParser


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
