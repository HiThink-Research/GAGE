from __future__ import annotations

import pytest

from gage_eval.agent_eval_kits.appworld.units import (
    build_appworld_instruction,
    build_appworld_messages,
)


@pytest.mark.fast
def test_appworld_instruction_enforces_tool_use_contract() -> None:
    sample = {"instruction": "Find the most-liked song in my Spotify playlists."}

    instruction = build_appworld_instruction(sample)
    messages = build_appworld_messages(sample)

    assert "Use the provided AppWorld tools" in instruction
    assert messages[0]["role"] == "system"
    assert "Do not answer from prior knowledge" in str(messages[0]["content"])
