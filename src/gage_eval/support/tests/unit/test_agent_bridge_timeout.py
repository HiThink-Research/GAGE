from __future__ import annotations

from gage_eval.support.agent_bridge import call_agent
from gage_eval.support.config import SupportConfig


def test_call_agent_handles_missing_command() -> None:
    cfg = SupportConfig()
    cfg.agent.command = "__nonexistent_agent__"
    result = call_agent("hi", cfg)
    assert result.returncode == 127

