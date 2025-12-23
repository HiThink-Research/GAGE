from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import List

from gage_eval.support import agent_bridge
from gage_eval.support.agent_bridge import AgentResult, render_files_from_agent
from gage_eval.support.config import SupportConfig


def test_call_agent_uses_stdin_when_preferred(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run(cmd, **kwargs):  # type: ignore[no-untyped-def]
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return SimpleNamespace(stdout="ok", stderr="", returncode=0)

    monkeypatch.setattr(agent_bridge.subprocess, "run", _fake_run)

    cfg = SupportConfig()
    cfg.agent.type = "gemini"
    cfg.agent.command = "gemini"
    cfg.agent.yolo_args = []

    agent_bridge.call_agent("hello", cfg, prefer_stdin=True)
    assert captured["input"] == "hello"


def test_render_files_retries_with_stdin_for_gemini(monkeypatch) -> None:
    calls: List[bool] = []

    def _fake_call(prompt: str, cfg: SupportConfig, prefer_stdin: bool = False) -> AgentResult:
        calls.append(prefer_stdin)
        if prefer_stdin:
            stdout = "### FILE: foo.txt\ncontent\n### END\n"
        else:
            stdout = "random text without protocol"
        return AgentResult(stdout=stdout, stderr="", returncode=0)

    monkeypatch.setattr(agent_bridge, "call_agent", _fake_call)

    cfg = SupportConfig()
    cfg.agent.type = "gemini"
    cfg.agent.command = "gemini"
    cfg.agent.yolo_args = []

    files = render_files_from_agent("p", cfg)
    assert calls == [False, True]
    assert files == {Path("foo.txt"): "content\n"}
