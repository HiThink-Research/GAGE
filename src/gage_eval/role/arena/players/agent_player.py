"""Agent-backed arena player implementation."""

from __future__ import annotations

from gage_eval.role.arena.players.llm_player import LLMPlayer


class AgentPlayer(LLMPlayer):
    """Arena player that delegates decisions to an agent adapter."""

    pass
