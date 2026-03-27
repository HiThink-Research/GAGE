"""Retro-scoped compatibility hook for the new player-driver runtime."""

from __future__ import annotations

def ensure_retro_human_queue_bridge() -> None:
    """Retained as a no-op hook after removing the legacy human queue bridge."""
    return None


__all__ = ["ensure_retro_human_queue_bridge"]
