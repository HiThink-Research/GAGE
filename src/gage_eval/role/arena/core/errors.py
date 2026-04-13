from __future__ import annotations


class GameArenaError(RuntimeError):
    """Base error for game arena core orchestration."""


class InvalidPlayerBindingError(GameArenaError, ValueError):
    """Raised when a GameKit player binding cannot be normalized."""


class PlayerDriverLookupError(GameArenaError, KeyError):
    """Raised when a requested player driver is not registered."""


class PlayerExecutionUnavailableError(GameArenaError):
    """Raised when a bound player cannot execute in the current runtime."""


class PlayerStopRequested(GameArenaError):
    """Raised when a bound player is interrupted by a session stop request."""
