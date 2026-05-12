"""Stable error helpers for external harness integration."""

from __future__ import annotations


class ExternalHarnessError(ValueError):
    """Exception carrying an Appendix A external-harness error code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        super().__init__(f"{code}: {message}")


class ExternalHarnessParseError(ExternalHarnessError):
    """Parse-time failure while importing external harness output."""


__all__ = ["ExternalHarnessError", "ExternalHarnessParseError"]
