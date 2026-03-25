"""Package bootstrap intentionally avoids registry auto-discovery."""

from __future__ import annotations

from gage_eval._loguru_compat import ensure_loguru

ensure_loguru()

__all__ = []
