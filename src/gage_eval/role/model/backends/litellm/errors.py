"""LiteLLM backend error classification helpers."""

from __future__ import annotations

from typing import Any


DEPENDENCY_UNAVAILABLE = "dependency_unavailable"
INVALID_MEDIA_PAYLOAD = "invalid_media_payload"
INVALID_REQUEST = "invalid_request"
MEDIA_PAYLOAD_TOO_LARGE = "media_payload_too_large"
UNSUPPORTED_CAPABILITY = "unsupported_capability"


class LiteLLMBackendError(RuntimeError):
    """Backend error with a stable machine-readable type."""

    def __init__(self, message: str, error_type: str) -> None:
        self.message = message
        self.error_type = error_type
        super().__init__(message)

    def __str__(self) -> str:
        return f"{self.message} (error_type={self.error_type})"


def status_code_from_error(exc: Exception) -> int | None:
    """Extract an HTTP status code from common LiteLLM/provider exception shapes."""

    for attr in ("status_code", "status"):
        value = getattr(exc, attr, None)
        code = _coerce_status(value)
        if code is not None:
            return code

    response = getattr(exc, "response", None)
    if response is not None:
        for attr in ("status_code", "status"):
            value = getattr(response, attr, None)
            code = _coerce_status(value)
            if code is not None:
                return code
        if isinstance(response, dict):
            code = _coerce_status(response.get("status_code") or response.get("status"))
            if code is not None:
                return code

    return None


def _coerce_status(value: Any) -> int | None:
    try:
        code = int(value)
    except (TypeError, ValueError):
        return None
    return code if 100 <= code <= 599 else None
