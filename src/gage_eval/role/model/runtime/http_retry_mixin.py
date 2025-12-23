"""Simple HTTP retry mixin shared by remote backends."""

from __future__ import annotations

import asyncio
from typing import Any, Dict

from loguru import logger


class HttpRetryMixin:
    """Adds exponential backoff retries to ``_ainvoke_backend`` implementations."""

    def __init__(self, *args, http_retry_attempts: int = 3, http_retry_interval: float = 1.0, **kwargs) -> None:
        self._http_retry_attempts = max(1, int(http_retry_attempts))
        self._http_retry_interval = float(http_retry_interval)
        super().__init__(*args, **kwargs)  # type: ignore[misc]

    async def _ainvoke_with_retry(self, request: Dict[str, Any], caller=None) -> Dict[str, Any]:
        last_exc: Exception | None = None
        for attempt in range(1, self._http_retry_attempts + 1):
            try:
                if caller:
                    return await caller(request)
                return await self._ainvoke_backend(request)
            except Exception as exc:  # pragma: no cover - network failures are nondeterministic
                last_exc = exc
                logger.warning(
                    "HTTP backend attempt {}/{} failed: {}",
                    attempt,
                    self._http_retry_attempts,
                    exc,
                )
                if attempt >= self._http_retry_attempts:
                    break
                await asyncio.sleep(self._http_retry_interval * attempt)
        assert last_exc is not None
        raise last_exc
