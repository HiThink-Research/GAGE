"""Simple token-bucket rate limiter used by Role pools."""

from __future__ import annotations

import threading
import time
from typing import Optional


class RateLimiter:
    def __init__(self, capacity: int, interval: float) -> None:
        if capacity <= 0:
            raise ValueError("RateLimiter capacity must be > 0")
        if interval <= 0:
            raise ValueError("RateLimiter interval must be > 0")
        self.capacity = float(capacity)
        self.tokens = float(capacity)
        self.interval = float(interval)
        self._lock = threading.Lock()
        self._updated = time.monotonic()

    def acquire(self, timeout: Optional[float] = None) -> None:
        deadline = None if timeout is None else time.monotonic() + timeout
        while True:
            wait = self._try_consume()
            if wait is None:
                return
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("RateLimiter timed out")
                wait = min(wait, remaining)
            time.sleep(wait)

    def _try_consume(self) -> Optional[float]:
        now = time.monotonic()
        with self._lock:
            elapsed = now - self._updated
            if elapsed > 0:
                self.tokens = min(self.capacity, self.tokens + elapsed * (self.capacity / self.interval))
                self._updated = now
            if self.tokens >= 1.0:
                self.tokens -= 1.0
                return None
            missing = 1.0 - self.tokens
            wait = (missing / self.capacity) * self.interval
            return max(wait, 0.001)
