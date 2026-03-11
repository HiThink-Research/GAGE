"""Sandbox pool implementation for reuse and lifecycle control."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Dict, List, Optional

from gage_eval.sandbox.base import BaseSandbox

# Builder signature: (**kwargs) -> BaseSandbox.  The pool forwards any
# keyword arguments from acquire() so callers can pass trace context
# without baking it into the builder closure at pool-creation time.
BuilderFn = Callable[..., BaseSandbox]


class SandboxPool:
    """Thread-safe pool to reuse sandbox runtimes."""

    def __init__(
        self,
        builder: BuilderFn,
        *,
        max_size: Optional[int] = None,
        max_uses: Optional[int] = None,
        idle_timeout_s: Optional[float] = None,
    ) -> None:
        self._builder = builder
        self._max_size = max_size if max_size is None else max(1, int(max_size))
        self._max_uses = max_uses if max_uses is None else max(1, int(max_uses))
        self._idle_timeout_s = (
            None if idle_timeout_s is None else max(0.0, float(idle_timeout_s))
        )
        self._available: List[BaseSandbox] = []
        self._in_use: Dict[int, BaseSandbox] = {}
        self._uses: Dict[int, int] = {}
        self._last_used_at: Dict[int, float] = {}
        self._lock = threading.Lock()

    def acquire(self, **builder_kwargs: Any) -> BaseSandbox:
        with self._lock:
            self._cleanup_idle()
            sandbox: Optional[BaseSandbox] = None
            while self._available:
                candidate = self._available.pop()
                candidate_id = id(candidate)
                if self._is_idle_expired(candidate_id):
                    self._destroy(candidate)
                    continue
                if _sandbox_is_healthy(candidate):
                    sandbox = candidate
                    break
                self._destroy(candidate)
            if sandbox is None:
                if self._max_size is not None and self._total_size() >= self._max_size:
                    raise RuntimeError("sandbox pool exhausted")
                sandbox = self._builder(**builder_kwargs)
            sandbox_id = id(sandbox)
            self._in_use[sandbox_id] = sandbox
            self._last_used_at.pop(sandbox_id, None)
            self._uses[sandbox_id] = self._uses.get(sandbox_id, 0) + 1
            return sandbox

    def release(self, sandbox: BaseSandbox) -> None:
        with self._lock:
            self._cleanup_idle()
            sandbox_id = id(sandbox)
            if sandbox_id not in self._in_use:
                return
            self._in_use.pop(sandbox_id, None)
            if (
                self._max_uses is not None
                and self._uses.get(sandbox_id, 0) >= self._max_uses
            ):
                self._destroy(sandbox)
                return
            if self._max_size is not None and len(self._available) >= self._max_size:
                self._destroy(sandbox)
                return
            self._last_used_at[sandbox_id] = time.monotonic()
            self._available.append(sandbox)

    def shutdown(self) -> None:
        with self._lock:
            for sandbox in list(self._available):
                self._destroy(sandbox)
            for sandbox in list(self._in_use.values()):
                self._destroy(sandbox)
            self._available.clear()
            self._in_use.clear()
            self._uses.clear()
            self._last_used_at.clear()

    def _total_size(self) -> int:
        return len(self._available) + len(self._in_use)

    def _cleanup_idle(self) -> None:
        if self._idle_timeout_s is None:
            return
        alive: List[BaseSandbox] = []
        for sandbox in self._available:
            sandbox_id = id(sandbox)
            if self._is_idle_expired(sandbox_id):
                self._destroy(sandbox)
                continue
            alive.append(sandbox)
        self._available = alive

    def _is_idle_expired(self, sandbox_id: int) -> bool:
        if self._idle_timeout_s is None:
            return False
        last_used_at = self._last_used_at.get(sandbox_id)
        if last_used_at is None:
            return False
        return (time.monotonic() - last_used_at) >= self._idle_timeout_s

    def _destroy(self, sandbox: BaseSandbox) -> None:
        sandbox_id = id(sandbox)
        self._in_use.pop(sandbox_id, None)
        self._uses.pop(sandbox_id, None)
        self._last_used_at.pop(sandbox_id, None)
        try:
            sandbox.teardown()
        except Exception:
            pass


def _sandbox_is_healthy(sandbox: BaseSandbox) -> bool:
    checker = getattr(sandbox, "is_alive", None)
    if not callable(checker):
        return True
    try:
        return bool(checker())
    except Exception:
        return False
