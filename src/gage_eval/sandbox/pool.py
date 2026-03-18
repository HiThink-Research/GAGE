"""Sandbox pool implementation for reuse and lifecycle control."""

from __future__ import annotations

import threading
from typing import Callable, Optional

from gage_eval.sandbox.base import BaseSandbox


class SandboxPool:
    """Simple pool to reuse sandbox runtimes."""

    def __init__(
        self,
        builder: Callable[[], BaseSandbox],
        *,
        max_size: Optional[int] = None,
        max_uses: Optional[int] = None,
    ) -> None:
        self._builder = builder
        self._max_size = max_size if max_size is None else max(1, int(max_size))
        self._max_uses = max_uses if max_uses is None else max(1, int(max_uses))
        self._available: list[BaseSandbox] = []
        self._in_use: dict[int, BaseSandbox] = {}
        self._uses: dict[int, int] = {}
        self._creating = 0
        self._closed = False
        self._lock = threading.Lock()

    def acquire(self) -> BaseSandbox:
        """Acquire a sandbox instance from the pool."""

        # STEP 1: Reuse an idle sandbox or reserve capacity for a new one.
        with self._lock:
            if self._closed:
                raise RuntimeError("sandbox pool is shut down")
            if self._available:
                sandbox = self._available.pop()
                self._mark_acquired_unlocked(sandbox)
                return sandbox
            if self._max_size is not None and self._total_size_unlocked() >= self._max_size:
                raise RuntimeError("sandbox pool exhausted")
            self._creating += 1

        # STEP 2: Build the sandbox outside the lock because startup can be slow.
        try:
            sandbox = self._builder()
        except Exception:
            with self._lock:
                self._creating -= 1
            raise

        # STEP 3: Publish the sandbox only if the pool is still open.
        should_teardown = False
        with self._lock:
            self._creating -= 1
            if self._closed:
                should_teardown = True
            else:
                self._mark_acquired_unlocked(sandbox)
                return sandbox

        sandbox.teardown()
        raise RuntimeError("sandbox pool is shut down")

    def release(self, sandbox: BaseSandbox) -> None:
        """Release a sandbox instance back into the pool."""

        sandbox_id = id(sandbox)
        should_teardown = False

        # STEP 1: Remove the sandbox from the in-use set under lock.
        with self._lock:
            if sandbox_id not in self._in_use:
                return
            self._in_use.pop(sandbox_id, None)

            if self._closed:
                should_teardown = True
            elif self._max_uses is not None and self._uses.get(sandbox_id, 0) >= self._max_uses:
                should_teardown = True
            elif self._max_size is not None and len(self._available) >= self._max_size:
                should_teardown = True
            else:
                self._available.append(sandbox)
                return

            self._uses.pop(sandbox_id, None)

        # STEP 2: Tear down outside the lock to keep pool operations responsive.
        if should_teardown:
            sandbox.teardown()

    def shutdown(self) -> None:
        """Shut down the pool and tear down tracked sandboxes."""

        # STEP 1: Mark the pool as closed and snapshot tracked sandboxes.
        with self._lock:
            self._closed = True
            sandboxes = list(self._available) + list(self._in_use.values())
            self._available.clear()
            self._in_use.clear()
            self._uses.clear()

        # STEP 2: Tear down outside the lock so callers can release safely afterward.
        for sandbox in sandboxes:
            sandbox.teardown()

    def _mark_acquired_unlocked(self, sandbox: BaseSandbox) -> None:
        sandbox_id = id(sandbox)
        self._in_use[sandbox_id] = sandbox
        self._uses[sandbox_id] = self._uses.get(sandbox_id, 0) + 1

    def _total_size_unlocked(self) -> int:
        return len(self._available) + len(self._in_use) + self._creating
