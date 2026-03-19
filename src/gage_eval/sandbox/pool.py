"""Sandbox pool implementation for reuse and lifecycle control."""

from __future__ import annotations

import threading
import time
from typing import Any, Callable, Optional

from gage_eval.sandbox.base import BaseSandbox

# Builder signature: (**kwargs) -> BaseSandbox. The pool forwards acquire()
# keyword arguments so callers can thread trace or scope metadata into lazy
# runtime creation without baking that state into the pool itself.
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
        self._available: list[BaseSandbox] = []
        self._in_use: dict[int, BaseSandbox] = {}
        self._uses: dict[int, int] = {}
        self._last_used_at: dict[int, float] = {}
        self._creating = 0
        self._closed = False
        self._lock = threading.Lock()

    def acquire(self, **builder_kwargs: Any) -> BaseSandbox:
        """Acquire a sandbox instance from the pool."""

        discarded: list[BaseSandbox] = []

        # STEP 1: Reuse a live sandbox or reserve capacity for a new build.
        with self._lock:
            if self._closed:
                raise RuntimeError("sandbox pool is shut down")
            discarded.extend(self._drain_idle_unlocked())
            while self._available:
                candidate = self._available.pop()
                candidate_id = id(candidate)
                if self._is_idle_expired(candidate_id) or not _sandbox_is_healthy(
                    candidate
                ):
                    self._forget_unlocked(candidate)
                    discarded.append(candidate)
                    continue
                self._mark_acquired_unlocked(candidate)
                sandbox = candidate
                break
            else:
                sandbox = None
                if (
                    self._max_size is not None
                    and self._total_size_unlocked() >= self._max_size
                ):
                    exhausted = True
                else:
                    exhausted = False
                    self._creating += 1

        for stale in discarded:
            self._destroy_sandbox(stale)
        if sandbox is not None:
            return sandbox
        if exhausted:
            raise RuntimeError("sandbox pool exhausted")

        # STEP 2: Build new sandboxes outside the lock because startup can block.
        try:
            sandbox = self._builder(**builder_kwargs)
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

        if should_teardown:
            self._destroy_sandbox(sandbox)
        raise RuntimeError("sandbox pool is shut down")

    def release(self, sandbox: BaseSandbox) -> None:
        """Release a sandbox instance back into the pool."""

        discarded: list[BaseSandbox] = []
        sandbox_id = id(sandbox)

        # STEP 1: Reconcile the in-use set and decide whether this runtime can be reused.
        with self._lock:
            discarded.extend(self._drain_idle_unlocked())
            if sandbox_id not in self._in_use:
                reusable = None
            else:
                reusable = True
                self._in_use.pop(sandbox_id, None)
                if self._closed:
                    reusable = False
                elif (
                    self._max_uses is not None
                    and self._uses.get(sandbox_id, 0) >= self._max_uses
                ):
                    reusable = False
                elif self._max_size is not None and len(self._available) >= self._max_size:
                    reusable = False
            if reusable:
                self._last_used_at[sandbox_id] = time.monotonic()
                self._available.append(sandbox)
            elif reusable is False:
                self._forget_unlocked(sandbox)
                discarded.append(sandbox)

        # STEP 2: Tear down discarded sandboxes outside the lock.
        seen: set[int] = set()
        for stale in discarded:
            stale_id = id(stale)
            if stale_id in seen:
                continue
            seen.add(stale_id)
            self._destroy_sandbox(stale)

    def shutdown(self) -> None:
        """Shut down the pool and tear down tracked sandboxes."""

        # STEP 1: Mark the pool as closed and snapshot tracked sandboxes.
        with self._lock:
            self._closed = True
            sandboxes = list(self._available) + list(self._in_use.values())
            self._available.clear()
            self._in_use.clear()
            self._uses.clear()
            self._last_used_at.clear()

        # STEP 2: Tear down outside the lock so concurrent release calls can return.
        for sandbox in sandboxes:
            self._destroy_sandbox(sandbox)

    def _mark_acquired_unlocked(self, sandbox: BaseSandbox) -> None:
        sandbox_id = id(sandbox)
        self._in_use[sandbox_id] = sandbox
        self._last_used_at.pop(sandbox_id, None)
        self._uses[sandbox_id] = self._uses.get(sandbox_id, 0) + 1

    def _forget_unlocked(self, sandbox: BaseSandbox) -> None:
        sandbox_id = id(sandbox)
        self._in_use.pop(sandbox_id, None)
        self._uses.pop(sandbox_id, None)
        self._last_used_at.pop(sandbox_id, None)

    def _total_size_unlocked(self) -> int:
        return len(self._available) + len(self._in_use) + self._creating

    def _drain_idle_unlocked(self) -> list[BaseSandbox]:
        if self._idle_timeout_s is None or not self._available:
            return []
        alive: list[BaseSandbox] = []
        stale: list[BaseSandbox] = []
        for sandbox in self._available:
            if self._is_idle_expired(id(sandbox)):
                self._forget_unlocked(sandbox)
                stale.append(sandbox)
                continue
            alive.append(sandbox)
        self._available = alive
        return stale

    def _is_idle_expired(self, sandbox_id: int) -> bool:
        if self._idle_timeout_s is None:
            return False
        last_used_at = self._last_used_at.get(sandbox_id)
        if last_used_at is None:
            return False
        return (time.monotonic() - last_used_at) >= self._idle_timeout_s

    @staticmethod
    def _destroy_sandbox(sandbox: BaseSandbox) -> None:
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
