"""Thread-safe role pool with borrow/return semantics."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import AbstractContextManager
from typing import Any, Callable, Deque, Optional

from gage_eval.role.role_instance import Role
from gage_eval.role.runtime.base_pool import BasePool


class RolePool(BasePool):
    """Pool that manages reusable :class:`Role` objects."""

    def __init__(self, adapter_id: str, builder: Callable[[], Role], max_size: Optional[int] = None) -> None:
        self.adapter_id = adapter_id
        self._builder = builder
        self._max_size = max_size if (max_size and max_size > 0) else None
        self._available: Deque[Role] = deque()
        self._in_use: set[int] = set()
        self._created = 0
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._closed = False

    def acquire(self, timeout: Optional[float] = None) -> "RoleLease":
        """Borrow a role from the pool."""

        role = self._take_or_wait(timeout)
        return RoleLease(self, role)

    def release(self, role: Role) -> None:
        """Return a role to the pool."""

        if role is None:
            return
        with self._condition:
            if self._closed:
                return
            role_id = id(role)
            self._in_use.discard(role_id)
            self._available.append(role)
            self._condition.notify()

    def shutdown(self) -> None:
        """Mark pool as closed and drop cached roles."""

        with self._condition:
            self._closed = True
            self._available.clear()
            self._in_use.clear()
            self._condition.notify_all()

    def snapshot(self) -> dict[str, Any]:
        """Return a lightweight runtime snapshot for diagnostics."""

        with self._condition:
            in_use = len(self._in_use)
            capacity = self._max_size
            available = None if capacity is None else max(0, capacity - in_use)
            if self._closed:
                available = 0 if capacity is not None else None
            return {
                "pool_type": "role_pool",
                "adapter_id": self.adapter_id,
                "capacity": capacity,
                "in_use": in_use,
                "available": available,
                "created": self._created,
                "healthy": not self._closed,
                "closed": self._closed,
                "extensions": {},
            }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _take_or_wait(self, timeout: Optional[float]) -> Role:
        with self._condition:
            lease = self._try_take_unlocked()
            if lease is not None:
                return lease

            deadline = None if timeout is None else time.monotonic() + timeout
            while True:
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise TimeoutError(f"Timed out acquiring role from pool '{self.adapter_id}'")
                self._condition.wait(timeout=remaining)
                lease = self._try_take_unlocked()
                if lease is not None:
                    return lease

    def _try_take_unlocked(self) -> Optional[Role]:
        if self._closed:
            raise RuntimeError(f"RolePool '{self.adapter_id}' is shut down")

        if self._available:
            role = self._available.popleft()
            self._in_use.add(id(role))
            return role

        if self._max_size is None or self._created < self._max_size:
            role = self._builder()
            self._created += 1
            self._in_use.add(id(role))
            return role
        return None


class RoleLease(AbstractContextManager):
    """Context manager wrapping a pooled Role."""

    def __init__(self, pool: RolePool, role: Role) -> None:
        self._pool = pool
        self._role = role
        self._released = False

    def __enter__(self) -> Role:
        return self._role

    def __exit__(self, exc_type, exc, tb) -> bool:
        self._release()
        return False

    def _release(self) -> None:
        if not self._released:
            self._pool.release(self._role)
            self._released = True
