"""Sandbox pool implementation for reuse and lifecycle control."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

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
        self._available: List[BaseSandbox] = []
        self._in_use: Dict[int, BaseSandbox] = {}
        self._uses: Dict[int, int] = {}

    def acquire(self) -> BaseSandbox:
        if self._available:
            sandbox = self._available.pop()
        else:
            if self._max_size is not None and self._total_size() >= self._max_size:
                raise RuntimeError("sandbox pool exhausted")
            sandbox = self._builder()
        self._in_use[id(sandbox)] = sandbox
        self._uses[id(sandbox)] = self._uses.get(id(sandbox), 0) + 1
        return sandbox

    def release(self, sandbox: BaseSandbox) -> None:
        sandbox_id = id(sandbox)
        if sandbox_id not in self._in_use:
            return
        self._in_use.pop(sandbox_id, None)
        if self._max_uses is not None and self._uses.get(sandbox_id, 0) >= self._max_uses:
            sandbox.teardown()
            self._uses.pop(sandbox_id, None)
            return
        if self._max_size is not None and len(self._available) >= self._max_size:
            sandbox.teardown()
            self._uses.pop(sandbox_id, None)
            return
        self._available.append(sandbox)

    def shutdown(self) -> None:
        for sandbox in self._available:
            sandbox.teardown()
        for sandbox in self._in_use.values():
            sandbox.teardown()
        self._available.clear()
        self._in_use.clear()
        self._uses.clear()

    def _total_size(self) -> int:
        return len(self._available) + len(self._in_use)
