"""Utility helpers shared across registry-aware components."""

from __future__ import annotations

import asyncio
import functools
import inspect
from typing import Any, Awaitable, Callable, TypeVar

T = TypeVar("T")


def ensure_async(func: Callable[..., T | Awaitable[T]]) -> Callable[..., Awaitable[T]]:
    """Wrap synchronous callables into awaitables."""

    if inspect.iscoroutinefunction(func):
        return func  # type: ignore[return-value]

    @functools.wraps(func)
    async def _async_wrapper(*args: Any, **kwargs: Any) -> T:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, functools.partial(func, *args, **kwargs))

    return _async_wrapper


def run_sync(awaitable: Awaitable[T]) -> T:
    """Execute an awaitable from sync code, reusing the running loop when possible."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(awaitable)
    raise RuntimeError("run_sync() cannot be used inside an active event loop")
