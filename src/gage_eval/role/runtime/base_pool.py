"""Base pooling/scheduler interfaces."""

from __future__ import annotations

from typing import ContextManager, Optional, Protocol

from gage_eval.role.role_instance import Role


class BasePool(Protocol):
    """Minimum interface for role pools."""

    def acquire(self, timeout: Optional[float] = None) -> ContextManager[Role]:
        ...

    def release(self, role: Role) -> None:
        ...

    def shutdown(self) -> None:
        ...


class BaseScheduler(Protocol):
    """Pool selection/notification abstraction."""

    def select_pool(self, adapter_id: str) -> BasePool:
        ...

    def report_event(self, event: dict) -> None:
        ...
