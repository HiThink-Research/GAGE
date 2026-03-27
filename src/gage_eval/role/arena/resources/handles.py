from __future__ import annotations

from typing import Protocol


class RuntimeHandle(Protocol):
    def close(self) -> None: ...

    def terminate(self) -> None: ...

    def reap(self) -> None: ...
