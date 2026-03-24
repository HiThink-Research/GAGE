"""Root journal writers for EvalCache sample persistence."""

from __future__ import annotations

import os
import threading
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Mapping

JsonLineSerializer = Callable[[Mapping[str, Any]], str]


class RunSampleJournal(ABC):
    """Persists root sample journal records for a single run."""

    def __init__(self, target: Path) -> None:
        self._target = Path(target)

    @property
    def target_path(self) -> Path:
        """Returns the root journal target path."""

        return self._target

    @abstractmethod
    def append(self, payload: Mapping[str, Any]) -> None:
        """Appends one payload to the journal."""

    @abstractmethod
    def flush(self) -> None:
        """Flushes any buffered payloads."""

    @abstractmethod
    def close(self) -> None:
        """Closes the journal."""


class LockedJsonlJournal(RunSampleJournal):
    """Writes one JSONL record at a time under a process-local mutex."""

    def __init__(
        self,
        target: Path,
        *,
        serializer: JsonLineSerializer,
        fsync_enabled: bool = True,
    ) -> None:
        super().__init__(target)
        self._serializer = serializer
        self._fsync_enabled = fsync_enabled
        self._lock = threading.Lock()
        self._closed = False
        self._target.parent.mkdir(parents=True, exist_ok=True)

    def append(self, payload: Mapping[str, Any]) -> None:
        entry = self._serializer(payload)
        if not entry.endswith("\n"):
            entry = f"{entry}\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("LockedJsonlJournal already closed")
            with self._target.open("a", encoding="utf-8") as handle:
                handle.write(entry)
                handle.flush()
                if self._fsync_enabled:
                    os.fsync(handle.fileno())

    def flush(self) -> None:
        # Each append is fully flushed before the file handle is closed.
        return

    def close(self) -> None:
        with self._lock:
            self._closed = True
