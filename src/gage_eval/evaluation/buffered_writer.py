"""Buffered writer used to batch auto-eval samples to disk."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import threading
import time
from pathlib import Path
from typing import Iterable, List, Sequence


class BufferedResultWriter:
    """Accumulates JSON payloads in memory and flushes them in batches."""

    def __init__(
        self,
        target: Path,
        *,
        max_batch_size: int = 64,
        flush_interval_s: float = 2.0,
        pending_suffix: str = ".pending",
    ) -> None:
        self._target = Path(target)
        self._pending = self._target.with_suffix(self._target.suffix + pending_suffix)
        self._max_batch_size = max(1, max_batch_size)
        self._flush_interval = max(0.1, flush_interval_s)
        self._buffer: List[str] = []
        self._lock = threading.Lock()
        self._last_flush = time.perf_counter()
        self._closed = False
        self._flush_count = 0
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._recover_pending()
        atexit.register(self.close)

    @property
    def target_path(self) -> Path:
        return self._target

    @property
    def flush_count(self) -> int:
        return self._flush_count

    def record(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        entry = f"{line}\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("BufferedResultWriter already closed")
            self._buffer.append(entry)
            if self._should_flush_locked():
                self._flush_locked()

    def flush(self) -> None:
        with self._lock:
            self._flush_locked(force=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked(force=True)
            self._closed = True

    def _should_flush_locked(self) -> bool:
        if len(self._buffer) >= self._max_batch_size:
            return True
        elapsed = time.perf_counter() - self._last_flush
        return elapsed >= self._flush_interval

    def _flush_locked(self, force: bool = False) -> None:
        if not self._buffer:
            return
        if not force and not self._should_flush_locked():
            return
        to_flush: Sequence[str] = list(self._buffer)
        self._buffer.clear()
        self._last_flush = time.perf_counter()
        try:
            self._write_lines(to_flush)
        except Exception:
            # Restore buffered payloads so callers can retry.
            self._buffer[0:0] = list(to_flush)
            raise
        else:
            self._flush_count += 1

    def _write_lines(self, entries: Iterable[str]) -> None:
        joined = "".join(entries)
        if not joined:
            return
        self._write_pending(joined)
        with self._target.open("a", encoding="utf-8") as dest:
            dest.write(joined)
            dest.flush()
            os.fsync(dest.fileno())
        if self._pending.exists():
            self._pending.unlink()

    def _write_pending(self, data: str) -> None:
        with self._pending.open("w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())

    def _recover_pending(self) -> None:
        if not self._pending.exists():
            return
        try:
            with self._pending.open("r", encoding="utf-8") as handle, self._target.open(
                "a", encoding="utf-8"
            ) as dest:
                shutil.copyfileobj(handle, dest)
            self._pending.unlink()
        except Exception:
            # Leave the pending file in place for manual inspection.
            pass
