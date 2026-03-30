"""Buffered writer used to batch auto-eval samples to disk."""

from __future__ import annotations

import atexit
import json
import os
import shutil
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence


@dataclass(frozen=True, slots=True)
class BufferedWriterStats:
    """Observable stats for one buffered result writer."""

    flush_count: int
    fsync_count: int
    durability_policy: str
    buffered_entries: int


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
        self._last_fsync = self._last_flush
        self._closed = False
        self._flush_count = 0
        self._fsync_count = 0
        self._durability_policy = _resolve_durability_policy(
            os.environ.get("GAGE_EVAL_BUFFER_DURABILITY_POLICY", "interval")
        )
        self._fsync_every_flushes = max(1, _env_int("GAGE_EVAL_BUFFER_FSYNC_EVERY_FLUSHES", 8))
        self._fsync_every_s = max(0.1, _env_float("GAGE_EVAL_BUFFER_FSYNC_EVERY_S", 3.0))
        self._last_fsync_flush_count = 0
        self._target.parent.mkdir(parents=True, exist_ok=True)
        self._recover_pending()
        atexit.register(self.close)

    @property
    def target_path(self) -> Path:
        return self._target

    @property
    def flush_count(self) -> int:
        return self._flush_count

    @property
    def fsync_count(self) -> int:
        return self._fsync_count

    @property
    def durability_policy(self) -> str:
        return self._durability_policy

    @property
    def stats(self) -> BufferedWriterStats:
        with self._lock:
            return BufferedWriterStats(
                flush_count=self._flush_count,
                fsync_count=self._fsync_count,
                durability_policy=self._durability_policy,
                buffered_entries=len(self._buffer),
            )

    def record(self, payload: dict) -> None:
        line = json.dumps(payload, ensure_ascii=False)
        entry = f"{line}\n"
        with self._lock:
            if self._closed:
                raise RuntimeError("BufferedResultWriter already closed")
            self._buffer.append(entry)
            if self._should_flush_locked():
                self._flush_locked()

    def flush(self, *, final: bool = False) -> None:
        with self._lock:
            self._flush_locked(force=True, final=final)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._flush_locked(force=True, final=True)
            if self._durability_policy == "interval" and self._flush_count > self._last_fsync_flush_count:
                self._fsync_existing_target_locked()
            self._closed = True

    def _should_flush_locked(self) -> bool:
        if len(self._buffer) >= self._max_batch_size:
            return True
        elapsed = time.perf_counter() - self._last_flush
        return elapsed >= self._flush_interval

    def _flush_locked(self, force: bool = False, final: bool = False) -> None:
        if not self._buffer:
            return
        if not force and not self._should_flush_locked():
            return
        to_flush: Sequence[str] = list(self._buffer)
        self._buffer.clear()
        now = time.perf_counter()
        next_flush_count = self._flush_count + 1
        should_fsync = self._should_fsync_locked(final=final, now=now, next_flush_count=next_flush_count)
        self._last_flush = now
        try:
            self._write_lines(to_flush, should_fsync=should_fsync)
        except Exception:
            # Restore buffered payloads so callers can retry.
            self._buffer[0:0] = list(to_flush)
            raise
        else:
            self._flush_count = next_flush_count
            if should_fsync:
                self._last_fsync = now
                self._last_fsync_flush_count = next_flush_count

    def _write_lines(self, entries: Iterable[str], *, should_fsync: bool) -> None:
        joined = "".join(entries)
        if not joined:
            return
        if self._durability_policy == "always":
            self._write_pending(joined)
            with self._target.open("a", encoding="utf-8") as dest:
                dest.write(joined)
                dest.flush()
                self._fsync_handle(dest)
            if self._pending.exists():
                self._pending.unlink()
            return
        with self._target.open("a", encoding="utf-8") as dest:
            dest.write(joined)
            dest.flush()
            if should_fsync:
                self._fsync_handle(dest)
        if self._pending.exists():
            try:
                self._pending.unlink()
            except FileNotFoundError:
                pass

    def _write_pending(self, data: str) -> None:
        with self._pending.open("w", encoding="utf-8") as handle:
            handle.write(data)
            handle.flush()
            self._fsync_handle(handle)

    def _fsync_handle(self, handle) -> None:
        os.fsync(handle.fileno())
        self._fsync_count += 1

    def _should_fsync_locked(self, *, final: bool, now: float, next_flush_count: int) -> bool:
        if self._durability_policy == "always":
            return True
        if self._durability_policy == "never":
            return False
        if final:
            return True
        if next_flush_count - self._last_fsync_flush_count >= self._fsync_every_flushes:
            return True
        return now - self._last_fsync >= self._fsync_every_s

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

    def _fsync_existing_target_locked(self) -> None:
        if not self._target.exists():
            return
        with self._target.open("a", encoding="utf-8") as handle:
            handle.flush()
            self._fsync_handle(handle)
        now = time.perf_counter()
        self._last_fsync = now
        self._last_fsync_flush_count = self._flush_count


def _resolve_durability_policy(value: str) -> str:
    normalized = str(value).strip().lower()
    if normalized in {"always", "interval", "never"}:
        return normalized
    return "interval"


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default
