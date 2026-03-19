"""Best-effort process cleanup helpers."""

from __future__ import annotations

import atexit
import gc
import logging
import signal
import threading
from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _CleanupCallbackEntry:
    callback: Callable[[], None]


_CALLBACKS: List[_CleanupCallbackEntry] = []
_LOCK = threading.Lock()
_INSTALLED_SIGNALS = set()
_ATEXIT_INSTALLED = False
_CLEANED_UP = False


def install_signal_cleanup(
    callback: Callable[[], None], signals: Optional[Iterable[int]] = None
) -> Callable[[], None]:
    """
    Register a callback that should run when the process exits or receives a termination signal.

    Args:
        callback: Zero-argument callable that performs best-effort cleanup.
        signals: Optional iterable of signal numbers to hook. Defaults to INT/TERM/QUIT.

    Returns:
        A zero-argument function that unregisters the callback when cleanup is no longer needed.
    """
    if not callable(callback):
        raise TypeError("cleanup callback must be callable")

    if signals is None:
        desired = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGQUIT"):
            desired.append(signal.SIGQUIT)  # type: ignore[arg-type]
    else:
        desired = list(signals)

    entry = _CleanupCallbackEntry(callback=callback)
    with _LOCK:
        _install_atexit_cleanup_locked()
        _CALLBACKS.append(entry)
        for sig in desired:
            if sig in _INSTALLED_SIGNALS:
                continue
            try:
                signal.signal(sig, _handle_signal)
                _INSTALLED_SIGNALS.add(sig)
            except ValueError as exc:
                logger.debug("Failed to install cleanup handler for %s: %s", sig, exc)
    return lambda: _unregister_cleanup(entry)


def torch_gpu_cleanup() -> None:
    """Release CUDA memory to reduce the chance of leaked allocations after crashes."""
    try:
        import torch
    except Exception:
        return

    try:
        if torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
            except Exception:
                pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception as exc:
        logger.debug("torch GPU cleanup failed: %s", exc)

    try:
        gc.collect()
    except Exception:
        pass


def _handle_signal(signum, frame):  # pragma: no cover - signal handler
    logger.info("Received signal %s, running registered cleanup callbacks", signum)
    _run_callbacks()
    raise SystemExit(0)


def _install_atexit_cleanup_locked() -> None:
    global _ATEXIT_INSTALLED
    if _ATEXIT_INSTALLED:
        return
    atexit.register(_run_callbacks)
    _ATEXIT_INSTALLED = True


def _unregister_cleanup(entry: _CleanupCallbackEntry) -> None:
    with _LOCK:
        if _CLEANED_UP:
            return
        try:
            _CALLBACKS.remove(entry)
        except ValueError:
            return


def _run_callbacks() -> None:
    global _CLEANED_UP
    with _LOCK:
        if _CLEANED_UP:
            return
        _CLEANED_UP = True
        callbacks = [entry.callback for entry in _CALLBACKS]
        _CALLBACKS.clear()
    for callback in callbacks:
        try:
            callback()
        except Exception:
            logger.exception("Cleanup callback failed")
