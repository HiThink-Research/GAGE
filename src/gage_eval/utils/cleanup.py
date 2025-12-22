import gc
import logging
import signal
import threading
from typing import Callable, Iterable, List, Optional

logger = logging.getLogger(__name__)

_CALLBACKS: List[Callable[[], None]] = []
_LOCK = threading.Lock()
_INSTALLED_SIGNALS = set()
_CLEANED_UP = False


def install_signal_cleanup(
    callback: Callable[[], None], signals: Optional[Iterable[int]] = None
) -> None:
    """
    Register a callback that should run when the process receives termination signals.

    Args:
        callback: Zero-argument callable that performs best-effort cleanup.
        signals: Optional iterable of signal numbers to hook. Defaults to INT/TERM/QUIT.
    """
    if not callable(callback):
        raise TypeError("cleanup callback must be callable")

    if signals is None:
        desired = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGQUIT"):
            desired.append(signal.SIGQUIT)  # type: ignore[arg-type]
    else:
        desired = list(signals)

    with _LOCK:
        _CALLBACKS.append(callback)
        for sig in desired:
            if sig in _INSTALLED_SIGNALS:
                continue
            try:
                signal.signal(sig, _handle_signal)
                _INSTALLED_SIGNALS.add(sig)
            except ValueError as exc:
                logger.debug("Failed to install cleanup handler for %s: %s", sig, exc)


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


def _run_callbacks() -> None:
    global _CLEANED_UP
    with _LOCK:
        if _CLEANED_UP:
            return
        _CLEANED_UP = True
        callbacks = list(_CALLBACKS)
    for callback in callbacks:
        try:
            callback()
        except Exception:
            logger.exception("Cleanup callback failed")
