from __future__ import annotations

import pytest

from gage_eval.role.model.backends.litellm_backend import LiteLLMBackend


def _build_backend_for_retry_tests(*, max_retries: int) -> LiteLLMBackend:
    backend = LiteLLMBackend.__new__(LiteLLMBackend)
    backend._max_retries = max_retries
    backend._retry_sleep = 0.0
    backend._retry_multiplier = 1.0
    return backend


@pytest.mark.fast
def test_call_with_retries_stops_for_shutdown_error() -> None:
    backend = _build_backend_for_retry_tests(max_retries=6)
    attempts = 0

    def _call() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("cannot schedule new futures after shutdown")

    with pytest.raises(RuntimeError, match="cannot schedule new futures after shutdown"):
        backend._call_with_retries(_call)

    assert attempts == 1


@pytest.mark.fast
def test_call_with_retries_retries_for_retryable_error() -> None:
    backend = _build_backend_for_retry_tests(max_retries=3)
    attempts = 0

    def _call() -> None:
        nonlocal attempts
        attempts += 1
        raise RuntimeError("temporary upstream error")

    with pytest.raises(RuntimeError, match="temporary upstream error"):
        backend._call_with_retries(_call)

    assert attempts == 3
