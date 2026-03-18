from __future__ import annotations

import threading

import pytest

from gage_eval.sandbox.manager import SandboxHandle, SandboxManager
from gage_eval.sandbox.pool import SandboxPool


class FakePoolSandbox:
    def __init__(self) -> None:
        self.stop_calls = 0

    def teardown(self) -> None:
        self.stop_calls += 1


class BlockingSandboxRuntime:
    _lock = threading.Lock()
    start_started = threading.Event()
    allow_start_finish = threading.Event()
    start_calls = 0

    @classmethod
    def reset(cls) -> None:
        with cls._lock:
            cls.start_calls = 0
        cls.start_started = threading.Event()
        cls.allow_start_finish = threading.Event()

    def __init__(self, runtime_configs=None, resources=None) -> None:
        self.stop_calls = 0

    def start(self, config):
        with type(self)._lock:
            type(self).start_calls += 1
        type(self).start_started.set()
        if not type(self).allow_start_finish.wait(timeout=2.0):
            raise RuntimeError("timed out waiting to finish sandbox startup")
        return {}

    def teardown(self) -> None:
        self.stop_calls += 1


class CountingSandboxRuntime:
    def __init__(self, runtime_configs=None, resources=None) -> None:
        self.stop_calls = 0

    def start(self, config):
        return {}

    def teardown(self) -> None:
        self.stop_calls += 1


@pytest.mark.fast
def test_sandbox_pool_acquire_respects_max_size_under_concurrency() -> None:
    build_started = threading.Event()
    allow_build_finish = threading.Event()
    second_done = threading.Event()
    build_count = 0
    build_count_lock = threading.Lock()
    acquired: list[FakePoolSandbox] = []
    errors: list[str] = []

    def builder() -> FakePoolSandbox:
        nonlocal build_count
        with build_count_lock:
            build_count += 1
        build_started.set()
        if not allow_build_finish.wait(timeout=2.0):
            raise RuntimeError("timed out waiting to finish pool build")
        return FakePoolSandbox()

    pool = SandboxPool(builder=builder, max_size=1)

    def acquire_into(items: list[FakePoolSandbox]) -> None:
        items.append(pool.acquire())

    def acquire_expect_error() -> None:
        try:
            pool.acquire()
        except Exception as exc:  # noqa: BLE001 - tests need the concrete message
            errors.append(str(exc))
        finally:
            second_done.set()

    threads = [
        threading.Thread(target=acquire_into, args=(acquired,)),
        threading.Thread(target=acquire_expect_error),
    ]

    try:
        threads[0].start()
        assert build_started.wait(timeout=1.0)
        threads[1].start()
        assert second_done.wait(timeout=1.0)
    finally:
        allow_build_finish.set()
        for thread in threads:
            thread.join(timeout=1.0)
        pool.shutdown()

    assert build_count == 1
    assert len(acquired) == 1
    assert errors == ["sandbox pool exhausted"]


@pytest.mark.fast
def test_sandbox_manager_acquire_respects_pool_max_under_concurrency() -> None:
    BlockingSandboxRuntime.reset()
    manager = SandboxManager()
    manager.register_runtime("blocking", BlockingSandboxRuntime)

    config = {
        "runtime": "blocking",
        "lifecycle": "per_run",
        "pool_key": "shared",
        "pool_max": 1,
    }
    handles: list[SandboxHandle] = []
    errors: list[str] = []
    second_done = threading.Event()

    def acquire_handle() -> None:
        handles.append(manager.acquire(config))

    def acquire_expect_error() -> None:
        try:
            manager.acquire(config)
        except Exception as exc:  # noqa: BLE001 - tests need the concrete message
            errors.append(str(exc))
        finally:
            second_done.set()

    threads = [
        threading.Thread(target=acquire_handle),
        threading.Thread(target=acquire_expect_error),
    ]

    try:
        threads[0].start()
        assert BlockingSandboxRuntime.start_started.wait(timeout=1.0)
        threads[1].start()
        assert second_done.wait(timeout=1.0)
    finally:
        BlockingSandboxRuntime.allow_start_finish.set()
        for thread in threads:
            thread.join(timeout=1.0)
        for handle in handles:
            manager.release(handle)
        manager.shutdown()

    assert BlockingSandboxRuntime.start_calls == 1
    assert len(handles) == 1
    assert errors == ["sandbox pool exhausted"]


@pytest.mark.fast
def test_sandbox_manager_release_after_shutdown_is_single_teardown() -> None:
    manager = SandboxManager()
    manager.register_runtime("counting", CountingSandboxRuntime)

    handle = manager.acquire({"runtime": "counting", "lifecycle": "per_run", "pool_max": 1})

    manager.shutdown()
    manager.release(handle)

    assert handle.sandbox.stop_calls == 1
