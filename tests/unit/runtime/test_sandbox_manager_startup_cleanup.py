from __future__ import annotations

import json

import pytest

from gage_eval.sandbox.lease_registry import SandboxLeaseRegistry
from gage_eval.sandbox.manager import SandboxManager


class TrackingSandbox:
    cleanup_calls: list[tuple[dict, dict]] = []

    def __init__(self, runtime_configs=None, resources=None):
        self.runtime_configs = runtime_configs or {}
        self.resources = resources or {}
        self.stop_calls = 0

    def start(self, config):
        return {"profile": "tracking"}

    def teardown(self):
        self.stop_calls += 1

    @classmethod
    def cleanup_stale_runtime(cls, config, runtime_handle):
        cls.cleanup_calls.append((dict(config), dict(runtime_handle)))
        return True


@pytest.mark.fast
def test_sandbox_manager_reaps_stale_leases_on_first_acquire(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    TrackingSandbox.cleanup_calls.clear()
    registry = SandboxLeaseRegistry()
    lease = registry.register(
        runtime="tracking",
        sandbox_id="demo",
        pool_key=None,
        run_id="run-1",
        task_id=None,
        sample_id="sample-1",
        config={"runtime": "tracking", "sandbox_id": "demo"},
        runtime_handle={"resource_id": "orphan-1"},
    )
    lease_path = registry.lease_dir / f"{lease.lease_id}.json"
    payload = json.loads(lease_path.read_text(encoding="utf-8"))
    payload["owner_pid"] = 999999
    lease_path.write_text(json.dumps(payload), encoding="utf-8")

    manager = SandboxManager()
    manager.register_runtime("tracking", TrackingSandbox)
    handle = manager.acquire({"runtime": "tracking"})

    assert TrackingSandbox.cleanup_calls == [
        ({"runtime": "tracking", "sandbox_id": "demo"}, {"resource_id": "orphan-1"})
    ]
    assert lease_path.exists() is False

    manager.release(handle)
    manager.shutdown()


@pytest.mark.fast
def test_sandbox_manager_keeps_pool_lease_until_shutdown(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    manager = SandboxManager()
    manager.register_runtime("tracking", TrackingSandbox)

    handle = manager.acquire({"runtime": "tracking", "lifecycle": "per_run"})
    lease_files = list((tmp_path / ".sandbox_leases").glob("*.json"))

    assert len(lease_files) == 1

    manager.release(handle)

    assert len(list((tmp_path / ".sandbox_leases").glob("*.json"))) == 1

    manager.shutdown()

    assert len(list((tmp_path / ".sandbox_leases").glob("*.json"))) == 0
    assert handle.sandbox.stop_calls == 1


@pytest.mark.fast
def test_sandbox_manager_reaps_stale_leases_after_pid_reuse(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    TrackingSandbox.cleanup_calls.clear()
    registry = SandboxLeaseRegistry()
    lease = registry.register(
        runtime="tracking",
        sandbox_id="demo",
        pool_key=None,
        run_id="run-2",
        task_id=None,
        sample_id="sample-2",
        config={"runtime": "tracking", "sandbox_id": "demo"},
        runtime_handle={"resource_id": "orphan-2"},
    )
    lease_path = registry.lease_dir / f"{lease.lease_id}.json"
    payload = json.loads(lease_path.read_text(encoding="utf-8"))
    payload["owner_pid"] = 1234
    payload["owner_host_identity"] = "stable-host"
    payload["owner_process_start"] = "old-start"
    lease_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr("gage_eval.sandbox.lease_registry._resolve_host_identity", lambda: "stable-host")
    monkeypatch.setattr("gage_eval.sandbox.lease_registry._is_process_alive", lambda pid: pid == 1234)
    monkeypatch.setattr("gage_eval.sandbox.lease_registry._resolve_process_start_marker", lambda pid: "new-start")

    manager = SandboxManager()
    manager.register_runtime("tracking", TrackingSandbox)
    handle = manager.acquire({"runtime": "tracking"})

    assert TrackingSandbox.cleanup_calls == [
        ({"runtime": "tracking", "sandbox_id": "demo"}, {"resource_id": "orphan-2"})
    ]
    assert lease_path.exists() is False

    manager.release(handle)
    manager.shutdown()


@pytest.mark.fast
def test_sandbox_manager_reaps_stale_leases_after_hostname_change_with_same_host_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    TrackingSandbox.cleanup_calls.clear()
    registry = SandboxLeaseRegistry()
    lease = registry.register(
        runtime="tracking",
        sandbox_id="demo",
        pool_key=None,
        run_id="run-3",
        task_id=None,
        sample_id="sample-3",
        config={"runtime": "tracking", "sandbox_id": "demo"},
        runtime_handle={"resource_id": "orphan-3"},
    )
    lease_path = registry.lease_dir / f"{lease.lease_id}.json"
    payload = json.loads(lease_path.read_text(encoding="utf-8"))
    payload["owner_pid"] = 999999
    payload["owner_host"] = "previous-hostname"
    payload["owner_host_identity"] = "stable-host"
    lease_path.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr("gage_eval.sandbox.lease_registry._resolve_host_identity", lambda: "stable-host")

    manager = SandboxManager()
    manager.register_runtime("tracking", TrackingSandbox)
    handle = manager.acquire({"runtime": "tracking"})

    assert TrackingSandbox.cleanup_calls == [
        ({"runtime": "tracking", "sandbox_id": "demo"}, {"resource_id": "orphan-3"})
    ]
    assert lease_path.exists() is False

    manager.release(handle)
    manager.shutdown()
