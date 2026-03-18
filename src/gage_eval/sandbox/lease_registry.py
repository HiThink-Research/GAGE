"""Persistent lease registry for managed sandbox runtimes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import socket
import uuid
from typing import Any, Dict, Iterable, Optional


@dataclass(frozen=True)
class SandboxLease:
    """Represents one persisted sandbox lease record."""

    lease_id: str
    runtime: str
    sandbox_id: Optional[str]
    pool_key: Optional[str]
    run_id: Optional[str]
    task_id: Optional[str]
    sample_id: Optional[str]
    owner_pid: int
    owner_host: str
    created_at: str
    config: Dict[str, Any]
    runtime_handle: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the lease to a JSON-friendly dict."""

        return {
            "lease_id": self.lease_id,
            "runtime": self.runtime,
            "sandbox_id": self.sandbox_id,
            "pool_key": self.pool_key,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "owner_pid": self.owner_pid,
            "owner_host": self.owner_host,
            "created_at": self.created_at,
            "config": self.config,
            "runtime_handle": self.runtime_handle,
        }

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "SandboxLease":
        """Build a lease instance from a persisted payload."""

        return cls(
            lease_id=str(payload.get("lease_id") or ""),
            runtime=str(payload.get("runtime") or ""),
            sandbox_id=_optional_string(payload.get("sandbox_id")),
            pool_key=_optional_string(payload.get("pool_key")),
            run_id=_optional_string(payload.get("run_id")),
            task_id=_optional_string(payload.get("task_id")),
            sample_id=_optional_string(payload.get("sample_id")),
            owner_pid=int(payload.get("owner_pid") or 0),
            owner_host=str(payload.get("owner_host") or ""),
            created_at=str(payload.get("created_at") or ""),
            config=_coerce_dict(payload.get("config")),
            runtime_handle=_coerce_dict(payload.get("runtime_handle")),
        )


class SandboxLeaseRegistry:
    """Stores sandbox leases on disk for crash recovery."""

    def __init__(self, base_dir: Optional[str] = None) -> None:
        env_base = os.environ.get("GAGE_EVAL_SAVE_DIR") or "./runs"
        self._base_dir = Path(base_dir or env_base).expanduser().resolve()
        self._lease_dir = self._base_dir / ".sandbox_leases"
        self._lease_dir.mkdir(parents=True, exist_ok=True)

    @property
    def lease_dir(self) -> Path:
        """Return the directory containing persisted leases."""

        return self._lease_dir

    def register(
        self,
        *,
        runtime: str,
        sandbox_id: Optional[str],
        pool_key: Optional[str],
        run_id: Optional[str],
        task_id: Optional[str],
        sample_id: Optional[str],
        config: Dict[str, Any],
        runtime_handle: Dict[str, Any],
    ) -> SandboxLease:
        """Persist a new lease record."""

        lease = SandboxLease(
            lease_id=uuid.uuid4().hex,
            runtime=str(runtime),
            sandbox_id=_optional_string(sandbox_id),
            pool_key=_optional_string(pool_key),
            run_id=_optional_string(run_id),
            task_id=_optional_string(task_id),
            sample_id=_optional_string(sample_id),
            owner_pid=os.getpid(),
            owner_host=socket.gethostname(),
            created_at=datetime.now(timezone.utc).isoformat(),
            config=_json_safe_copy(config),
            runtime_handle=_json_safe_copy(runtime_handle),
        )
        self._write_lease(lease)
        return lease

    def release(self, lease_id: str) -> None:
        """Delete a persisted lease if it exists."""

        path = self._lease_path(lease_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return

    def iter_leases(self) -> Iterable[SandboxLease]:
        """Yield all well-formed persisted leases."""

        for path in sorted(self._lease_dir.glob("*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                lease = SandboxLease.from_dict(payload)
            except Exception:
                continue
            if not lease.lease_id or not lease.runtime:
                continue
            yield lease

    def is_stale(self, lease: SandboxLease) -> bool:
        """Return whether the lease owner process is no longer alive."""

        owner_host = str(lease.owner_host or "")
        if owner_host and owner_host != socket.gethostname():
            return False
        return not _is_process_alive(lease.owner_pid)

    def _write_lease(self, lease: SandboxLease) -> None:
        path = self._lease_path(lease.lease_id)
        temp_path = path.with_suffix(".tmp")
        temp_path.write_text(
            json.dumps(lease.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temp_path.replace(path)

    def _lease_path(self, lease_id: str) -> Path:
        return self._lease_dir / f"{lease_id}.json"


def _coerce_dict(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return dict(raw)
    return {}


def _optional_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value)
    return text or None


def _json_safe_copy(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe_copy(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe_copy(item) for item in value]
    return str(value)


def _is_process_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True
