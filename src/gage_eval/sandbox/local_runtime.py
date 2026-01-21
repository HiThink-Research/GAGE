"""Local subprocess sandbox implementation (unsafe, host-only)."""

from __future__ import annotations

import subprocess
import time
from typing import Any, Dict, Optional

from gage_eval.sandbox.base import BaseSandbox, ExecResult, SandboxOptionalMixin


class LocalSubprocessSandbox(SandboxOptionalMixin, BaseSandbox):
    """Local subprocess sandbox for development/debugging."""

    def __init__(self, runtime_configs: Optional[Dict[str, Any]] = None, resources: Optional[Dict[str, Any]] = None) -> None:
        self._runtime_configs = dict(runtime_configs or {})
        self._resources = dict(resources or {})
        self._running = False

    def start(self, config: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        self._runtime_configs.update(config.get("runtime_configs", {}) or {})
        self._running = True
        return {"profile": "local_subprocess"}

    def exec(self, command: str, timeout: int = 30) -> ExecResult:
        runner = self._runtime_configs.get("command_runner")
        if callable(runner):
            result = runner(command, timeout)
            if isinstance(result, ExecResult):
                return result
            if isinstance(result, dict):
                return ExecResult(
                    exit_code=int(result.get("exit_code", 0)),
                    stdout=str(result.get("stdout", "")),
                    stderr=str(result.get("stderr", "")),
                    duration_ms=float(result.get("duration_ms", 0.0)),
                )
            return ExecResult(exit_code=0, stdout=str(result), stderr="", duration_ms=0.0)
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        return ExecResult(
            exit_code=completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=elapsed_ms,
        )

    def teardown(self) -> None:
        self._running = False

    def is_alive(self, timeout_s: float | None = None) -> bool:
        return self._running
