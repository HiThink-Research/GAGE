"""Experimental local model server lifecycle support for LiteLLM backends."""

from __future__ import annotations

import os
import re
import signal
import subprocess  # nosec
import time
from dataclasses import dataclass
from typing import Any, Callable
from urllib.request import Request, urlopen as default_urlopen

from loguru import logger

from gage_eval.role.model.backends.litellm.errors import DEPENDENCY_UNAVAILABLE, LiteLLMBackendError


@dataclass(frozen=True)
class ModelServerState:
    """Result of a model server readiness check."""

    enabled: bool
    ready: bool
    started: bool
    health_urls: tuple[str, ...] = ()


class ModelServerLifecycle:
    """Start a user-provided local model server command and wait for health."""

    _MIN_HEALTH_INTERVAL_SECONDS = 0.1
    _UNTRUSTED_PATTERNS = (
        re.compile(r"\bcurl\b.+\|\s*(?:bash|sh)\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\bwget\b.+\|\s*(?:bash|sh)\b", re.IGNORECASE | re.DOTALL),
        re.compile(r"\brm\s+-[^\n;|&]*[rf][^\n;|&]*\s+/", re.IGNORECASE),
        re.compile(r"\bmkfs(?:\.[A-Za-z0-9_+-]+)?\b", re.IGNORECASE),
    )

    def __init__(
        self,
        *,
        popen: Callable[..., Any] = subprocess.Popen,
        urlopen: Callable[..., Any] = default_urlopen,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
        getpgid: Callable[[int], int] = os.getpgid,
        killpg: Callable[[int, int], None] = os.killpg,
    ) -> None:
        self._popen = popen
        self._urlopen = urlopen
        self._sleep = sleep
        self._monotonic = monotonic
        self._getpgid = getpgid
        self._killpg = killpg
        self._started_processes: dict[tuple[str, tuple[str, ...]], Any] = {}
        self._started_shutdown_policies: dict[tuple[str, tuple[str, ...]], str] = {}

    def ensure_ready(self, config: Any) -> ModelServerState:
        """Ensure the configured experimental model server is ready."""

        if not self._enabled(config):
            return ModelServerState(enabled=False, ready=True, started=False)

        startup_command = self._validate_trusted_command(self._get(config, "startup_command"))
        urls = self._health_urls(config)
        shutdown_policy = self._shutdown_policy(config)
        if not urls:
            raise ValueError("model_server.health.urls must contain at least one health URL when enabled.")

        if self._are_healthy(urls):
            return ModelServerState(enabled=True, ready=True, started=False, health_urls=tuple(urls))

        key = (startup_command, tuple(urls))
        process = self._started_processes.get(key)
        started = False
        if process is None:
            process = self._start_command(startup_command, shutdown_policy=shutdown_policy)
            self._started_processes[key] = process
            self._started_shutdown_policies[key] = shutdown_policy
            started = True
        timeout_seconds = self._health_timeout_seconds(config)
        interval_seconds = self._health_interval_seconds(config)
        self._wait_until_healthy(
            urls,
            timeout_seconds=timeout_seconds,
            interval_seconds=interval_seconds,
            process=process,
            startup_command=startup_command,
        )
        return ModelServerState(enabled=True, ready=True, started=started, health_urls=tuple(urls))

    def close(self) -> None:
        """Best-effort shutdown for commands explicitly owned by this lifecycle."""

        for key, process in list(self._started_processes.items()):
            policy = self._started_shutdown_policies.get(key, "keep_running")
            if policy != "terminate_on_exit":
                continue
            self._terminate_process_group(process)
            self._started_processes.pop(key, None)
            self._started_shutdown_policies.pop(key, None)

    def stop(self) -> None:
        """Alias retained for callers that use stop terminology."""

        self.close()

    def _wait_until_healthy(
        self,
        urls: list[str],
        *,
        timeout_seconds: float,
        interval_seconds: float,
        process: Any | None,
        startup_command: str,
    ) -> None:
        deadline = self._monotonic() + timeout_seconds
        while True:
            if self._are_healthy(urls):
                return
            self._raise_if_process_exited(process, startup_command)
            now = self._monotonic()
            if now >= deadline:
                raise LiteLLMBackendError(
                    "Model server health check timed out before all URLs became healthy: "
                    + ", ".join(urls),
                    DEPENDENCY_UNAVAILABLE,
                )
            self._sleep(min(interval_seconds, max(0.0, deadline - now)))

    def _are_healthy(self, urls: list[str]) -> bool:
        return all(self._is_healthy(url) for url in urls)

    def _is_healthy(self, url: str) -> bool:
        request = Request(url, method="GET")
        try:
            with self._urlopen(request, timeout=2.0) as response:
                status = getattr(response, "status", None) or getattr(response, "code", None) or 200
                if int(status) >= 400:
                    return False
                read = getattr(response, "read", None)
                if callable(read):
                    read()
                return True
        except Exception:
            return False

    def _start_command(self, startup_command: str, *, shutdown_policy: str) -> Any:
        try:
            return self._popen(
                ["bash", "-lc", startup_command],
                stdin=subprocess.DEVNULL,
                start_new_session=shutdown_policy == "terminate_on_exit",
            )
        except Exception as exc:
            raise LiteLLMBackendError(
                f"Model server startup command failed: {exc}",
                DEPENDENCY_UNAVAILABLE,
            ) from exc

    def _terminate_process_group(self, process: Any) -> None:
        pid = getattr(process, "pid", None)
        if not isinstance(pid, int) or pid <= 0:
            return
        try:
            pgid = self._getpgid(pid)
            self._killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return
        except Exception as exc:
            logger.warning("Model server process group shutdown failed for pid={}: {}", pid, exc)
            return

    def _raise_if_process_exited(self, process: Any | None, startup_command: str) -> None:
        if process is None:
            return
        poll = getattr(process, "poll", None)
        if not callable(poll):
            return
        returncode = poll()
        if returncode not in (None, 0):
            raise LiteLLMBackendError(
                f"Model server startup command exited with code {returncode}: {startup_command}",
                DEPENDENCY_UNAVAILABLE,
            )

    def _validate_trusted_command(self, startup_command: Any) -> str:
        if not isinstance(startup_command, str):
            raise ValueError("model_server.startup_command must be a string.")
        command = startup_command.strip()
        if not command:
            raise ValueError("model_server.startup_command must not be empty.")
        if "\x00" in command:
            raise ValueError("model_server.startup_command must not contain NUL bytes.")
        if any(pattern.search(command) for pattern in self._UNTRUSTED_PATTERNS):
            raise ValueError("model_server.startup_command contains obviously unsafe shell input.")
        return startup_command

    @staticmethod
    def _enabled(config: Any) -> bool:
        return bool(ModelServerLifecycle._get(config, "enabled", False))

    @staticmethod
    def _get(config: Any, key: str, default: Any = None) -> Any:
        if isinstance(config, dict):
            return config.get(key, default)
        return getattr(config, key, default)

    def _health_urls(self, config: Any) -> list[str]:
        health = self._get(config, "health")
        urls = self._get(health, "urls", []) if health is not None else []
        if urls is None:
            return []
        if not isinstance(urls, list):
            raise ValueError("model_server.health.urls must be a list of URLs.")
        validated: list[str] = []
        for url in urls:
            if not isinstance(url, str) or not url.strip():
                raise ValueError("model_server.health.urls must only contain non-empty strings.")
            validated.append(url.strip())
        return validated

    def _health_timeout_seconds(self, config: Any) -> float:
        health = self._get(config, "health")
        value = self._get(health, "timeout_seconds", None) if health is not None else None
        if value is None:
            return 300.0
        return float(value)

    def _health_interval_seconds(self, config: Any) -> float:
        health = self._get(config, "health")
        value = self._get(health, "interval_seconds", None) if health is not None else None
        if value is None:
            return 2.0
        return max(self._MIN_HEALTH_INTERVAL_SECONDS, float(value))

    def _shutdown_policy(self, config: Any) -> str:
        value = self._get(config, "shutdown_policy", "keep_running") or "keep_running"
        policy = str(value).strip().lower().replace("-", "_")
        if policy not in {"keep_running", "terminate_on_exit"}:
            raise ValueError("model_server.shutdown_policy must be keep_running or terminate_on_exit.")
        return policy
