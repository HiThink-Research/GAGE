"""Streamable HTTP MCP client for AppWorld integration."""

from __future__ import annotations

import asyncio
import json
import socket
import threading
import time
from concurrent.futures import Future, TimeoutError as FutureTimeout
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence
from urllib.parse import urlparse


@dataclass
class _SessionState:
    task: Future[Any]
    stop_event: asyncio.Event
    ready_event: asyncio.Event


class AppWorldStreamableMcpClient:
    """MCP client wrapper that speaks streamable HTTP."""

    def __init__(
        self,
        *,
        mcp_client_id: str,
        transport: Optional[str] = None,
        endpoint: Optional[str] = None,
        timeout_s: Optional[int] = None,
        allowlist: Optional[Sequence[str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.mcp_client_id = mcp_client_id
        self.transport = transport or "streamable_http"
        self.endpoint = endpoint
        self.timeout_s = timeout_s or 30
        self.allowlist = set(allowlist or [])
        self._params = params or {}
        self._static_tools = list(self._params.get("tools") or self._params.get("static_tools") or [])
        self._requester: Optional[Callable[[str, Dict[str, Any]], Any]] = self._params.get("requester")
        self._executor: Optional[Callable[[str, Any], Any]] = self._params.get("executor")
        retry_attempts = self._params.get("session_retry_attempts")
        self._session_retry_attempts = (
            _coerce_retry_attempts(retry_attempts) if retry_attempts is not None else None
        )
        self._session_retry_delay_s = _coerce_retry_delay_s(self._params.get("session_retry_delay_s"))
        self._session_retry_timeout_s = _coerce_retry_timeout_s(
            self._params.get("session_retry_timeout_s")
        )
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._loop_lock = threading.Lock()
        self._session_state: Optional[_SessionState] = None
        self._session: Any = None

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP tool definitions available for this client."""

        tools = self._static_tools or []
        if not tools and callable(self._requester):
            tools = self._requester("list_tools", {"endpoint": self.endpoint}) or []
        if not tools:
            self._ensure_session()
            result = self._submit_coroutine(self._session.list_tools(), timeout_s=self.timeout_s)
            tools = _normalize_tool_list(result)
        tools = [tool for tool in tools if isinstance(tool, dict)]
        if self.allowlist:
            tools = [tool for tool in tools if tool.get("name") in self.allowlist]
        return tools

    def call_tool(self, name: str, arguments: Any) -> Any:
        """Call a tool by name with the provided arguments."""

        if self.allowlist and name not in self.allowlist:
            raise ValueError(f"Tool '{name}' not in allowlist")
        if callable(self._executor):
            return self._executor(name, arguments)
        if callable(self._requester):
            return self._requester(
                "call_tool",
                {"name": name, "arguments": arguments, "endpoint": self.endpoint},
            )
        self._ensure_session()
        result = self._submit_coroutine(
            self._session.call_tool(name, arguments=arguments or {}),
            timeout_s=self.timeout_s,
        )
        return normalize_mcp_call_result(result)

    def disconnect(self) -> None:
        """Close the MCP session and event loop."""

        if self._session_state:
            try:
                self._submit_coroutine(_async_set_event(self._session_state.stop_event), timeout_s=5)
                self._session_state.task.result(timeout=5)
            except FutureTimeout:
                self._session_state.task.cancel()
            self._session_state = None
            self._session = None
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._thread:
                self._thread.join()
        if self._loop:
            try:
                self._loop.close()
            except Exception:
                pass
        self._loop = None
        self._thread = None

    def _ensure_session(self) -> None:
        if self._session is not None:
            return
        last_error: Optional[BaseException] = None
        deadline = None
        if self._session_retry_timeout_s is not None:
            deadline = time.monotonic() + self._session_retry_timeout_s
        max_attempts = self._session_retry_attempts
        if max_attempts is None and deadline is None:
            max_attempts = 1
        attempt = 0
        while True:
            attempt += 1
            try:
                self._ensure_session_once()
                return
            except BaseException as exc:
                last_error = exc
                self._reset_session_state()
                if max_attempts is not None and attempt >= max_attempts:
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    break
                delay = self._session_retry_delay_s
                if delay <= 0 and deadline is not None:
                    delay = 0.2
                if delay > 0:
                    time.sleep(delay)
                continue
        if last_error is not None:
            raise last_error

    def _ensure_session_once(self) -> None:
        if not self.endpoint:
            raise RuntimeError("MCP client missing endpoint")
        _require_mcp()
        _check_reachability(self.endpoint, timeout_s=self.timeout_s)
        self._ensure_loop()
        ready_event = asyncio.Event()
        stop_event = asyncio.Event()
        error_ref: Dict[str, Optional[BaseException]] = {"exc": None}

        async def runner() -> None:
            from contextlib import AsyncExitStack

            try:
                from mcp import ClientSession
                from mcp.client.streamable_http import streamablehttp_client

                url = _normalize_mcp_url(self.endpoint)
                async with AsyncExitStack() as stack:
                    transport = await stack.enter_async_context(streamablehttp_client(url=url))
                    read, write, *_ = transport
                    session = await stack.enter_async_context(ClientSession(read, write, None))
                    await asyncio.wait_for(session.initialize(), timeout=self.timeout_s)
                    self._session = session
                    ready_event.set()
                    await stop_event.wait()
            except BaseException as exc:
                error_ref["exc"] = exc
                ready_event.set()

        task = asyncio.run_coroutine_threadsafe(runner(), self._loop)
        self._session_state = _SessionState(task=task, stop_event=stop_event, ready_event=ready_event)
        try:
            self._submit_coroutine(ready_event.wait(), timeout_s=self.timeout_s)
        except BaseException:
            task.cancel()
            self._reset_session_state()
            raise
        if error_ref["exc"] is not None:
            task.cancel()
            self._reset_session_state()
            raise RuntimeError(f"mcp_session_failed: {error_ref['exc']}") from error_ref["exc"]
        if self._session is None:
            task.cancel()
            self._reset_session_state()
            raise RuntimeError("mcp_session_failed: session_not_initialized")

    def _reset_session_state(self) -> None:
        if self._session_state:
            try:
                self._session_state.task.cancel()
            except Exception:
                pass
        self._session_state = None
        self._session = None

    def _ensure_loop(self) -> None:
        if self._loop:
            return
        with self._loop_lock:
            if self._loop:
                return
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=self._run_loop, args=(loop,), daemon=True)
            self._loop = loop
            self._thread = thread
            thread.start()

    def _run_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    def _submit_coroutine(self, coro: Any, *, timeout_s: Optional[float] = None) -> Any:
        if not self._loop:
            raise RuntimeError("mcp_client_loop_missing")
        timeout = self.timeout_s if timeout_s is None else max(0.1, float(timeout_s))
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout as exc:
            future.cancel()
            raise RuntimeError(f"mcp_request_timeout:{timeout}s") from exc


def _require_mcp() -> None:
    try:
        import mcp  # noqa: F401
        from mcp.client import streamable_http  # noqa: F401
    except Exception as exc:
        raise RuntimeError("mcp_package_missing: install 'mcp>=1.19.0'") from exc


def _check_reachability(endpoint: str, *, timeout_s: float) -> None:
    url = _normalize_mcp_url(endpoint).rstrip("/") + "/"
    parsed = urlparse(url)
    host = parsed.hostname
    port = parsed.port
    if not host:
        raise RuntimeError(f"mcp_endpoint_invalid:{url}")
    if port is None:
        port = 443 if parsed.scheme == "https" else 80
    timeout = max(0.2, min(1.0, float(timeout_s or 3)))
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return
    except OSError as exc:
        raise RuntimeError(f"mcp_endpoint_unreachable:{url}") from exc


def _normalize_mcp_url(endpoint: str) -> str:
    trimmed = endpoint.rstrip("/")
    if trimmed.endswith("/mcp"):
        return trimmed
    return f"{trimmed}/mcp"


def _normalize_tool_list(result: Any) -> List[Dict[str, Any]]:
    items = getattr(result, "tools", result)
    tools: List[Dict[str, Any]] = []
    for tool in items or []:
        tools.append(_normalize_tool_entry(tool))
    return tools


def _coerce_retry_attempts(value: Any) -> int:
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _coerce_retry_delay_s(value: Any) -> float:
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return 0.0


def _coerce_retry_timeout_s(value: Any) -> Optional[float]:
    try:
        timeout = float(value)
    except (TypeError, ValueError):
        return None
    if timeout <= 0:
        return None
    return timeout


def _normalize_tool_entry(tool: Any) -> Dict[str, Any]:
    if isinstance(tool, dict):
        return dict(tool)
    if hasattr(tool, "model_dump"):
        return tool.model_dump()
    if hasattr(tool, "dict"):
        return tool.dict()
    return {
        "name": getattr(tool, "name", ""),
        "description": getattr(tool, "description", ""),
        "inputSchema": getattr(tool, "inputSchema", None) or getattr(tool, "input_schema", None),
        "outputSchema": getattr(tool, "outputSchema", None) or getattr(tool, "output_schema", None),
    }


def normalize_mcp_call_result(result: Any) -> Dict[str, Any]:
    """Normalize MCP call_tool output to a JSON-friendly payload."""

    structured_content = getattr(result, "structuredContent", None)
    if structured_content is not None:
        return structured_content
    text_content = _extract_text_content(result)
    if isinstance(text_content, str):
        try:
            parsed = json.loads(text_content)
            if isinstance(parsed, dict) and "response" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass
    output: Dict[str, Any] = {"response": {"message": text_content}}
    if getattr(result, "isError", False):
        output["response"]["is_error"] = True
    return output


def _extract_text_content(result: Any) -> Optional[str]:
    content = getattr(result, "content", None)
    if not content:
        return None
    first = content[0]
    if isinstance(first, dict) and "text" in first:
        return str(first["text"])
    if hasattr(first, "text"):
        return str(first.text)
    return None


async def _async_set_event(event: asyncio.Event) -> None:
    event.set()
