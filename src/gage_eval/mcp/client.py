"""Minimal MCP client wrapper for tool discovery and execution."""

from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence


class McpClient:
    """Client facade for MCP servers.

    Args:
        mcp_client_id: Stable identifier for the MCP client.
        transport: Transport name (for example: http_sse).
        endpoint: MCP server endpoint.
        timeout_s: Request timeout in seconds.
        allowlist: Tool name allowlist. Empty means allow all.
        params: Additional configuration (static tools, custom executors).
    """

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
        self.transport = transport or "http_sse"
        self.endpoint = endpoint
        self.timeout_s = timeout_s or 30
        self.allowlist = set(allowlist or [])
        self._params = params or {}
        self._static_tools = list(self._params.get("tools") or self._params.get("static_tools") or [])
        self._requester: Optional[Callable[[str, Dict[str, Any]], Any]] = self._params.get("requester")
        self._executor: Optional[Callable[[str, Any], Any]] = self._params.get("executor")

    def list_tools(self) -> List[Dict[str, Any]]:
        """Return MCP tool definitions available for this client.

        Returns:
            A list of MCP tool dictionaries.
        """

        tools = self._static_tools or []
        if not tools and callable(self._requester):
            tools = self._requester("list_tools", {"endpoint": self.endpoint}) or []
        if not tools and self.endpoint:
            response = self._request("list_tools", {})
            tools = response.get("tools") if isinstance(response, dict) else response
        tools = [tool for tool in tools if isinstance(tool, dict)]
        if self.allowlist:
            tools = [tool for tool in tools if tool.get("name") in self.allowlist]
        return tools

    def call_tool(self, name: str, arguments: Any) -> Any:
        """Call a tool by name with the provided arguments.

        Args:
            name: Tool name.
            arguments: Tool arguments payload.

        Returns:
            Tool execution response.
        """

        if self.allowlist and name not in self.allowlist:
            raise ValueError(f"Tool '{name}' not in allowlist")
        if callable(self._executor):
            return self._executor(name, arguments)
        if callable(self._requester):
            return self._requester("call_tool", {"name": name, "arguments": arguments, "endpoint": self.endpoint})
        if self.endpoint:
            return self._request("call_tool", {"name": name, "arguments": arguments})
        raise RuntimeError("MCP client missing executor/requester")

    def list_resources(self) -> List[Dict[str, Any]]:
        """List MCP resources such as prompts or documents.

        Returns:
            A list of resource entries.
        """

        response = self._request("list_resources", {})
        if isinstance(response, dict):
            items = response.get("resources") or response.get("items") or []
        else:
            items = response
        return [item for item in items if isinstance(item, dict)]

    def read_resource(self, uri: str) -> Any:
        """Read a resource by URI from the MCP server.

        Args:
            uri: Resource URI to retrieve.

        Returns:
            Resource payload from the MCP server.
        """

        return self._request("read_resource", {"uri": uri})

    def sample(self, payload: Dict[str, Any]) -> Any:
        """Submit a sampling request to the MCP server.

        Args:
            payload: Sampling request payload.

        Returns:
            Sampling response payload.
        """

        return self._request("sample", payload)

    def _request(self, method: str, payload: Dict[str, Any]) -> Any:
        if callable(self._requester):
            return self._requester(method, payload)
        if not self.endpoint:
            raise RuntimeError("MCP client missing endpoint")
        try:
            import requests
        except Exception as exc:
            raise RuntimeError("mcp_request_dependency_missing: install requests") from exc
        timeout = max(1.0, float(self.timeout_s or 30))
        connect_timeout = min(5.0, timeout)
        try:
            response = requests.post(
                self.endpoint,
                json={"method": method, "params": payload},
                timeout=(connect_timeout, timeout),
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"mcp_request_failed:{exc}") from exc
        try:
            return response.json()
        except Exception:
            return {"content": response.text}
