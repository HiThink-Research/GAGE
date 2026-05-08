from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from gage_eval.agent_runtime.tooling.contracts import ToolSchemaIR


ToolProviderKind = Literal["environment", "human", "local_function", "mcp"]


@dataclass(frozen=True)
class RuntimeToolEntry:
    schema: ToolSchemaIR
    provider_kind: ToolProviderKind
    executor: Any = None
    metadata: dict[str, Any] = field(default_factory=dict)


class RuntimeToolRegistry:
    """Single runtime entry point for all tool contributions."""

    def __init__(self) -> None:
        self._entries: dict[str, RuntimeToolEntry] = {}

    def register_provider_schema(
        self,
        raw_schema: dict[str, Any],
        *,
        provider: str | None = None,
        provider_kind: ToolProviderKind = "local_function",
        executor: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolSchemaIR:
        schema = ToolSchemaIR.from_provider_schema(raw_schema, provider=provider)
        self.register_schema(schema, provider_kind=provider_kind, executor=executor, metadata=metadata)
        return schema

    def register_schema(
        self,
        schema: ToolSchemaIR,
        *,
        provider_kind: ToolProviderKind = "local_function",
        executor: Any = None,
        metadata: dict[str, Any] | None = None,
    ) -> ToolSchemaIR:
        self._entries[schema.name] = RuntimeToolEntry(
            schema=schema,
            provider_kind=provider_kind,
            executor=executor,
            metadata=dict(metadata or {}),
        )
        return schema

    def register_environment_tool(self, schema: ToolSchemaIR, *, metadata: dict[str, Any] | None = None) -> ToolSchemaIR:
        return self.register_schema(schema, provider_kind="environment", metadata=metadata)

    def register_human_tool(
        self,
        schema: ToolSchemaIR,
        gateway: Any,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolSchemaIR:
        return self.register_schema(schema, provider_kind="human", executor=gateway, metadata=metadata)

    def register_local_function(
        self,
        schema: ToolSchemaIR,
        function: Callable[..., Any],
        *,
        metadata: dict[str, Any] | None = None,
    ) -> ToolSchemaIR:
        return self.register_schema(schema, provider_kind="local_function", executor=function, metadata=metadata)

    def register_mcp_tool(self, schema: ToolSchemaIR, client: Any, *, metadata: dict[str, Any] | None = None) -> ToolSchemaIR:
        return self.register_schema(schema, provider_kind="mcp", executor=client, metadata=metadata)

    def contribute(self, schemas: list[ToolSchemaIR], *, provider_kind: ToolProviderKind = "local_function") -> None:
        for schema in schemas:
            self.register_schema(schema, provider_kind=provider_kind)

    def get(self, name: str) -> RuntimeToolEntry | None:
        return self._entries.get(name)

    def entries(self) -> dict[str, RuntimeToolEntry]:
        return dict(self._entries)

    def project_tool_schemas(self) -> list[dict[str, Any]]:
        return [entry.schema.to_provider_schema() for entry in self._entries.values()]
