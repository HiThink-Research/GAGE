from __future__ import annotations

from gage_eval.agent_eval_kits.terminal_bench.units import build_terminal_tools
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry


def build_tool_registry() -> RuntimeToolRegistry:
    registry = RuntimeToolRegistry()
    for schema in build_terminal_tools({}):
        registry.register_provider_schema(schema, provider="appworld", provider_kind="environment")
    return registry
