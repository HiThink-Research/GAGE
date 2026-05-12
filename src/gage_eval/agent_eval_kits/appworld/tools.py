from __future__ import annotations

from gage_eval.agent_eval_kits.appworld.units import build_appworld_tools
from gage_eval.agent_runtime.tooling.registry import RuntimeToolRegistry


def build_tool_registry() -> RuntimeToolRegistry:
    registry = RuntimeToolRegistry()
    for schema in build_appworld_tools({}):
        registry.register_provider_schema(schema, provider="appworld", provider_kind="environment")
    return registry
