"""DUT agent adapter (tool-calling agent with sandbox control)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderer
from gage_eval.role.adapters.base import RoleAdapter, RoleAdapterState
from gage_eval.role.agent.hooks import build_hook_chain
from gage_eval.role.agent.human_gateway import HumanGateway, build_default_human_gateway
from gage_eval.role.agent.loop import AgentLoop
from gage_eval.role.agent.tool_router import ToolRouter
from gage_eval.role.agent.backends.base import AgentBackend
from gage_eval.mcp import McpClient
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider


@registry.asset(
    "roles",
    "dut_agent",
    desc="DUT agent adapter with tool-calling capabilities",
    tags=("role", "agent"),
    role_type="dut_agent",
)
class DUTAgentAdapter(RoleAdapter):
    """Agent adapter that runs an AgentLoop with tool routing."""

    def __init__(
        self,
        adapter_id: str,
        role_type: str,
        capabilities,
        *,
        agent_backend: AgentBackend,
        prompt_renderer: Optional[PromptRenderer] = None,
        sandbox_manager: Optional[SandboxManager] = None,
        sandbox_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        tool_router: Optional[ToolRouter] = None,
        mcp_clients: Optional[Dict[str, McpClient]] = None,
        human_gateway: Optional[HumanGateway] = None,
        max_turns: int = 8,
        **params,
    ) -> None:
        super().__init__(
            adapter_id=adapter_id,
            role_type=role_type,
            capabilities=capabilities,
            resource_requirement=params.pop("resource_requirement", None),
            sandbox_config=params.pop("sandbox_config", None),
        )
        self._agent_backend = agent_backend
        resolved_gateway = human_gateway or build_default_human_gateway()
        self._tool_router = tool_router or ToolRouter(mcp_clients=mcp_clients, human_gateway=resolved_gateway)
        self._sandbox_manager = sandbox_manager or SandboxManager(profiles=sandbox_profiles)
        self._max_turns = max(1, int(max_turns))
        self._pre_hooks = build_hook_chain(params.pop("pre_hooks", None) or params.pop("pre_hook", None))
        self._post_hooks = build_hook_chain(params.pop("post_hooks", None) or params.pop("post_hook", None))
        self._prompt_renderer = prompt_renderer

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        sample = payload.get("sample", {}) if isinstance(payload, dict) else {}
        messages = list(sample.get("messages") or payload.get("messages") or [])
        messages = self._apply_prompt(payload, sample, messages)
        system_prompt = _extract_system_prompt(messages)
        tools = self._resolve_tools(sample)
        tool_choice = sample.get("tool_choice")
        sandbox_provider = payload.get("sandbox_provider") if isinstance(payload, dict) else None
        sandbox_config = self._resolve_sandbox_config(sample, sandbox_provider)
        loop = AgentLoop(
            backend=self._agent_backend,
            tool_router=self._tool_router,
            max_turns=self._max_turns,
            pre_hooks=self._pre_hooks,
            post_hooks=self._post_hooks,
        )
        result = loop.run(
            messages=messages,
            tools=tools,
            tool_choice=tool_choice,
            sandbox_config=sandbox_config,
            sandbox_provider=sandbox_provider if isinstance(sandbox_provider, SandboxProvider) else None,
            metadata=sample.get("metadata") or {},
            sample=sample,
        )
        if system_prompt:
            result["system_prompt"] = system_prompt
        return result

    def _resolve_sandbox_config(
        self,
        sample: Dict[str, Any],
        sandbox_provider: Optional[SandboxProvider],
    ) -> Optional[Dict[str, Any]]:
        if sandbox_provider and sandbox_provider.sandbox_config:
            return sandbox_provider.sandbox_config
        role_config = dict(self.sandbox_config or {})
        sample_config = sample.get("sandbox") if isinstance(sample.get("sandbox"), dict) else None
        if not role_config and not sample_config:
            return None
        return self._sandbox_manager.resolve_config(role_config, sample_config)

    def _apply_prompt(
        self,
        payload: Dict[str, Any],
        sample: Dict[str, Any],
        messages: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if not self._prompt_renderer:
            return list(messages)
        context = PromptContext(
            sample=sample,
            payload=payload,
            history=payload.get("history") or [],
            extras={"adapter_id": self.adapter_id, "role_type": self.role_type},
        )
        rendered = self._prompt_renderer.render(context)
        if rendered.messages is not None:
            return rendered.messages
        if rendered.prompt:
            return [{"role": "system", "content": rendered.prompt}] + list(messages)
        return list(messages)

    @staticmethod
    def _resolve_tools(sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        tools: List[Dict[str, Any]] = []
        tools = _merge_tools(tools, sample.get("tools") or [])
        for output in sample.get("support_outputs") or []:
            if isinstance(output, dict):
                tools = _merge_tools(tools, output.get("tools_schema") or output.get("tools") or [])
        return tools


def _merge_tools(existing: List[Dict[str, Any]], new_tools: Any) -> List[Dict[str, Any]]:
    merged = list(existing)
    if isinstance(new_tools, dict):
        new_tools = [new_tools]
    for tool in new_tools or []:
        normalized = _normalize_tool_entry(tool)
        if not normalized:
            continue
        name = normalized.get("function", {}).get("name") if normalized.get("type") == "function" else None
        if name:
            merged = [t for t in merged if t.get("function", {}).get("name") != name]
        merged.append(normalized)
    return merged


def _normalize_tool_entry(tool: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(tool, dict):
        return None
    if tool.get("type") == "function" and "function" in tool:
        return dict(tool)
    if "name" in tool and "parameters" in tool:
        return {
            "type": "function",
            "function": {
                "name": tool.get("name"),
                "description": tool.get("description", ""),
                "parameters": tool.get("parameters") or {},
            },
        }
    return dict(tool)


def _extract_system_prompt(messages: Sequence[Dict[str, Any]]) -> Optional[str]:
    for message in messages:
        if not isinstance(message, dict):
            continue
        if message.get("role") != "system":
            continue
        content = message.get("content")
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    parts.append(item["text"])
            if parts:
                return "\n".join(parts).strip()
            return json.dumps(content, ensure_ascii=False)
        if content is not None:
            return str(content)
    return None
