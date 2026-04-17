"""DUT agent adapter (tool-calling agent with sandbox control)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Sequence

from gage_eval.registry import registry
from gage_eval.assets.prompts.renderers import PromptContext, PromptRenderer
from gage_eval.evaluation.support_artifacts import resolve_support_tools
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
        agent_backend: AgentBackend | None = None,
        prompt_renderer: Optional[PromptRenderer] = None,
        sandbox_manager: Optional[SandboxManager] = None,
        sandbox_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
        tool_router: Optional[ToolRouter] = None,
        mcp_clients: Optional[Dict[str, McpClient]] = None,
        human_gateway: Optional[HumanGateway] = None,
        agent_runtime_id: Optional[str] = None,
        compat_runtime_id: Optional[str] = None,
        executor_ref: Optional[Any] = None,
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
        self.agent_runtime_id = agent_runtime_id
        self.compat_runtime_id = compat_runtime_id
        self.executor_ref = executor_ref
        self.params = dict(params)
        self.max_turns = self._max_turns

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        if self.executor_ref is not None:
            return await self.executor_ref.aexecute(
                sample=payload.get("sample") or {},
                payload=payload,
                trace=payload.get("trace"),
            )
        if self._agent_backend is None:
            raise RuntimeError("agent_backend_missing_for_framework_loop")
        runtime_payload = dict(payload or {})
        runtime_payload.pop("trace", None)
        sample = runtime_payload.get("sample", {}) if isinstance(runtime_payload, dict) else {}
        messages = list(sample.get("messages") or runtime_payload.get("messages") or [])
        messages = self._apply_prompt(runtime_payload, sample, messages)
        system_prompt = _extract_system_prompt(messages)
        tools = self._resolve_tools(sample)
        tool_choice = sample.get("tool_choice")
        sandbox_provider = runtime_payload.get("sandbox_provider") if isinstance(runtime_payload, dict) else None
        sandbox_config = self._resolve_sandbox_config(sample, sandbox_provider)
        loop = AgentLoop(
            backend=self._agent_backend,
            tool_router=self._tool_router,
            max_turns=self._max_turns,
            pre_hooks=self._pre_hooks,
            post_hooks=self._post_hooks,
        )
        result = await loop.arun(
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

    def shutdown(self) -> None:
        """Release adapter-owned backend and sandbox resources."""

        issues: list[Exception] = []

        # STEP 1: Shut down the benchmark executor-owned sandbox manager.
        try:
            if self.executor_ref is not None:
                resource_manager = getattr(self.executor_ref, "resource_manager", None)
                sandbox_manager = getattr(resource_manager, "_sandbox_manager", None)
                shutdown_fn = getattr(sandbox_manager, "shutdown", None)
                if callable(shutdown_fn):
                    shutdown_fn()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            issues.append(exc)

        # STEP 2: Shut down the framework-loop agent backend.
        try:
            shutdown_fn = getattr(self._agent_backend, "shutdown", None)
            if callable(shutdown_fn):
                shutdown_fn()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            issues.append(exc)

        # STEP 3: Shut down the adapter-owned sandbox manager.
        try:
            self._sandbox_manager.shutdown()
        except Exception as exc:  # pragma: no cover - best-effort cleanup
            issues.append(exc)

        if issues:
            raise RuntimeError(
                "; ".join(f"{type(issue).__name__}: {issue}" for issue in issues)
            )

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
        prompt_payload = dict(payload or {})
        prompt_payload.setdefault("instruction", _extract_instruction(sample))
        prompt_payload.setdefault("max_steps", self._max_turns)
        context = PromptContext(
            sample=sample,
            payload=prompt_payload,
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
        prompt_context = sample.get("prompt_context")
        if isinstance(prompt_context, dict):
            tools = prompt_context.get("tools_schema")
            if isinstance(tools, list):
                return [dict(tool) for tool in tools if isinstance(tool, dict)]
        return resolve_support_tools(sample)


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


def _extract_instruction(sample: Dict[str, Any]) -> str:
    instruction = sample.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    prompt = sample.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        first = messages[0] if isinstance(messages[0], dict) else None
        if first:
            content = first.get("content")
            if isinstance(content, list) and content:
                text = content[0].get("text")
                if isinstance(text, str) and text.strip():
                    return text.strip()
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


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
