"""DUT agent adapter (tool-calling agent with sandbox control)."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence

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

if TYPE_CHECKING:
    from gage_eval.agent_runtime.resolver import AgentRuntimeResolver


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
        agent_backend: Optional[AgentBackend] = None,
        agent_runtime_resolver: Optional["AgentRuntimeResolver"] = None,
        agent_runtime_id: Optional[str] = None,
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
        self._sandbox_profiles = dict(sandbox_profiles or {})
        self._max_turns = max(1, int(max_turns))
        self._pre_hooks = build_hook_chain(params.pop("pre_hooks", None) or params.pop("pre_hook", None))
        self._post_hooks = build_hook_chain(params.pop("post_hooks", None) or params.pop("post_hook", None))
        self._prompt_renderer = prompt_renderer
        self._agent_runtime_resolver = agent_runtime_resolver
        self._agent_runtime_id = agent_runtime_id
        self.params = dict(params)

    async def ainvoke(self, payload: Dict[str, Any], state: RoleAdapterState) -> Dict[str, Any]:
        sample = payload.get("sample", {}) if isinstance(payload, dict) else {}
        runtime_id = (
            payload.get("agent_runtime_id")
            or sample.get("agent_runtime_id")
            or self._agent_runtime_id
        )
        if runtime_id and self._agent_runtime_resolver is not None:
            return self._run_via_agent_runtime(runtime_id, payload, sample, state)
        if self._agent_backend is None:
            raise ValueError("dut_agent requires agent_backend when agent_runtime is not configured")
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

    def _run_via_agent_runtime(
        self,
        runtime_id: str,
        payload: Dict[str, Any],
        sample: Dict[str, Any],
        state: RoleAdapterState,
    ) -> Dict[str, Any]:
        from gage_eval.agent_runtime.artifacts.layout import ArtifactLayout
        from gage_eval.agent_runtime.environment.provider import EnvironmentProvider
        from gage_eval.agent_runtime.resources.bundle import ResourceBundle
        from gage_eval.agent_runtime.session import AgentRuntimeSession
        from gage_eval.observability.trace import ObservabilityTrace

        trace = payload.get("trace")
        if not isinstance(trace, ObservabilityTrace):
            trace = ObservabilityTrace()
        plan = self._agent_runtime_resolver.resolve(runtime_id)
        scheduler = self._agent_runtime_resolver.build_scheduler(plan)
        environment = EnvironmentProvider(profiles=self._sandbox_profiles).build(plan, sample)
        resources = ResourceBundle(
            environment=environment,
            remote_sandbox=getattr(environment, "contract", None),
        )
        sample_id = str(
            sample.get("sample_id")
            or sample.get("id")
            or sample.get("instance_id")
            or "unknown"
        )
        task_id = (
            payload.get("task_id")
            or sample.get("task_id")
            or sample.get("_gage_task_id")
            or state.metadata.get("task_id")
        )
        artifacts = ArtifactLayout.for_sample(
            base_dir=str(sample.get("output_dir", "runs")),
            run_id=str(sample.get("run_id") or trace.run_id),
            sample_id=sample_id,
            task_id=None if task_id is None else str(task_id),
        )
        session = AgentRuntimeSession(
            sample=sample,
            trace=trace,
            plan=plan,
            resources=resources,
            artifacts=artifacts,
            metadata=dict(state.metadata or {}),
        )
        result = scheduler.run(session)
        output = dict(result.raw_output or {})
        verifier_output = self._run_runtime_verifier(plan, sample, result, artifacts, resources)
        if verifier_output:
            output.setdefault("verifier_result", verifier_output)
            output.setdefault("eval_result", verifier_output)
        if result.answer is not None:
            output.setdefault("answer", result.answer)
        output.setdefault("status", result.status)
        if result.patch_path is not None:
            output.setdefault("patch_path", result.patch_path)
        if result.stdout_path is not None:
            output.setdefault("stdout_path", result.stdout_path)
        if result.trajectory_path is not None:
            output.setdefault("trajectory_path", result.trajectory_path)
        if result.artifacts:
            output.setdefault("artifacts", dict(result.artifacts))
        if result.metrics:
            output.setdefault("metrics", dict(result.metrics))
        if verifier_output:
            sample["eval_result"] = dict(verifier_output)
            _persist_runtime_verifier_result(artifacts, verifier_output)
        return output

    def _run_runtime_verifier(
        self,
        plan,
        sample: Dict[str, Any],
        scheduler_result,
        artifacts,
        resources,
    ) -> Optional[Dict[str, Any]]:
        benchmark_kit_id = getattr(plan, "benchmark_kit_id", None)
        verifier_params = _resolve_runtime_verifier_params(plan)
        if benchmark_kit_id == "terminal_bench":
            from gage_eval.agent_eval_kits.terminal_bench.judge_bridge import build_verifier_input
            from gage_eval.agent_runtime.verifier.terminal_bench import TerminalBenchVerifier

            verifier_input = build_verifier_input(sample, scheduler_result, artifacts, resources)
            verifier_result = TerminalBenchVerifier().verify(verifier_input)
            return _normalize_verifier_result(verifier_result)
        if benchmark_kit_id == "swebench":
            from gage_eval.agent_eval_kits.swebench.judge_bridge import build_verifier_input
            from gage_eval.agent_runtime.verifier.judge_adapter import JudgeVerifierAdapter

            verifier_input = build_verifier_input(sample, scheduler_result, artifacts)
            verifier_result = JudgeVerifierAdapter(**verifier_params).verify(verifier_input)
            return _normalize_verifier_result(verifier_result)
        if benchmark_kit_id == "appworld":
            from gage_eval.agent_eval_kits.appworld.judge_bridge import build_verifier_input
            from gage_eval.role.judge.appworld_evaluate import AppWorldEvaluate

            verifier_input = build_verifier_input(sample, scheduler_result, artifacts, resources)
            judge_output = AppWorldEvaluate(**verifier_params).invoke(
                _build_runtime_judge_payload(verifier_input)
            )
            return _normalize_appworld_judge_output(judge_output)
        if benchmark_kit_id == "tau2":
            from gage_eval.agent_eval_kits.tau2.judge_bridge import build_verifier_input
            from gage_eval.role.judge.tau2_eval import Tau2Evaluate

            verifier_input = build_verifier_input(sample, scheduler_result, artifacts, resources)
            judge_output = Tau2Evaluate().invoke(_build_runtime_judge_payload(verifier_input))
            return _normalize_tau2_judge_output(judge_output)
        return None

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


def _normalize_verifier_result(verifier_result: Any) -> Dict[str, Any]:
    return {
        "status": verifier_result.status,
        "score": verifier_result.score,
        "summary": verifier_result.summary,
        "raw_output": dict(verifier_result.raw_output or {}),
    }


def _resolve_runtime_verifier_params(plan: Any) -> Dict[str, Any]:
    params = getattr(plan, "params", None)
    if not isinstance(params, dict):
        return {}
    raw_verifier = params.get("verifier")
    if isinstance(raw_verifier, dict):
        return dict(raw_verifier)
    return {}


def _build_runtime_judge_payload(verifier_input: Any) -> Dict[str, Any]:
    payload = dict(getattr(verifier_input, "payload", {}) or {})
    payload.setdefault(
        "sample",
        {"id": verifier_input.sample_id, "metadata": dict(getattr(verifier_input, "metadata", {}) or {})},
    )
    payload.setdefault("params", {})
    if getattr(verifier_input, "artifact_paths", None):
        payload.setdefault("artifact_paths", dict(verifier_input.artifact_paths))
    if getattr(verifier_input, "metadata", None):
        payload.setdefault("metadata", dict(verifier_input.metadata))
    if getattr(verifier_input, "runtime_handle", None):
        payload.setdefault("runtime_handle", dict(verifier_input.runtime_handle))
    return payload


def _normalize_appworld_judge_output(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "status": "error",
            "score": None,
            "summary": "judge_returned_non_mapping",
            "raw_output": {"value": result},
        }
    appworld = result.get("appworld") if isinstance(result.get("appworld"), dict) else {}
    failure_reason = appworld.get("failure_reason")
    status_hint = str(appworld.get("status") or "").strip().lower()
    tgc = _coerce_float(appworld.get("tgc"))
    if status_hint == "error":
        status = "error"
    elif tgc is None:
        status = "unknown"
    else:
        status = "pass" if tgc >= 1.0 else "fail"
    summary = str(failure_reason or (f"tgc={tgc}" if tgc is not None else "missing_tgc"))
    return {
        "status": status,
        "score": tgc,
        "summary": summary,
        "appworld": dict(appworld),
        "raw_output": dict(result),
    }


def _normalize_tau2_judge_output(result: Any) -> Dict[str, Any]:
    if not isinstance(result, dict):
        return {
            "status": "error",
            "score": None,
            "summary": "judge_returned_non_mapping",
            "raw_output": {"value": result},
        }
    tau2 = result.get("tau2") if isinstance(result.get("tau2"), dict) else {}
    reward = _coerce_float(tau2.get("reward"))
    failure_reason = tau2.get("failure_reason") or tau2.get("termination_reason")
    if reward is None:
        status = "unknown"
    elif (1.0 - 1e-6) <= reward <= (1.0 + 1e-6):
        status = "pass"
    else:
        status = "fail"
    summary = str(failure_reason or (f"reward={reward}" if reward is not None else "missing_reward"))
    return {
        "status": status,
        "score": reward,
        "summary": summary,
        "tau2": dict(tau2),
        "raw_output": dict(result),
    }


def _coerce_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _persist_runtime_verifier_result(artifacts: Any, verifier_output: Dict[str, Any]) -> None:
    target = getattr(artifacts, "verifier_result_file", None)
    if not target:
        return
    path = Path(str(target))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(verifier_output), ensure_ascii=False, indent=2), encoding="utf-8")


def _json_safe(value: Any) -> Any:
    if is_dataclass(value):
        return _json_safe(asdict(value))
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)
