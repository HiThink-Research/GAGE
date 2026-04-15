from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle


@dataclass(frozen=True)
class BenchmarkKitEntry:
    """Defines the callable surfaces exposed by one benchmark kit."""

    benchmark_kit_id: str
    runtime_version: str
    supported_schedulers: tuple[str, ...]
    verifier_kind: str
    resource_requirements: dict[str, Any]
    lifecycle_policy: dict[str, Any]
    state_schema_keys: tuple[str, ...]
    runtime_entry: Any
    workflow_resolver: Callable[[str], SchedulerWorkflowBundle]
    verifier_resource_resolver: Callable[[], dict[str, Any]]
    trace_mapper: Callable[..., dict[str, Any]] | None = None

    def resolve_workflow_bundle(self, scheduler_type: str) -> SchedulerWorkflowBundle:
        """Resolve the scheduler-local workflow bundle."""

        return self.workflow_resolver(scheduler_type)

    def resolve_verifier_resources(self) -> dict[str, Any]:
        """Resolve the runtime-owned verifier resources."""

        return self.verifier_resource_resolver()


def extract_instruction(sample: dict[str, Any]) -> str:
    """Extract the primary sample instruction for runtime-owned workflows."""

    instruction = sample.get("instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    prompt = sample.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        return prompt.strip()
    messages = sample.get("messages")
    if isinstance(messages, list):
        for message in messages:
            if not isinstance(message, dict):
                continue
            content = message.get("content")
            if isinstance(content, str) and content.strip():
                return content.strip()
    return ""


def normalize_messages(sample: dict[str, Any], fallback_text: str | None = None) -> list[dict[str, Any]]:
    """Resolve sample messages while keeping the workflow deterministic."""

    messages = sample.get("messages")
    if isinstance(messages, list) and messages:
        return [dict(message) for message in messages if isinstance(message, dict)]
    instruction = fallback_text or extract_instruction(sample)
    if instruction:
        return [{"role": "user", "content": instruction}]
    return []


def normalize_tools(sample: dict[str, Any], fallback_tools: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
    """Resolve sample tools with optional runtime fallback schemas."""

    tools = sample.get("tools")
    if isinstance(tools, list) and tools:
        return [dict(tool) for tool in tools if isinstance(tool, dict)]
    return [dict(tool) for tool in (fallback_tools or []) if isinstance(tool, dict)]


def build_noop_trace_mapping(*_, **__) -> dict[str, Any]:
    """Return a stable no-op trace payload."""

    return {}


def resolve_sample_artifact_target(session: Any, filename: str) -> tuple[Path, str]:
    """Resolve a canonical artifact target under the sample-scoped artifact root.

    Args:
        session: Runtime session carrying the artifact layout.
        filename: Artifact filename or relative path under the artifacts directory.

    Returns:
        A tuple of the absolute target path and the sample-root-relative path.
    """

    artifact_layout = dict(getattr(session, "artifact_layout", {}) or {})
    sample_root = Path(str(artifact_layout.get("sample_root") or "."))
    artifacts_dir = Path(str(artifact_layout.get("artifacts_dir") or sample_root / "artifacts"))
    relative_path = Path("artifacts") / filename
    target = artifacts_dir / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    return target, relative_path.as_posix()
