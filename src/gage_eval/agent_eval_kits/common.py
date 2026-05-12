from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict

from gage_eval.agent_runtime.compiled_plan import SchedulerWorkflowBundle


_INIT_PROFILE_OMIT_KEYS = {"image", "privileged", "cpu", "memory", "network", "network_policy"}
_UNSET = object()


class EmptyKitConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _build_empty_tool_registry() -> dict[str, Any]:
    return {}


@dataclass(frozen=True, init=False)
class BenchmarkKitEntry:
    """Defines the callable surfaces exposed by one benchmark kit."""

    kit_id: str
    config_schema: type[BaseModel] | None
    default_environment_provider: str
    default_environment_profile_by_provider: dict[str, str]
    environment_profiles: dict[str, dict[str, Any]]
    verifier_environment_policy: str
    verifier_environment_profile_id: str | None
    supported_schedulers: tuple[str, ...]
    workflow_resolver: Callable[[str], SchedulerWorkflowBundle]
    tool_registry_factory: Callable[[], Any] | None
    verifier_adapter_factory: Callable[[], Any] | None
    artifact_manifest_factory: Callable[[], Any] | None
    provider_config_resolver: Callable[..., dict[str, Any]] | None
    runtime_entry: Any

    def __init__(
        self,
        *,
        kit_id: str | None = None,
        config_schema: type[BaseModel] | None = None,
        default_environment_provider: str | None = None,
        default_environment_profile_by_provider: dict[str, str] | None = None,
        environment_profiles: dict[str, dict[str, Any]] | None = None,
        verifier_environment_policy: str = "reuse",
        verifier_environment_profile_id: str | None | object = _UNSET,
        supported_schedulers: tuple[str, ...],
        workflow_resolver: Callable[[str], SchedulerWorkflowBundle],
        tool_registry_factory: Callable[[], Any] | None = _build_empty_tool_registry,
        verifier_adapter_factory: Callable[[], Any] | None = None,
        artifact_manifest_factory: Callable[[], Any] | None = None,
        provider_config_resolver: Callable[..., dict[str, Any]] | None = None,
        runtime_entry: Any = None,
    ) -> None:
        if not kit_id:
            raise ValueError("kit_id is required")
        provider = default_environment_provider or ""
        explicit_verifier_profile = None if verifier_environment_profile_id is _UNSET else verifier_environment_profile_id
        profile_id = (default_environment_profile_by_provider or {}).get(provider) or explicit_verifier_profile
        profile_by_provider = dict(default_environment_profile_by_provider or {})
        profiles = dict(environment_profiles or {})
        if verifier_environment_profile_id is _UNSET:
            verifier_profile_id = profile_id
        else:
            verifier_profile_id = verifier_environment_profile_id

        object.__setattr__(self, "kit_id", kit_id)
        object.__setattr__(self, "config_schema", config_schema)
        object.__setattr__(self, "default_environment_provider", provider)
        object.__setattr__(self, "default_environment_profile_by_provider", profile_by_provider)
        object.__setattr__(self, "environment_profiles", profiles)
        object.__setattr__(self, "verifier_environment_policy", verifier_environment_policy)
        object.__setattr__(self, "verifier_environment_profile_id", verifier_profile_id)
        object.__setattr__(self, "supported_schedulers", tuple(supported_schedulers))
        object.__setattr__(self, "workflow_resolver", workflow_resolver)
        object.__setattr__(self, "tool_registry_factory", tool_registry_factory)
        object.__setattr__(self, "verifier_adapter_factory", verifier_adapter_factory)
        object.__setattr__(self, "artifact_manifest_factory", artifact_manifest_factory)
        object.__setattr__(self, "provider_config_resolver", provider_config_resolver)
        object.__setattr__(self, "runtime_entry", runtime_entry)

    def resolve_workflow_bundle(self, scheduler_type: str) -> SchedulerWorkflowBundle:
        """Resolve the scheduler-local workflow bundle."""

        return self.workflow_resolver(scheduler_type)

    def resolve_verifier_resources(self) -> dict[str, Any]:
        """Resolve the runtime-owned verifier resources."""

        adapter = self.build_verifier_adapter()
        return {"adapter": adapter} if adapter is not None else {}

    def build_tool_registry(self) -> Any:
        """Build the framework-loop tool registry for this kit."""

        if self.tool_registry_factory is None:
            return None
        return self.tool_registry_factory()

    def build_verifier_adapter(self) -> Any:
        """Build the verifier adapter for this kit."""

        if self.verifier_adapter_factory is not None:
            return self.verifier_adapter_factory()
        return None

    def build_artifact_manifest(self) -> Any:
        """Build the artifact manifest declaration for this kit."""

        if self.artifact_manifest_factory is None:
            return None
        return self.artifact_manifest_factory()


def validate_benchmark_kit_entry(entry: BenchmarkKitEntry) -> BenchmarkKitEntry:
    """Validate the v2 benchmark kit registry contract."""

    if entry.config_schema is None:
        raise ValueError("config_schema must be explicitly declared")
    if not isinstance(entry.config_schema, type) or not issubclass(entry.config_schema, BaseModel):
        raise ValueError("config_schema must inherit from pydantic.BaseModel")
    if entry.config_schema.model_config.get("extra") != "forbid":
        raise ValueError("config_schema.extra_forbid is required")
    if entry.default_environment_provider not in entry.default_environment_profile_by_provider:
        raise ValueError("default_environment_provider must have a default profile")
    default_profile = entry.default_environment_profile_by_provider[entry.default_environment_provider]
    if default_profile not in entry.environment_profiles:
        raise ValueError("default_environment_profile is missing from environment_profiles")
    for profile_id, profile in entry.environment_profiles.items():
        if not isinstance(profile, dict) or not profile.get("asset_dir"):
            raise ValueError(f"environment_profiles.{profile_id}.asset_dir is required")
    if entry.verifier_environment_policy not in {"reuse", "fresh_from_profile", "kit_judge"}:
        raise ValueError("verifier_environment_policy must be reuse, fresh_from_profile, or kit_judge")
    if entry.verifier_environment_policy == "reuse" and entry.verifier_environment_profile_id is None:
        pass
    elif entry.verifier_environment_profile_id not in entry.environment_profiles:
        raise ValueError("verifier_environment_profile_id must reference an environment profile")
    for scheduler in entry.supported_schedulers:
        try:
            entry.resolve_workflow_bundle(scheduler)
        except Exception as exc:
            raise ValueError(f"supported_schedulers.{scheduler}.workflow is unavailable") from exc
    if "framework_loop" in entry.supported_schedulers and entry.tool_registry_factory is None:
        raise ValueError("framework_loop.tool_registry is required")
    if entry.verifier_adapter_factory is None:
        raise ValueError("verifier_adapter_factory is required")
    if entry.artifact_manifest_factory is None:
        raise ValueError("artifact_manifest_factory is required")
    return entry


def _minimal_init_environment_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in profile.items() if key not in _INIT_PROFILE_OMIT_KEYS}


def build_agentkit_v2_init_payload(kit_id: str) -> dict[str, Any]:
    """Build a minimal Agent Eval v2 config for a registered benchmark kit."""

    from gage_eval.agent_eval_kits import load_benchmark_kit
    from gage_eval.environment.providers.registry import create_default_provider_registry

    kit = load_benchmark_kit(kit_id)
    provider = kit.default_environment_provider
    provider_registry = create_default_provider_registry()
    if provider not in provider_registry.registered_provider_ids():
        raise ValueError(f"Kit '{kit_id}' references unregistered environment provider '{provider}'")

    profile_id = kit.default_environment_profile_by_provider[provider]
    profile = _minimal_init_environment_profile(dict(kit.environment_profiles[profile_id]))
    base_id = kit.kit_id
    backend_id = f"{base_id}_model"
    agent_id = f"{base_id}_agent"
    benchmark_id = f"{base_id}_benchmark"
    env_id = f"{base_id}_env"

    return {
        "kind": "PipelineConfig",
        "metadata": {"name": f"{base_id}-agent-eval"},
        "backends": [
            {
                "backend_id": backend_id,
                "type": "litellm",
                "config": {
                    "model": "gpt-4.1-mini",
                    "api_key": "${ENV.MODEL_API_KEY}",
                },
            }
        ],
        "agents": [
            {
                "agent_id": agent_id,
                "scheduler": {
                    "type": "framework_loop",
                    "backend_id": backend_id,
                },
                "config": {},
            }
        ],
        "benchmarks": [
            {
                "benchmark_id": benchmark_id,
                "kit_id": kit.kit_id,
                "config": {},
            }
        ],
        "environments": [
            {
                "env_id": env_id,
                "provider": provider,
                "profile_id": profile_id,
                "profile": profile,
            }
        ],
        "dut_agents": [
            {
                "dut_id": f"{base_id}_dut",
                "agent_id": agent_id,
                "env_id": env_id,
                "benchmark_id": benchmark_id,
            }
        ],
    }


def build_environment_resource_plan(
    *,
    runtime_spec: Any,
    default_provider: str,
    default_profile_id: str,
    environment_profiles: Mapping[str, dict[str, Any]],
    environment_profile: Mapping[str, Any] | None = None,
    provider_config: Mapping[str, Any] | None = None,
    resources: Mapping[str, Any] | None = None,
    startup_env: Mapping[str, Any] | None = None,
    lifecycle: str | None = None,
    provider_config_resolver: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build the v2 resource plan consumed directly by EnvironmentManager."""

    requested_profile = dict(environment_profile or {})
    provider = str(
        requested_profile.get("provider")
        or getattr(runtime_spec, "resource_policy", {}).get("resource_kind")
        or default_provider
    )
    profile_id = str(
        requested_profile.get("profile_id")
        or (
            getattr(runtime_spec, "environment_profile_id", None)
            if getattr(runtime_spec, "environment_profile_id", None) in environment_profiles
            else None
        )
        or default_profile_id
    )
    base_profile = dict(environment_profiles.get(profile_id) or {})
    profile_payload = _deep_merge(base_profile, requested_profile)
    profile_payload["profile_id"] = profile_id
    profile_payload["provider"] = provider

    if isinstance(profile_payload.get("provider_config"), Mapping):
        raise ValueError("environment_profile.provider_config.unsupported: use environment_profile.config")
    base_provider_config = {}
    value = profile_payload.get("config")
    if isinstance(value, Mapping):
        base_provider_config = _deep_merge(base_provider_config, dict(value))
    if provider_config:
        base_provider_config = _deep_merge(base_provider_config, dict(provider_config))
    profile_payload["config"] = dict(base_provider_config)
    profile_payload.pop("provider_config", None)

    effective_resources = dict(resources or profile_payload.get("resources") or {})
    effective_startup_env = {
        str(key): str(value)
        for key, value in dict(startup_env or profile_payload.get("startup_env") or {}).items()
    }
    effective_lifecycle = str(
        lifecycle
        or profile_payload.get("lifecycle")
        or getattr(runtime_spec, "resource_policy", {}).get("lifecycle")
        or "per_sample"
    )
    if effective_resources:
        profile_payload["resources"] = dict(effective_resources)
    else:
        profile_payload.pop("resources", None)
    if effective_startup_env:
        profile_payload["startup_env"] = dict(effective_startup_env)
    else:
        profile_payload.pop("startup_env", None)
    profile_payload["lifecycle"] = effective_lifecycle

    plan: dict[str, Any] = {
        "resource_kind": provider,
        "environment_profile": profile_payload,
        "provider_config": dict(base_provider_config),
        "resources": effective_resources,
        "startup_env": effective_startup_env,
        "lifecycle": effective_lifecycle,
        "cleanup_policy": {"mode": "provider_release"},
    }
    if provider_config_resolver is not None:
        plan["provider_config_resolver"] = provider_config_resolver
    return plan


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base or {})
    for key, value in (override or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = _deep_merge(dict(merged[key]), dict(value))
        else:
            merged[key] = value
    return merged


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
