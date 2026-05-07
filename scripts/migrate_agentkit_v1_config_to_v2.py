#!/usr/bin/env python3
"""Minimal one-time migration from legacy AgentKit v1 YAML to v2 YAML."""

from __future__ import annotations

import argparse
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, NamedTuple

import yaml


class MigrationResult(NamedTuple):
    ok: bool
    payload: dict[str, Any] | None
    manual_fixes: list[str]


_SUPPORTED_TOP_LEVEL_KEYS = {
    "api_version",
    "schema_version",
    "runtime_version",
    "kind",
    "metadata",
    "backends",
    "agent_backends",
    "agent_backend_id",
    "scheduler",
    "agents",
    "role_adapters",
}

_FULL_PIPELINE_TOP_LEVEL_KEYS = {
    "api_version",
    "kind",
    "metadata",
    "prompts",
    "datasets",
    "backends",
    "sandbox_profiles",
    "role_adapters",
    "metrics",
    "summary_generators",
    "tasks",
}
_PIPELINE_DATASET_KEYS = {"dataset_id", "hub", "hub_params", "loader", "params"}
_PIPELINE_BACKEND_KEYS = {"backend_id", "type", "config"}
_PIPELINE_SANDBOX_PROFILE_KEYS = {
    "sandbox_id",
    "runtime",
    "resources",
    "runtime_configs",
}
_PIPELINE_DUT_ADAPTER_KEYS = {
    "adapter_id",
    "role_type",
    "backend_id",
    "agent_runtime_id",
    "prompt_id",
    "sandbox",
    "params",
}
_PIPELINE_JUDGE_ADAPTER_KEYS = {
    "adapter_id",
    "role_type",
    "sandbox",
    "params",
}
_PIPELINE_TASK_KEYS = {
    "task_id",
    "dataset_id",
    "steps",
    "max_samples",
    "concurrency",
    "trial_policy",
    "trials",
}
_PIPELINE_TASK_STEP_KEYS = {"step", "adapter_id"}

_ROLE_ADAPTER_KEYS = {
    "adapter_id",
    "role_type",
    "agent_backend_id",
    "agent_runtime_id",
}

_AGENT_BACKEND_KEYS = {
    "agent_backend_id",
    "type",
    "backend_id",
    "config",
}

_BACKEND_KEYS = {
    "backend_id",
    "type",
    "config",
}

_AGENT_KEYS = {
    "agent_id",
    "scheduler",
    "config",
}

_SCHEDULER_KEYS = {
    "type",
    "backend_id",
    "model",
}

_SUPPORTED_SCHEDULER_TYPES = {"framework_loop", "installed_client"}
_SUPPORTED_AGENT_BACKEND_TYPE = "model_backend"
_KNOWN_AGENT_JUDGE_IMPLEMENTATIONS = {
    "appworld_evaluate",
    "swebench_docker",
    "tau2_eval",
}
_KNOWN_RUNTIME_SCHEDULER_TYPES = {
    "framework_loop": "framework_loop",
    "installed_client": "installed_client",
    "tau2_framework_loop": "framework_loop",
    "swebench_framework_loop": "framework_loop",
    "terminal_bench_framework_loop": "framework_loop",
    "appworld_framework_loop": "framework_loop",
    "tau2_installed_client": "installed_client",
    "swebench_installed_client": "installed_client",
    "terminal_bench_installed_client": "installed_client",
    "appworld_installed_client": "installed_client",
}


def migrate_file(input_path: Path, output_path: Path) -> MigrationResult:
    """Migrate a YAML file and write the v2 payload when migration succeeds."""

    if input_path.resolve() == output_path.resolve():
        return MigrationResult(
            ok=False,
            payload=None,
            manual_fixes=["manual migration required: input and output paths must be different"],
        )

    with input_path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        return MigrationResult(
            ok=False,
            payload=None,
            manual_fixes=[f"{input_path}: expected a YAML mapping at the top level"],
        )

    result = migrate_payload(payload)
    if result.ok and result.payload is not None:
        _write_yaml_atomic(output_path, result.payload)
    return result


def migrate_payload(payload: dict[str, Any]) -> MigrationResult:
    """Return a minimal v2 payload or manual-fix guidance for unsupported v1 fields."""

    if _is_full_pipeline_config(payload):
        return _migrate_pipeline_config_payload(payload)

    manual_fixes = _manual_fix_messages(payload)
    if manual_fixes:
        return MigrationResult(ok=False, payload=None, manual_fixes=manual_fixes)

    static_backends = _index_static_backends(payload.get("backends"))
    backend_specs: dict[str, dict[str, Any]] = {}
    agent_backend_to_backend = _migrate_agent_backends(
        payload.get("agent_backends"),
        static_backends,
        backend_specs,
    )

    agents = _migrate_role_adapter_agents(payload.get("role_adapters"), agent_backend_to_backend)
    agents.extend(_migrate_legacy_agents(payload.get("agents"), backend_specs))
    top_level_agent = _migrate_top_level_agent(payload, agent_backend_to_backend, backend_specs)
    if top_level_agent is not None:
        agents.append(top_level_agent)

    if not agents:
        return MigrationResult(
            ok=False,
            payload=None,
            manual_fixes=["manual migration required: no DUT agent could be derived from role_adapters or agents"],
        )
    if not backend_specs:
        return MigrationResult(
            ok=False,
            payload=None,
            manual_fixes=["manual migration required: no backend could be derived from agent_backends or scheduler.model"],
        )
    reference_messages = _agent_backend_reference_manual_fix_messages(agents, backend_specs)
    if reference_messages:
        return MigrationResult(ok=False, payload=None, manual_fixes=reference_messages)

    kit = _infer_placeholder_kit(payload)
    benchmark_id = f"{kit['kit_id']}_benchmark"
    env_id = f"{kit['kit_id']}_env"
    used_dut_ids: set[str] = set()

    migrated = {
        "kind": "AgentEvalConfig",
        "metadata": _metadata(payload),
        "backends": list(backend_specs.values()),
        "agents": agents,
        "benchmarks": [
            {
                "benchmark_id": benchmark_id,
                "kit_id": kit["kit_id"],
                "config": kit["benchmark_config"],
            }
        ],
        "environments": [
            {
                "env_id": env_id,
                "provider": kit["provider"],
                "profile_id": kit["profile_id"],
                "profile": {"asset_dir": kit["asset_dir"]},
            }
        ],
        "dut_agents": [
            {
                "dut_id": _unique_id(f"{agent['agent_id']}_dut", used_dut_ids),
                "agent_id": agent["agent_id"],
                "env_id": env_id,
                "benchmark_id": benchmark_id,
            }
            for agent in agents
        ],
    }
    sanity_messages = _final_sanity_manual_fix_messages(migrated)
    if sanity_messages:
        return MigrationResult(ok=False, payload=None, manual_fixes=sanity_messages)
    loader_messages = _loader_validation_manual_fix_messages(migrated)
    if loader_messages:
        return MigrationResult(ok=False, payload=None, manual_fixes=loader_messages)
    return MigrationResult(ok=True, payload=migrated, manual_fixes=[])


def _is_full_pipeline_config(payload: dict[str, Any]) -> bool:
    if "agent_backends" in payload or "agent_backend_id" in payload:
        return False
    return any(
        key in payload
        for key in (
            "datasets",
            "sandbox_profiles",
            "tasks",
            "metrics",
            "prompts",
            "summary_generators",
        )
    )


def _migrate_pipeline_config_payload(payload: dict[str, Any]) -> MigrationResult:
    manual_fixes = _pipeline_manual_fix_messages(payload)
    if manual_fixes:
        return MigrationResult(ok=False, payload=None, manual_fixes=manual_fixes)

    datasets = _index_by_id(payload.get("datasets"), "dataset_id")
    backends = _index_by_id(payload.get("backends"), "backend_id")
    sandbox_profiles = _index_by_id(payload.get("sandbox_profiles"), "sandbox_id")
    role_adapters = _index_by_id(payload.get("role_adapters"), "adapter_id")
    tasks = [task for task in payload.get("tasks") or [] if isinstance(task, dict)]

    migrated_backends = [
        {
            "backend_id": str(backend["backend_id"]),
            "type": str(backend["type"]),
            "config": deepcopy(backend.get("config") or {}),
        }
        for backend in payload.get("backends") or []
        if isinstance(backend, dict) and isinstance(backend.get("backend_id"), str)
    ]
    agents: list[dict[str, Any]] = []
    benchmarks: list[dict[str, Any]] = []
    environments: list[dict[str, Any]] = []
    dut_agents: list[dict[str, Any]] = []
    used_env_ids: set[str] = set()
    used_agent_ids: set[str] = set()
    used_dut_ids: set[str] = set()

    for task in tasks:
        inference_adapter_id = _first_task_step_adapter_id(task, "inference")
        if inference_adapter_id is None:
            continue
        adapter = role_adapters[str(inference_adapter_id)]
        dataset = datasets[str(task["dataset_id"])]
        sandbox_id = _adapter_sandbox_id(adapter)
        sandbox_profile = sandbox_profiles[str(sandbox_id)]
        kit_id = _infer_pipeline_kit_id(payload=payload, task=task, dataset=dataset, adapter=adapter)
        agent_id = _unique_id(_sanitize_id(str(adapter["adapter_id"])), used_agent_ids)
        scheduler_type = _scheduler_type_from_runtime(adapter.get("agent_runtime_id")) or "framework_loop"
        scheduler: dict[str, Any] = {"type": scheduler_type}
        backend_id = adapter.get("backend_id")
        if isinstance(backend_id, str):
            scheduler["backend_id"] = backend_id
        scheduler_config = _scheduler_config_from_adapter(adapter)
        if scheduler_config:
            scheduler["config"] = scheduler_config
        agents.append(
            {
                "agent_id": agent_id,
                "scheduler": scheduler,
                "config": _agent_config_from_pipeline_adapter(adapter),
            }
        )

        benchmark_id = str(task["task_id"])
        benchmarks.append(
            {
                "benchmark_id": benchmark_id,
                "kit_id": kit_id,
                "config": _benchmark_config_from_pipeline(
                    kit_id=kit_id,
                    dataset=dataset,
                    sandbox_profile=sandbox_profile,
                ),
            }
        )
        if sandbox_id not in used_env_ids:
            environments.append(
                _environment_from_pipeline_sandbox(
                    sandbox_profile=sandbox_profile,
                    adapters=payload.get("role_adapters") or [],
                    kit_id=kit_id,
                )
            )
            used_env_ids.add(str(sandbox_id))
        dut: dict[str, Any] = {
            "dut_id": _unique_id(f"{agent_id}_dut", used_dut_ids),
            "agent_id": agent_id,
            "env_id": str(sandbox_id),
            "benchmark_id": benchmark_id,
        }
        trial_policy = _trial_policy_from_pipeline_task(task=task, dataset=dataset)
        if trial_policy:
            dut["trial_policy"] = trial_policy
        dut_agents.append(dut)

    migrated = {
        "kind": "AgentEvalConfig",
        "metadata": _pipeline_metadata(payload),
        "backends": migrated_backends,
        "agents": agents,
        "benchmarks": benchmarks,
        "environments": environments,
        "dut_agents": dut_agents,
        "trial_policy": {"trials": 1},
    }
    sanity_messages = _final_sanity_manual_fix_messages(migrated)
    if sanity_messages:
        return MigrationResult(ok=False, payload=None, manual_fixes=sanity_messages)
    loader_messages = _loader_validation_manual_fix_messages(migrated)
    if loader_messages:
        return MigrationResult(ok=False, payload=None, manual_fixes=loader_messages)
    return MigrationResult(ok=True, payload=migrated, manual_fixes=[])


def _pipeline_manual_fix_messages(payload: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    unsupported = _unsupported_keys(payload, _FULL_PIPELINE_TOP_LEVEL_KEYS)
    if unsupported:
        messages.append(
            "manual migration required: unsupported PipelineConfig top-level fields "
            + _format_key_list(unsupported)
        )

    messages.extend(_pipeline_list_section_messages(payload, "datasets", "dataset_id"))
    messages.extend(_pipeline_list_section_messages(payload, "backends", "backend_id"))
    messages.extend(_pipeline_list_section_messages(payload, "sandbox_profiles", "sandbox_id"))
    messages.extend(_pipeline_list_section_messages(payload, "role_adapters", "adapter_id"))
    messages.extend(_pipeline_list_section_messages(payload, "tasks", "task_id"))
    if messages:
        return messages

    datasets = _index_by_id(payload.get("datasets"), "dataset_id")
    backends = _index_by_id(payload.get("backends"), "backend_id")
    sandbox_profiles = _index_by_id(payload.get("sandbox_profiles"), "sandbox_id")
    role_adapters = _index_by_id(payload.get("role_adapters"), "adapter_id")
    messages.extend(_pipeline_unsupported_nested_field_messages(payload))

    if not payload.get("tasks"):
        messages.append("manual migration required: PipelineConfig tasks must not be empty")
    if not any(
        isinstance(adapter, dict) and adapter.get("role_type") == "dut_agent"
        for adapter in payload.get("role_adapters") or []
    ):
        messages.append("manual migration required: PipelineConfig has no dut_agent role_adapter")

    for index, backend in enumerate(payload.get("backends") or []):
        if not isinstance(backend, dict):
            continue
        backend_type = backend.get("type")
        if not isinstance(backend_type, str):
            messages.append(f"manual migration required: backends[{index}].type must be a string")
        config = backend.get("config")
        if config is not None and not isinstance(config, dict):
            messages.append(f"manual migration required: backends[{index}].config must be a mapping")

    for index, adapter in enumerate(payload.get("role_adapters") or []):
        if not isinstance(adapter, dict):
            continue
        role_type = adapter.get("role_type")
        if role_type == "dut_agent":
            runtime_id = adapter.get("agent_runtime_id")
            if _scheduler_type_from_runtime(runtime_id) is None:
                messages.append(
                    f"manual migration required: role_adapters[{index}] "
                    f"agent_runtime_id={runtime_id} is not supported"
                )
            backend_id = adapter.get("backend_id")
            if not isinstance(backend_id, str):
                messages.append(
                    f"manual migration required: role_adapters[{index}] dut_agent must have string backend_id"
                )
            elif backend_id not in backends:
                messages.append(
                    f"manual migration required: role_adapters[{index}] backend_id={backend_id} is unknown"
                )
            sandbox_id = _adapter_sandbox_id(adapter)
            if not isinstance(sandbox_id, str):
                messages.append(
                    f"manual migration required: role_adapters[{index}] dut_agent must have sandbox.sandbox_id"
                )
            elif sandbox_id not in sandbox_profiles:
                messages.append(
                    f"manual migration required: role_adapters[{index}] sandbox_id={sandbox_id} is unknown"
                )
            lifecycle = _adapter_sandbox_lifecycle(adapter)
            if lifecycle is not None and lifecycle != "per_sample":
                messages.append(
                    f"manual migration required: role_adapters[{index}] sandbox lifecycle={lifecycle} is not supported"
                )
            continue

        if role_type == "judge_extend":
            params = adapter.get("params")
            if not isinstance(params, dict):
                messages.append(
                    f"manual migration required: role_adapters[{index}] judge_extend params must be a mapping"
                )
                continue
            implementation = params.get("implementation")
            if implementation not in _KNOWN_AGENT_JUDGE_IMPLEMENTATIONS:
                messages.append(
                    f"manual migration required: role_adapters[{index}] "
                    f"judge implementation={implementation} is not a known AgentKit v2 verifier"
                )
            continue

        messages.append(
            f"manual migration required: role_adapters[{index}] role_type={role_type!r} is not supported"
        )

    for task_index, task in enumerate(payload.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        dataset_id = task.get("dataset_id")
        if not isinstance(dataset_id, str) or dataset_id not in datasets:
            messages.append(
                f"manual migration required: tasks[{task_index}] dataset_id={dataset_id} is unknown"
            )
        inference_adapter_id = _first_task_step_adapter_id(task, "inference")
        if not isinstance(inference_adapter_id, str):
            messages.append(
                f"manual migration required: tasks[{task_index}] has no inference step adapter_id"
            )
        elif inference_adapter_id not in role_adapters:
            messages.append(
                f"manual migration required: tasks[{task_index}] inference adapter_id={inference_adapter_id} is unknown"
            )
        elif role_adapters[inference_adapter_id].get("role_type") != "dut_agent":
            messages.append(
                f"manual migration required: tasks[{task_index}] inference adapter_id={inference_adapter_id} "
                "does not reference a dut_agent"
            )
    return messages


def _pipeline_unsupported_nested_field_messages(payload: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    for index, dataset in enumerate(payload.get("datasets") or []):
        if isinstance(dataset, dict):
            messages.extend(
                _unsupported_nested_messages(
                    dataset,
                    _PIPELINE_DATASET_KEYS,
                    f"datasets[{index}]",
                )
            )
    for index, backend in enumerate(payload.get("backends") or []):
        if isinstance(backend, dict):
            messages.extend(
                _unsupported_nested_messages(
                    backend,
                    _PIPELINE_BACKEND_KEYS,
                    f"backends[{index}]",
                )
            )
    for index, profile in enumerate(payload.get("sandbox_profiles") or []):
        if isinstance(profile, dict):
            messages.extend(
                _unsupported_nested_messages(
                    profile,
                    _PIPELINE_SANDBOX_PROFILE_KEYS,
                    f"sandbox_profiles[{index}]",
                )
            )
    for index, adapter in enumerate(payload.get("role_adapters") or []):
        if not isinstance(adapter, dict):
            continue
        role_type = adapter.get("role_type")
        allowed = (
            _PIPELINE_DUT_ADAPTER_KEYS
            if role_type == "dut_agent"
            else _PIPELINE_JUDGE_ADAPTER_KEYS
            if role_type == "judge_extend"
            else {"adapter_id", "role_type"}
        )
        messages.extend(
            _unsupported_nested_messages(
                adapter,
                allowed,
                f"role_adapters[{index}]",
            )
        )
        sandbox = adapter.get("sandbox")
        if isinstance(sandbox, dict):
            messages.extend(
                _unsupported_nested_messages(
                    sandbox,
                    {"sandbox_id", "lifecycle"},
                    f"role_adapters[{index}].sandbox",
                )
            )
    for index, task in enumerate(payload.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        messages.extend(
            _unsupported_nested_messages(
                task,
                _PIPELINE_TASK_KEYS,
                f"tasks[{index}]",
            )
        )
        steps = task.get("steps")
        if isinstance(steps, list):
            for step_index, step in enumerate(steps):
                if isinstance(step, dict):
                    messages.extend(
                        _unsupported_nested_messages(
                            step,
                            _PIPELINE_TASK_STEP_KEYS,
                            f"tasks[{index}].steps[{step_index}]",
                        )
                    )
    return messages


def _unsupported_nested_messages(
    mapping: dict[Any, Any],
    allowed: set[str],
    path: str,
) -> list[str]:
    unsupported = _unsupported_keys(mapping, allowed)
    if not unsupported:
        return []
    return [
        f"manual migration required: {path} contains unsupported fields "
        + _format_key_list(unsupported)
    ]


def _pipeline_list_section_messages(
    payload: dict[str, Any],
    section: str,
    id_field: str,
) -> list[str]:
    value = payload.get(section)
    if not isinstance(value, list) or not value:
        return [f"manual migration required: PipelineConfig {section} must be a non-empty list"]
    messages: list[str] = []
    seen: dict[str, int] = {}
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            messages.append(f"manual migration required: {section}[{index}] must be a mapping")
            continue
        item_id = item.get(id_field)
        if not isinstance(item_id, str) or not item_id:
            messages.append(f"manual migration required: {section}[{index}].{id_field} must be a string")
            continue
        previous = seen.get(item_id)
        if previous is not None:
            messages.append(
                f"manual migration required: duplicate {section}.{id_field}={item_id} at entries {previous} and {index}"
            )
        seen[item_id] = index
    return messages


def _index_by_id(value: Any, id_field: str) -> dict[str, dict[str, Any]]:
    if not isinstance(value, list):
        return {}
    return {
        str(item[id_field]): item
        for item in value
        if isinstance(item, dict) and isinstance(item.get(id_field), str)
    }


def _first_task_step_adapter_id(task: dict[str, Any], step_name: str) -> str | None:
    steps = task.get("steps")
    if not isinstance(steps, list):
        return None
    for step in steps:
        if not isinstance(step, dict):
            continue
        if step.get("step") == step_name and isinstance(step.get("adapter_id"), str):
            return str(step["adapter_id"])
    return None


def _adapter_sandbox_id(adapter: dict[str, Any]) -> str | None:
    sandbox = adapter.get("sandbox")
    if isinstance(sandbox, dict) and isinstance(sandbox.get("sandbox_id"), str):
        return str(sandbox["sandbox_id"])
    return None


def _adapter_sandbox_lifecycle(adapter: dict[str, Any]) -> str | None:
    sandbox = adapter.get("sandbox")
    if isinstance(sandbox, dict) and sandbox.get("lifecycle") is not None:
        return str(sandbox["lifecycle"])
    return None


def _infer_pipeline_kit_id(
    *,
    payload: dict[str, Any],
    task: dict[str, Any],
    dataset: dict[str, Any],
    adapter: dict[str, Any],
) -> str:
    haystack = yaml.safe_dump(
        {
            "metadata": payload.get("metadata"),
            "task": task,
            "dataset": dataset,
            "adapter": adapter,
        },
        sort_keys=True,
    ).lower()
    if "swebench" in haystack or "swe-bench" in haystack:
        return "swebench"
    if "appworld" in haystack:
        return "appworld"
    if "terminal_bench" in haystack or "terminal-bench" in haystack:
        return "terminal_bench"
    return "tau2"


def _benchmark_config_from_pipeline(
    *,
    kit_id: str,
    dataset: dict[str, Any],
    sandbox_profile: dict[str, Any],
) -> dict[str, Any]:
    params = dataset.get("params") if isinstance(dataset.get("params"), dict) else {}
    hub_params = dataset.get("hub_params") if isinstance(dataset.get("hub_params"), dict) else {}
    runtime_configs = (
        sandbox_profile.get("runtime_configs")
        if isinstance(sandbox_profile.get("runtime_configs"), dict)
        else {}
    )
    if kit_id == "swebench":
        split = hub_params.get("split") or params.get("split")
        return {"split": split} if isinstance(split, str) else {}
    if kit_id == "tau2":
        config: dict[str, Any] = {}
        domain = params.get("domain")
        if isinstance(domain, str):
            config["domain"] = domain
        user_model = runtime_configs.get("user_model")
        user_model_args = runtime_configs.get("user_model_args")
        if user_model is not None or user_model_args is not None:
            config["user_simulator"] = {
                "model": user_model,
                "model_args": deepcopy(user_model_args) if isinstance(user_model_args, dict) else {},
            }
        return config
    return {}


def _scheduler_config_from_adapter(adapter: dict[str, Any]) -> dict[str, Any]:
    params = adapter.get("params")
    if not isinstance(params, dict):
        return {}
    config: dict[str, Any] = {}
    if "max_turns" in params:
        config["max_turns"] = deepcopy(params["max_turns"])
    return config


def _agent_config_from_pipeline_adapter(adapter: dict[str, Any]) -> dict[str, Any]:
    config: dict[str, Any] = {}
    prompt_id = adapter.get("prompt_id")
    if isinstance(prompt_id, str):
        config["prompt_id"] = prompt_id
    params = adapter.get("params")
    if isinstance(params, dict):
        legacy_params = {key: deepcopy(value) for key, value in params.items() if key != "max_turns"}
        if legacy_params:
            config["legacy_params"] = legacy_params
    return config


def _environment_from_pipeline_sandbox(
    *,
    sandbox_profile: dict[str, Any],
    adapters: list[Any],
    kit_id: str,
) -> dict[str, Any]:
    sandbox_id = str(sandbox_profile["sandbox_id"])
    runtime = str(sandbox_profile.get("runtime") or "")
    runtime_configs = (
        deepcopy(sandbox_profile.get("runtime_configs"))
        if isinstance(sandbox_profile.get("runtime_configs"), dict)
        else {}
    )
    if kit_id == "tau2":
        runtime_configs.pop("user_model", None)
        runtime_configs.pop("user_model_args", None)
    env: dict[str, Any] = {
        "env_id": sandbox_id,
        "provider": _environment_provider_from_legacy_runtime(runtime),
        "profile_id": sandbox_id,
        "profile": {},
        "lifecycle": _lifecycle_for_sandbox_id(sandbox_id, adapters),
    }
    if runtime_configs:
        env["provider_config"] = runtime_configs
    resources = sandbox_profile.get("resources")
    if isinstance(resources, dict) and resources:
        env["resources"] = _environment_resources_from_legacy(resources)
    return env


def _environment_provider_from_legacy_runtime(runtime: str) -> str:
    if runtime in {"tau2", "local", "local_process", "process"}:
        return "local_process"
    if runtime in {"docker", "docker_runtime"}:
        return "docker"
    if runtime == "e2b":
        return "e2b"
    return runtime or "local_process"


def _environment_resources_from_legacy(resources: dict[str, Any]) -> dict[str, Any]:
    migrated = deepcopy(resources)
    memory = migrated.pop("memory", None)
    if memory is not None:
        migrated["memory_gb"] = _memory_to_gb(memory)
    return migrated


def _memory_to_gb(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise ValueError(f"unsupported memory resource value: {value!r}")
    text = value.strip().lower()
    if text.endswith("gb"):
        return float(text[:-2].strip())
    if text.endswith("g"):
        return float(text[:-1].strip())
    if text.endswith("mb"):
        return float(text[:-2].strip()) / 1024.0
    if text.endswith("m"):
        return float(text[:-1].strip()) / 1024.0
    return float(text)


def _lifecycle_for_sandbox_id(sandbox_id: str, adapters: list[Any]) -> str:
    for adapter in adapters:
        if not isinstance(adapter, dict):
            continue
        if _adapter_sandbox_id(adapter) == sandbox_id:
            return _adapter_sandbox_lifecycle(adapter) or "per_sample"
    return "per_sample"


def _trial_policy_from_pipeline_task(
    *,
    task: dict[str, Any],
    dataset: dict[str, Any],
) -> dict[str, Any] | None:
    params = dataset.get("params") if isinstance(dataset.get("params"), dict) else {}
    if "num_trials" in params:
        return {"trials": deepcopy(params["num_trials"])}
    trial_policy = task.get("trial_policy")
    if isinstance(trial_policy, dict) and "trials" in trial_policy:
        return {"trials": deepcopy(trial_policy["trials"])}
    if "trials" in task:
        return {"trials": deepcopy(task["trials"])}
    return None


def _pipeline_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = _metadata(payload)
    migration: dict[str, Any] = {
        "source_kind": payload.get("kind"),
        "source_api_version": payload.get("api_version"),
    }
    preserved_sections: dict[str, Any] = {}
    for section in (
        "prompts",
        "datasets",
        "sandbox_profiles",
        "role_adapters",
        "metrics",
        "summary_generators",
        "tasks",
    ):
        value = payload.get(section)
        if value:
            preserved_sections[section] = deepcopy(value)
    if preserved_sections:
        migration["preserved_legacy_sections"] = preserved_sections
    legacy_judge_adapters = _legacy_judge_adapters(payload.get("role_adapters"))
    if legacy_judge_adapters:
        migration["legacy_judge_adapters"] = legacy_judge_adapters
    runtime_unapplied_fields = _pipeline_runtime_unapplied_fields(payload)
    if runtime_unapplied_fields:
        migration["manual_review_required"] = True
        migration["runtime_unapplied_fields"] = runtime_unapplied_fields
    metadata["migration"] = migration
    return metadata


def _pipeline_runtime_unapplied_fields(payload: dict[str, Any]) -> list[str]:
    fields: set[str] = set()
    for index, dataset in enumerate(payload.get("datasets") or []):
        if not isinstance(dataset, dict):
            continue
        for key in ("hub", "loader"):
            if key in dataset:
                fields.add(f"datasets[{index}].{key}")
        hub_params = dataset.get("hub_params")
        if isinstance(hub_params, dict):
            for key in sorted(hub_params):
                if key != "split":
                    fields.add(f"datasets[{index}].hub_params.{key}")
        params = dataset.get("params")
        if isinstance(params, dict):
            for key in sorted(params):
                if key in {"domain", "num_trials"}:
                    continue
                fields.add(f"datasets[{index}].params.{key}")
            preprocess_kwargs = params.get("preprocess_kwargs")
            if isinstance(preprocess_kwargs, dict):
                for key in sorted(preprocess_kwargs):
                    fields.add(f"datasets[{index}].params.preprocess_kwargs.{key}")
    for index, adapter in enumerate(payload.get("role_adapters") or []):
        if not isinstance(adapter, dict):
            continue
        if adapter.get("role_type") == "judge_extend":
            fields.add(f"role_adapters[{index}]")
        params = adapter.get("params")
        if isinstance(params, dict):
            for key in sorted(params):
                if adapter.get("role_type") == "dut_agent" and key == "max_turns":
                    continue
                fields.add(f"role_adapters[{index}].params.{key}")
    for index, task in enumerate(payload.get("tasks") or []):
        if not isinstance(task, dict):
            continue
        for key in ("max_samples", "concurrency"):
            if key in task:
                fields.add(f"tasks[{index}].{key}")
        steps = task.get("steps")
        if isinstance(steps, list):
            for step_index, step in enumerate(steps):
                if isinstance(step, dict) and step.get("step") != "inference":
                    fields.add(f"tasks[{index}].steps[{step_index}]")
    return sorted(fields)


def _legacy_judge_adapters(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    adapters: list[dict[str, Any]] = []
    for adapter in value:
        if not isinstance(adapter, dict) or adapter.get("role_type") != "judge_extend":
            continue
        params = adapter.get("params") if isinstance(adapter.get("params"), dict) else {}
        adapters.append(
            {
                "adapter_id": adapter.get("adapter_id"),
                "implementation": params.get("implementation"),
                "implementation_params": deepcopy(params.get("implementation_params") or {}),
            }
        )
    return adapters


def _manual_fix_messages(payload: dict[str, Any]) -> list[str]:
    unsupported = _unsupported_keys(payload, _SUPPORTED_TOP_LEVEL_KEYS)
    messages = (
        [
            "manual migration required: unsupported top-level fields "
            + _format_key_list(unsupported)
            + ". This tool only migrates agent_backends, agent_backend_id, and scheduler.model "
            "into v2 backends/agents placeholders."
        ]
        if unsupported
        else []
    )

    static_backends = _index_static_backends(payload.get("backends"))
    referenced_backend_ids = _referenced_static_backend_ids(payload.get("agent_backends"))
    messages.extend(_mixed_role_adapter_top_level_manual_fix_messages(payload))
    messages.extend(_metadata_manual_fix_messages(payload.get("metadata")))
    messages.extend(_backend_manual_fix_messages(payload.get("backends"), referenced_backend_ids))
    messages.extend(_agent_backend_manual_fix_messages(payload.get("agent_backends"), static_backends))
    messages.extend(_legacy_agent_manual_fix_messages(payload.get("agents")))
    agent_backend_to_backend = _migrate_agent_backends(payload.get("agent_backends"), static_backends, {})
    messages.extend(_top_level_scheduler_manual_fix_messages(payload, agent_backend_to_backend))
    messages.extend(_top_level_agent_backend_manual_fix_messages(payload, agent_backend_to_backend))
    messages.extend(_role_adapter_manual_fix_messages(payload.get("role_adapters"), agent_backend_to_backend))
    return messages


def _mixed_role_adapter_top_level_manual_fix_messages(payload: dict[str, Any]) -> list[str]:
    role_adapters = payload.get("role_adapters")
    if not isinstance(role_adapters, list) or not role_adapters:
        return []

    mixed_fields: list[str] = []
    if "agent_backend_id" in payload:
        mixed_fields.append("top-level agent_backend_id")
    if "scheduler" in payload:
        mixed_fields.append("top-level scheduler")
    if not mixed_fields:
        return []

    return [
        "manual migration required: role_adapters cannot be mixed with "
        + " and ".join(mixed_fields)
        + " in this minimal migrator; split the config or migrate the agent mapping manually"
    ]


def _metadata_manual_fix_messages(value: Any) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, dict):
        return ["manual migration required: metadata must be a mapping"]
    return _non_string_key_messages(value, "metadata")


def _backend_manual_fix_messages(value: Any, referenced_backend_ids: set[str]) -> list[str]:
    messages: list[str] = []
    if value is None:
        return messages
    if not isinstance(value, list):
        return ["manual migration required: backends must be a list"]

    for index, backend in enumerate(value):
        if not isinstance(backend, dict):
            messages.append(f"manual migration required: backends[{index}] must be a mapping")
            continue
        backend_id = backend.get("backend_id")
        if not isinstance(backend_id, str):
            messages.append(f"manual migration required: backends[{index}] must have string backend_id")
        elif backend_id not in referenced_backend_ids:
            messages.append(
                f"manual migration required: backends[{index}].backend_id={backend_id} "
                "is not referenced by agent_backends"
            )
        backend_type = backend.get("type")
        if not isinstance(backend_type, str):
            messages.append(f"manual migration required: backends[{index}].type must be a string")
        extra_keys = _unsupported_keys(backend, _BACKEND_KEYS)
        if extra_keys:
            messages.append(
                f"manual migration required: backends[{index}] contains unsupported fields "
                + _format_key_list(extra_keys)
            )
        config = backend.get("config")
        if config is not None and not isinstance(config, dict):
            messages.append(f"manual migration required: backends[{index}].config must be a mapping")
        elif isinstance(config, dict):
            messages.extend(_non_string_key_messages(config, f"backends[{index}].config"))
    messages.extend(_duplicate_raw_id_messages("backends", "backend_id", value))
    return messages


def _referenced_static_backend_ids(value: Any) -> set[str]:
    if not isinstance(value, list):
        return set()
    return {
        item["backend_id"]
        for item in value
        if isinstance(item, dict) and isinstance(item.get("backend_id"), str)
    }


def _agent_backend_manual_fix_messages(
    value: Any,
    static_backends: dict[str, dict[str, Any]],
) -> list[str]:
    messages: list[str] = []
    if value is None:
        return messages
    if not isinstance(value, list):
        return ["manual migration required: agent_backends must be a list"]

    sanitized_ids: dict[str, str] = {}
    for index, agent_backend in enumerate(value):
        if not isinstance(agent_backend, dict):
            messages.append(f"manual migration required: agent_backends[{index}] must be a mapping")
            continue
        extra_keys = _unsupported_keys(agent_backend, _AGENT_BACKEND_KEYS)
        if extra_keys:
            messages.append(
                f"manual migration required: agent_backends[{index}] contains unsupported fields "
                + _format_key_list(extra_keys)
            )

        agent_backend_id = agent_backend.get("agent_backend_id")
        if not isinstance(agent_backend_id, str):
            messages.append(f"manual migration required: agent_backends[{index}] must have string agent_backend_id")
        else:
            sanitized = _sanitize_id(agent_backend_id)
            previous = sanitized_ids.get(sanitized)
            if previous is not None and previous != agent_backend_id:
                messages.append(
                    "manual migration required: sanitized backend_id collision for agent_backends "
                    f"{previous!r} and {agent_backend_id!r} both produce {sanitized!r}"
                )
            sanitized_ids.setdefault(sanitized, agent_backend_id)

        agent_backend_type = agent_backend.get("type")
        if agent_backend_type != _SUPPORTED_AGENT_BACKEND_TYPE:
            messages.append(
                f"manual migration required: agent_backends[{index}].type={agent_backend_type} "
                f"is not supported; expected {_SUPPORTED_AGENT_BACKEND_TYPE}"
            )

        config = agent_backend.get("config")
        if config is not None and not isinstance(config, dict):
            messages.append(f"manual migration required: agent_backends[{index}].config must be a mapping")
        elif config:
            messages.append(
                f"manual migration required: agent_backends[{index}].config contains unsupported fields "
                + _format_key_list(config)
                + " because agent backend config cannot be represented in v2"
            )

        backend_id = agent_backend.get("backend_id")
        if not isinstance(backend_id, str):
            messages.append(
                f"manual migration required: agent_backends[{index}] agent_backend_id={agent_backend_id!r} "
                "must reference an existing static backend with backend_id"
            )
        elif backend_id not in static_backends:
            messages.append(
                f"manual migration required: agent_backends[{index}] agent_backend_id={agent_backend_id!r} "
                f"references unknown backend_id={backend_id}"
            )
    messages.extend(_duplicate_raw_id_messages("agent_backends", "agent_backend_id", value))
    return messages


def _final_sanity_manual_fix_messages(payload: dict[str, Any]) -> list[str]:
    messages: list[str] = []
    messages.extend(_duplicate_id_messages("backends", "backend_id", payload.get("backends")))
    messages.extend(_duplicate_id_messages("agents", "agent_id", payload.get("agents")))
    messages.extend(_duplicate_id_messages("dut_agents", "dut_id", payload.get("dut_agents")))

    backend_ids = {
        item.get("backend_id")
        for item in payload.get("backends") or []
        if isinstance(item, dict) and isinstance(item.get("backend_id"), str)
    }
    for agent in payload.get("agents") or []:
        if not isinstance(agent, dict):
            continue
        scheduler = agent.get("scheduler")
        if not isinstance(scheduler, dict):
            continue
        scheduler_type = scheduler.get("type")
        if scheduler_type not in _SUPPORTED_SCHEDULER_TYPES:
            messages.append(
                "manual migration required: final sanity check found unsupported "
                f"agents.{agent.get('agent_id')}.scheduler.type={scheduler_type}"
            )
        backend_id = scheduler.get("backend_id")
        if isinstance(backend_id, str) and backend_id not in backend_ids:
            messages.append(
                "manual migration required: final sanity check found "
                f"agents.{agent.get('agent_id')}.scheduler.backend_id={backend_id} without a generated backend"
            )
    return messages


def _loader_validation_manual_fix_messages(payload: dict[str, Any]) -> list[str]:
    try:
        src_path = Path(__file__).resolve().parents[1] / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from gage_eval.config.agentkit_v2 import materialize_agentkit_v2_config_payload

        materialize_agentkit_v2_config_payload(deepcopy(payload), source_path=None)
    except Exception as exc:
        return [f"manual migration required: loader validation failed: {exc}"]
    return []


def _duplicate_raw_id_messages(section: str, id_field: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: dict[str, int] = {}
    messages: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        item_id = item.get(id_field)
        if not isinstance(item_id, str):
            continue
        previous = seen.get(item_id)
        if previous is not None:
            messages.append(
                "manual migration required: duplicate "
                f"{section}.{id_field}={item_id} at entries {previous} and {index}"
            )
        seen[item_id] = index
    return messages


def _non_string_key_messages(value: Any, path: str) -> list[str]:
    messages: list[str] = []

    def visit(item: Any, item_path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    messages.append(
                        f"manual migration required: {item_path} contains non-string key {key!r}"
                    )
                    child_path = item_path
                else:
                    child_path = f"{item_path}.{key}"
                visit(child, child_path)
            return
        if isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{item_path}[{index}]")

    visit(value, path)
    return messages


def _unsupported_keys(mapping: dict[Any, Any], supported_keys: set[str]) -> list[Any]:
    return sorted(
        (key for key in mapping if key not in supported_keys),
        key=_format_key,
    )


def _format_key_list(keys: Any) -> str:
    return ", ".join(_format_key(key) for key in keys)


def _format_key(key: Any) -> str:
    return key if isinstance(key, str) else repr(key)


def _duplicate_id_messages(section: str, id_field: str, value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    seen: dict[str, int] = {}
    messages: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            continue
        item_id = item.get(id_field)
        if not isinstance(item_id, str):
            continue
        previous = seen.get(item_id)
        if previous is not None:
            messages.append(
                "manual migration required: final sanity check found duplicate "
                f"{section}.{id_field}={item_id} at entries {previous} and {index}"
            )
        seen[item_id] = index
    return messages


def _legacy_agent_manual_fix_messages(value: Any) -> list[str]:
    messages: list[str] = []
    if value is None:
        return messages
    if not isinstance(value, list):
        return ["manual migration required: agents must be a list"]

    for index, agent in enumerate(value):
        if not isinstance(agent, dict):
            messages.append(f"manual migration required: agents[{index}] must be a mapping")
            continue
        agent_id = agent.get("agent_id")
        if agent_id is not None and not isinstance(agent_id, str):
            messages.append(f"manual migration required: agents[{index}].agent_id must be a string")
        extra_keys = _unsupported_keys(agent, _AGENT_KEYS)
        if extra_keys:
            messages.append(
                f"manual migration required: agents[{index}] contains unsupported fields "
                + _format_key_list(extra_keys)
            )
        config = agent.get("config")
        if config is not None and not isinstance(config, dict):
            messages.append(f"manual migration required: agents[{index}].config must be a mapping")
        elif isinstance(config, dict):
            messages.extend(_non_string_key_messages(config, f"agents[{index}].config"))

        scheduler = agent.get("scheduler")
        if scheduler is None:
            messages.append(
                f"manual migration required: agents[{index}] agent_id={agent.get('agent_id')!r} "
                "is missing scheduler"
            )
            continue
        if not isinstance(scheduler, dict):
            messages.append(f"manual migration required: agents[{index}].scheduler must be a mapping")
            continue
        scheduler_extra_keys = _unsupported_keys(scheduler, _SCHEDULER_KEYS)
        if scheduler_extra_keys:
            messages.append(
                f"manual migration required: agents[{index}].scheduler contains unsupported fields "
                + _format_key_list(scheduler_extra_keys)
            )
        if "backend_id" in scheduler and "model" in scheduler:
            messages.append(
                f"manual migration required: agents[{index}].scheduler cannot contain both backend_id and model"
            )
        backend_id = scheduler.get("backend_id")
        has_model = "model" in scheduler
        model = scheduler.get("model")
        if "backend_id" in scheduler and not isinstance(backend_id, str):
            messages.append(
                f"manual migration required: agents[{index}].scheduler.backend_id must be a string"
            )
        if has_model and not isinstance(model, str):
            messages.append(
                f"manual migration required: agents[{index}].scheduler.model must be a string"
            )
        if "backend_id" not in scheduler and not has_model:
            messages.append(
                f"manual migration required: agents[{index}] agent_id={agent.get('agent_id')!r} "
                "scheduler must contain string backend_id or model"
            )
        scheduler_type = scheduler.get("type")
        if scheduler_type is not None and scheduler_type not in _SUPPORTED_SCHEDULER_TYPES:
            messages.append(
                f"manual migration required: agents[{index}].scheduler.type={scheduler_type} "
                "is not supported"
            )
    return messages


def _top_level_scheduler_manual_fix_messages(
    payload: dict[str, Any],
    agent_backend_to_backend: dict[str, str],
) -> list[str]:
    scheduler = payload.get("scheduler")
    if scheduler is None:
        return []
    if not isinstance(scheduler, dict):
        return ["manual migration required: scheduler must be a mapping"]
    messages: list[str] = []
    extra_keys = _unsupported_keys(scheduler, _SCHEDULER_KEYS)
    if extra_keys:
        messages.append(
            "manual migration required: scheduler contains unsupported fields "
            + _format_key_list(extra_keys)
        )
    if "backend_id" in scheduler and "model" in scheduler:
        messages.append("manual migration required: scheduler cannot contain both backend_id and model")
    if "model" in scheduler and not isinstance(scheduler.get("model"), str):
        messages.append("manual migration required: scheduler.model must be a string")
    if "backend_id" in scheduler and not isinstance(scheduler.get("backend_id"), str):
        messages.append("manual migration required: scheduler.backend_id must be a string")
    scheduler_type = scheduler.get("type")
    if scheduler_type is not None and scheduler_type not in _SUPPORTED_SCHEDULER_TYPES:
        messages.append(f"manual migration required: scheduler.type={scheduler_type} is not supported")

    agent_backend_id = payload.get("agent_backend_id")
    role_adapters = payload.get("role_adapters")
    has_role_adapters = isinstance(role_adapters, list) and bool(role_adapters)
    if agent_backend_id is None and has_role_adapters and "backend_id" in scheduler:
        messages.append(
            "manual migration required: scheduler.backend_id with role_adapters would be dropped; "
            "use a supported top-level agent migration or remove the top-level scheduler"
        )
    if (
        isinstance(agent_backend_id, str)
        and agent_backend_id in agent_backend_to_backend
        and "model" in scheduler
    ):
        messages.append(
            "manual migration required: scheduler.model is ambiguous when top-level "
            f"agent_backend_id={agent_backend_id} resolves to a backend"
        )
    if (
        isinstance(agent_backend_id, str)
        and agent_backend_id in agent_backend_to_backend
        and "backend_id" in scheduler
    ):
        messages.append(
            "manual migration required: scheduler.backend_id is ambiguous when top-level "
            f"agent_backend_id={agent_backend_id} resolves to a backend"
        )
    return messages


def _top_level_agent_backend_manual_fix_messages(
    payload: dict[str, Any],
    agent_backend_to_backend: dict[str, str],
) -> list[str]:
    agent_backend_id = payload.get("agent_backend_id")
    if agent_backend_id is None:
        return []
    if not isinstance(agent_backend_id, str):
        return ["manual migration required: top-level agent_backend_id must be a string"]
    if agent_backend_id in agent_backend_to_backend:
        return []
    return [
        "manual migration required: "
        f"top-level agent_backend_id={agent_backend_id} does not resolve to a migrated backend"
    ]


def _agent_backend_reference_manual_fix_messages(
    agents: list[dict[str, Any]],
    backend_specs: dict[str, dict[str, Any]],
) -> list[str]:
    backend_ids = set(backend_specs)
    messages: list[str] = []
    for agent in agents:
        scheduler = agent.get("scheduler")
        if not isinstance(scheduler, dict):
            continue
        backend_id = scheduler.get("backend_id")
        if isinstance(backend_id, str) and backend_id not in backend_ids:
            messages.append(
                "manual migration required: "
                f"agents.{agent.get('agent_id')}.scheduler.backend_id={backend_id} "
                "has no migrated backend"
            )
    return messages


def _role_adapter_manual_fix_messages(
    value: Any,
    agent_backend_to_backend: dict[str, str],
) -> list[str]:
    messages: list[str] = []
    role_adapters = value
    if isinstance(role_adapters, list):
        sanitized_agent_ids: dict[str, str] = {}
        for index, adapter in enumerate(role_adapters):
            if not isinstance(adapter, dict):
                messages.append(f"manual migration required: role_adapters[{index}] must be a mapping")
                continue
            if adapter.get("role_type") != "dut_agent":
                messages.append(
                    f"manual migration required: role_adapters[{index}] role_type={adapter.get('role_type')!r} "
                    "is outside the minimal DUT agent migration scope"
                )
            extra_keys = _unsupported_keys(adapter, _ROLE_ADAPTER_KEYS)
            if extra_keys:
                messages.append(
                    f"manual migration required: role_adapters[{index}] contains unsupported fields "
                    + _format_key_list(extra_keys)
                )
            if adapter.get("role_type") == "dut_agent":
                adapter_id = adapter.get("adapter_id")
                agent_backend_id = adapter.get("agent_backend_id")
                if adapter_id is not None and not isinstance(adapter_id, str):
                    messages.append(
                        f"manual migration required: role_adapters[{index}].adapter_id must be a string"
                    )
                raw_agent_id = str(adapter_id or agent_backend_id)
                sanitized_agent_id = _sanitize_id(raw_agent_id)
                previous = sanitized_agent_ids.get(sanitized_agent_id)
                if previous is not None and previous != raw_agent_id:
                    messages.append(
                        "manual migration required: sanitized agent_id collision for role_adapters "
                        f"{previous!r} and {raw_agent_id!r} both produce {sanitized_agent_id!r}"
                    )
                sanitized_agent_ids.setdefault(sanitized_agent_id, raw_agent_id)

                if not isinstance(agent_backend_id, str):
                    messages.append(
                        f"manual migration required: role_adapters[{index}] adapter_id={adapter.get('adapter_id')!r} "
                        "is a dut_agent without agent_backend_id"
                    )
                elif agent_backend_id not in agent_backend_to_backend:
                    messages.append(
                        f"manual migration required: role_adapters[{index}] adapter_id={adapter.get('adapter_id')!r} "
                        f"agent_backend_id={agent_backend_id} does not resolve to a migrated backend"
                    )
                runtime_id = adapter.get("agent_runtime_id")
                scheduler_type = _scheduler_type_from_runtime(runtime_id)
                if scheduler_type is None:
                    messages.append(
                        f"manual migration required: role_adapters[{index}] "
                        f"agent_runtime_id={runtime_id} is not supported"
                    )
    elif role_adapters is not None:
        messages.append("manual migration required: role_adapters must be a list")

    return messages


def _index_static_backends(value: Any) -> dict[str, dict[str, Any]]:
    indexed: dict[str, dict[str, Any]] = {}
    if not isinstance(value, list):
        return indexed
    for backend in value:
        if not isinstance(backend, dict):
            continue
        backend_id = backend.get("backend_id")
        if isinstance(backend_id, str):
            indexed[backend_id] = backend
    return indexed


def _migrate_agent_backends(
    value: Any,
    static_backends: dict[str, dict[str, Any]],
    backend_specs: dict[str, dict[str, Any]],
) -> dict[str, str]:
    mapping: dict[str, str] = {}
    if not isinstance(value, list):
        return mapping

    for agent_backend in value:
        if not isinstance(agent_backend, dict):
            continue
        agent_backend_id = agent_backend.get("agent_backend_id")
        if not isinstance(agent_backend_id, str):
            continue

        referenced_backend_id = agent_backend.get("backend_id")
        if isinstance(referenced_backend_id, str) and referenced_backend_id in static_backends:
            if not _is_valid_static_backend(static_backends[referenced_backend_id]):
                continue
            backend_specs.setdefault(
                referenced_backend_id,
                _to_v2_backend(static_backends[referenced_backend_id]),
            )
            mapping[agent_backend_id] = referenced_backend_id
    return mapping


def _is_valid_static_backend(backend: dict[str, Any]) -> bool:
    config = backend.get("config")
    return (
        isinstance(backend.get("backend_id"), str)
        and isinstance(backend.get("type"), str)
        and (config is None or isinstance(config, dict))
        and not _non_string_key_messages(config or {}, "backends[].config")
    )


def _to_v2_backend(backend: dict[str, Any]) -> dict[str, Any]:
    return {
        "backend_id": backend["backend_id"],
        "type": backend["type"],
        "config": deepcopy(backend.get("config") or {}),
    }


def _migrate_role_adapter_agents(
    value: Any,
    agent_backend_to_backend: dict[str, str],
) -> list[dict[str, Any]]:
    agents: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return agents

    for adapter in value:
        if not isinstance(adapter, dict) or adapter.get("role_type") != "dut_agent":
            continue
        agent_backend_id = adapter.get("agent_backend_id")
        backend_id = (
            agent_backend_to_backend.get(agent_backend_id)
            if isinstance(agent_backend_id, str)
            else None
        )
        if backend_id is None:
            continue
        adapter_id = adapter.get("adapter_id")
        raw_agent_id = adapter_id if isinstance(adapter_id, str) else agent_backend_id
        agent_id = _sanitize_id(raw_agent_id)
        agents.append(
            {
                "agent_id": agent_id,
                "scheduler": {
                    "type": _scheduler_type_from_runtime(adapter.get("agent_runtime_id")) or "framework_loop",
                    "backend_id": backend_id,
                },
                "config": {},
            }
        )
    return agents


def _migrate_legacy_agents(value: Any, backend_specs: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    agents: list[dict[str, Any]] = []
    if not isinstance(value, list):
        return agents

    for index, agent in enumerate(value):
        if not isinstance(agent, dict):
            continue
        scheduler = agent.get("scheduler")
        if not isinstance(scheduler, dict):
            continue
        raw_agent_id = agent.get("agent_id")
        agent_id = _sanitize_id(raw_agent_id if isinstance(raw_agent_id, str) else f"agent_{index + 1}")
        backend_id = scheduler.get("backend_id")
        migrated_scheduler = {
            "type": str(scheduler.get("type") or "framework_loop"),
        }
        if isinstance(backend_id, str):
            migrated_scheduler["backend_id"] = backend_id
        elif "model" in scheduler:
            generated_backend_id = _unique_id(f"{agent_id}_model", set(backend_specs))
            backend_specs[generated_backend_id] = {
                "backend_id": generated_backend_id,
                "type": "litellm",
                "config": {"model": scheduler["model"]},
            }
            migrated_scheduler["backend_id"] = generated_backend_id
        else:
            continue
        agents.append(
            {
                "agent_id": agent_id,
                "scheduler": migrated_scheduler,
                "config": deepcopy(agent.get("config") if isinstance(agent.get("config"), dict) else {}),
            }
        )
    return agents


def _migrate_top_level_agent(
    payload: dict[str, Any],
    agent_backend_to_backend: dict[str, str],
    backend_specs: dict[str, dict[str, Any]],
) -> dict[str, Any] | None:
    agent_backend_id = payload.get("agent_backend_id")
    scheduler = payload.get("scheduler")
    backend_id = agent_backend_to_backend.get(agent_backend_id) if isinstance(agent_backend_id, str) else None

    if backend_id is None and isinstance(scheduler, dict) and "model" in scheduler:
        backend_id = _unique_id("agent_model", set(backend_specs))
        backend_specs[backend_id] = {
            "backend_id": backend_id,
            "type": "litellm",
            "config": {"model": scheduler["model"]},
        }
    if backend_id is None:
        return None

    scheduler_type = (
        scheduler.get("type")
        if isinstance(scheduler, dict) and scheduler.get("type")
        else "framework_loop"
    )
    return {
        "agent_id": "agent",
        "scheduler": {
            "type": str(scheduler_type),
            "backend_id": backend_id,
        },
        "config": {},
    }


def _infer_placeholder_kit(payload: dict[str, Any]) -> dict[str, Any]:
    haystack = yaml.safe_dump(payload, sort_keys=True).lower()
    if "swebench" in haystack or "swe-bench" in haystack:
        return {
            "kit_id": "swebench",
            "benchmark_config": {"split": "test"},
            "provider": "docker",
            "profile_id": "swebench_runtime",
            "asset_dir": "tests/fixtures/agentkit_v2/swebench",
        }
    if "terminal_bench" in haystack or "terminal-bench" in haystack:
        return {
            "kit_id": "terminal_bench",
            "benchmark_config": {},
            "provider": "docker",
            "profile_id": "terminal_bench_runtime",
            "asset_dir": "tests/fixtures/agentkit_v2/terminal_bench",
        }
    if "appworld" in haystack:
        return {
            "kit_id": "appworld",
            "benchmark_config": {},
            "provider": "local_process",
            "profile_id": "appworld_local",
            "asset_dir": "tests/fixtures/agentkit_v2/appworld",
        }
    return {
        "kit_id": "tau2",
        "benchmark_config": {},
        "provider": "local_process",
        "profile_id": "tau2_local",
        "asset_dir": "tests/fixtures/agentkit_v2/tau2",
    }


def _metadata(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = payload.get("metadata")
    return deepcopy(metadata) if isinstance(metadata, dict) else {}


def _scheduler_type_from_runtime(value: Any) -> str | None:
    if value is None or value == "":
        return "framework_loop"
    if not isinstance(value, str):
        return None
    return _KNOWN_RUNTIME_SCHEDULER_TYPES.get(value)


def _sanitize_id(value: str) -> str:
    sanitized = "".join(character if character.isalnum() or character == "_" else "_" for character in value)
    return sanitized.strip("_") or "generated"


def _unique_id(base: str, used: set[str]) -> str:
    candidate = _sanitize_id(base)
    if candidate not in used:
        used.add(candidate)
        return candidate
    index = 2
    while f"{candidate}_{index}" in used:
        index += 1
    unique = f"{candidate}_{index}"
    used.add(unique)
    return unique


def _write_yaml_atomic(output_path: Path, payload: dict[str, Any]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with temp_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False, allow_unicode=False)
        temp_path.replace(output_path)
    except Exception:
        temp_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Legacy v1 YAML path")
    parser.add_argument("--output", required=True, type=Path, help="Destination v2 YAML path")
    args = parser.parse_args(argv)

    try:
        result = migrate_file(args.input, args.output)
    except yaml.YAMLError as exc:
        print(f"manual migration required: failed to read input YAML: {exc}", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"manual migration required: failed to write output YAML: {exc}", file=sys.stderr)
        return 1
    if result.ok:
        return 0
    for message in result.manual_fixes:
        print(message, file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
