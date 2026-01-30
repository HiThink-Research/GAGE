"""Lightweight schema validation helpers for PipelineConfig payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Sequence

ALLOWED_STEPS = {"support", "inference", "arena", "judge", "auto_eval", "report", "hook"}


class SchemaValidationError(ValueError):
    """Raised when the config payload fails structural validation."""

    def __init__(self, errors: Sequence[str]) -> None:
        message = "PipelineConfig validation failed:\n" + "\n".join(f"  - {err}" for err in errors)
        super().__init__(message)
        self.errors = list(errors)


def normalize_pipeline_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and normalize the raw payload produced by YAML/JSON readers."""

    errors: List[str] = []
    data = deepcopy(payload)

    datasets = _ensure_list(data.get("datasets"), "datasets", errors)
    models = _ensure_list(data.get("models"), "models", errors)
    backends = _ensure_list(data.get("backends"), "backends", errors)
    agent_backends = _ensure_list(data.get("agent_backends"), "agent_backends", errors)
    sandbox_profiles = _ensure_list(data.get("sandbox_profiles"), "sandbox_profiles", errors)
    mcp_clients = _ensure_list(data.get("mcp_clients"), "mcp_clients", errors)
    prompts = _ensure_list(data.get("prompts"), "prompts", errors)
    role_adapters = _ensure_list(data.get("role_adapters"), "role_adapters", errors)
    metrics = _ensure_list(data.get("metrics"), "metrics", errors)
    tasks = _ensure_list(data.get("tasks"), "tasks", errors)
    summary_generators = _ensure_list(data.get("summary_generators"), "summary_generators", errors)

    builtin = data.get("builtin")
    custom = data.get("custom")

    if not datasets:
        errors.append("at least one dataset must be declared")
    if not role_adapters:
        errors.append("at least one role adapter must be declared")
    # NOTE: In the TaskOrchestrator mode, a config that declares `tasks` does not
    # need to explicitly provide `builtin`/`custom`. We only treat the payload as
    # incomplete when both `builtin/custom` and `tasks` are missing.
    if not (custom or builtin) and not tasks:
        errors.append("either 'builtin' or 'custom' pipeline must be provided when 'tasks' is empty")

    dataset_ids = _ensure_unique(datasets, "dataset_id", "dataset", errors)
    model_ids = _ensure_unique(models, "model_id", "model", errors)
    backend_ids = _ensure_unique(backends, "backend_id", "backend", errors)
    agent_backend_ids = _ensure_unique(agent_backends, "agent_backend_id", "agent backend", errors)
    _normalize_sandbox_profile_ids(sandbox_profiles, errors)
    _ensure_unique(sandbox_profiles, "sandbox_id", "sandbox profile", errors)
    mcp_client_ids = _ensure_unique(mcp_clients, "mcp_client_id", "mcp client", errors)
    prompt_ids = _ensure_unique(prompts, "prompt_id", "prompt", errors)
    adapter_ids = _ensure_unique(role_adapters, "adapter_id", "role adapter", errors)
    metric_ids = _ensure_unique(metrics, "metric_id", "metric", errors, allow_str=True)

    _validate_role_bindings(role_adapters, backend_ids, agent_backend_ids, prompt_ids, mcp_client_ids, errors)
    _validate_steps(custom, errors)
    _validate_tasks(
        tasks=tasks,
        dataset_ids=dataset_ids,
        adapter_ids=adapter_ids,
        metric_ids=metric_ids,
        has_custom=bool(custom),
        has_builtin=bool(builtin),
        errors=errors,
    )

    if errors:
        raise SchemaValidationError(errors)

    data["datasets"] = datasets
    data["models"] = models
    data["backends"] = backends
    data["agent_backends"] = agent_backends
    data["sandbox_profiles"] = sandbox_profiles
    data["mcp_clients"] = mcp_clients
    data["prompts"] = prompts
    data["role_adapters"] = role_adapters
    data["metrics"] = metrics
    data["tasks"] = tasks
    data["summary_generators"] = summary_generators
    return data


def _ensure_list(value: Any, field: str, errors: List[str]) -> List[dict]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    errors.append(f"'{field}' must be a list")
    return []


def _ensure_unique(items: List[dict], key: str, label: str, errors: List[str], allow_str: bool = False) -> List[str]:
    ids: List[str] = []
    seen = set()
    for item in items:
        if isinstance(item, dict):
            value = item.get(key)
            if not value and allow_str and len(item) == 1:
                value = next(iter(item.keys()))
            if not value:
                errors.append(f"{label} entries must declare '{key}'")
                continue
        elif allow_str and isinstance(item, str):
            value = item
        else:
            errors.append(f"{label} entries must be dictionaries")
            continue
        if value in seen:
            errors.append(f"duplicate {label} id '{value}' detected")
        else:
            seen.add(value)
            ids.append(value)
    return ids


def _normalize_sandbox_profile_ids(items: List[dict], errors: List[str]) -> None:
    for item in items:
        if not isinstance(item, dict):
            continue
        sandbox_id = item.get("sandbox_id")
        template_name = item.get("template_name")
        if sandbox_id and template_name and sandbox_id != template_name:
            errors.append(
                f"sandbox profile sandbox_id '{sandbox_id}' does not match template_name '{template_name}'"
            )
            continue
        if not sandbox_id and template_name:
            item["sandbox_id"] = template_name


def _validate_role_bindings(
    role_adapters: List[dict],
    backend_ids: List[str],
    agent_backend_ids: List[str],
    prompt_ids: List[str],
    mcp_client_ids: List[str],
    errors: List[str],
) -> None:
    backend_set = set(backend_ids)
    agent_backend_set = set(agent_backend_ids)
    prompt_set = set(prompt_ids)
    mcp_client_set = set(mcp_client_ids)
    for adapter in role_adapters:
        backend_id = adapter.get("backend_id")
        inline_backend = adapter.get("backend")
        agent_backend_id = adapter.get("agent_backend_id")
        inline_agent_backend = adapter.get("agent_backend")
        prompt_id = adapter.get("prompt_id")
        mcp_client_id = adapter.get("mcp_client_id")
        adapter_id = adapter.get("adapter_id", "<unknown>")
        if backend_id and backend_id not in backend_set:
            errors.append(
                f"role adapter '{adapter_id}' references unknown backend '{backend_id}'"
            )
        if inline_backend is not None:
            if not isinstance(inline_backend, dict):
                errors.append(
                    f"role adapter '{adapter_id}' inline backend must be a mapping with 'type'/'config'"
                )
            elif not inline_backend.get("type"):
                errors.append(
                    f"role adapter '{adapter_id}' inline backend missing required field 'type'"
                )
        if agent_backend_id and agent_backend_id not in agent_backend_set:
            errors.append(
                f"role adapter '{adapter_id}' references unknown agent backend '{agent_backend_id}'"
            )
        if inline_agent_backend is not None:
            if not isinstance(inline_agent_backend, dict):
                errors.append(
                    f"role adapter '{adapter_id}' inline agent backend must be a mapping with 'type'/'config'"
                )
            elif not inline_agent_backend.get("type"):
                errors.append(
                    f"role adapter '{adapter_id}' inline agent backend missing required field 'type'"
                )
        if prompt_id and prompt_id not in prompt_set:
            if not _prompt_id_in_registry(prompt_id):
                errors.append(
                    f"role adapter '{adapter_id}' references unknown prompt '{prompt_id}'"
                )
        if mcp_client_id and mcp_client_id not in mcp_client_set:
            errors.append(
                f"role adapter '{adapter_id}' references unknown mcp client '{mcp_client_id}'"
            )


def _prompt_id_in_registry(prompt_id: str) -> bool:
    if not prompt_id:
        return False
    try:
        from gage_eval.registry import registry
    except Exception:
        return False
    try:
        registry.auto_discover("prompts", "gage_eval.assets.prompts.catalog")
        registry.get("prompts", prompt_id)
        return True
    except KeyError:
        return False
    except Exception:
        return False


def _validate_steps(custom: Any, errors: List[str]) -> None:
    if not custom:
        return
    steps = custom.get("steps")
    if steps is None:
        errors.append("custom pipeline must declare 'steps'")
        return
    for idx, step in enumerate(steps):
        step_type = (step or {}).get("step")
        if step_type not in ALLOWED_STEPS:
            errors.append(
                f"custom steps[{idx}] uses invalid step '{step_type}', allowed values: {sorted(ALLOWED_STEPS)}"
            )


def _validate_tasks(
    *,
    tasks: List[dict],
    dataset_ids: List[str],
    adapter_ids: List[str],
    metric_ids: List[str],
    has_custom: bool,
    has_builtin: bool,
    errors: List[str],
) -> None:
    dataset_set = set(dataset_ids)
    adapter_set = set(adapter_ids)
    metric_set = set(metric_ids)
    for task in tasks:
        task_id = task.get("task_id", "<unknown>")
        dataset_id = task.get("dataset_id") or task.get("dataset_ref")
        if dataset_id not in dataset_set:
            errors.append(f"task '{task_id}' references unknown dataset '{dataset_id}'")
        steps = task.get("steps") or []
        if not steps and not (has_custom or has_builtin):
            errors.append(
                f"task '{task_id}' must declare steps when pipeline has no builtin/custom definition"
            )
        for idx, step in enumerate(steps):
            step_type = (step or {}).get("step")
            if step_type not in ALLOWED_STEPS:
                errors.append(
                    f"task '{task_id}' step[{idx}] uses invalid step '{step_type}' "
                    f"(allowed: {sorted(ALLOWED_STEPS)})"
                )
            adapter_id = (step or {}).get("adapter_id") or (step or {}).get("role_ref")
            if adapter_id and adapter_id not in adapter_set:
                errors.append(
                    f"task '{task_id}' references unknown role adapter '{adapter_id}'"
                )
        overrides = task.get("metric_overrides") or []
        for metric in overrides:
            metric_id = metric.get("metric_id")
            if metric_id and metric_id not in metric_set:
                errors.append(
                    f"task '{task_id}' overrides metric '{metric_id}' which is not defined globally"
                )
