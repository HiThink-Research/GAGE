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
    # 在支持 TaskOrchestrator 模式后，若已声明 tasks，则可以不显式提供 builtin/custom。
    # 仅当既没有 builtin/custom，又没有 tasks 时才认为配置不完整。
    if not (custom or builtin) and not tasks:
        errors.append("either 'builtin' or 'custom' pipeline must be provided when 'tasks' is empty")

    dataset_ids = _ensure_unique(datasets, "dataset_id", "dataset", errors)
    model_ids = _ensure_unique(models, "model_id", "model", errors)
    backend_ids = _ensure_unique(backends, "backend_id", "backend", errors)
    prompt_ids = _ensure_unique(prompts, "prompt_id", "prompt", errors)
    adapter_ids = _ensure_unique(role_adapters, "adapter_id", "role adapter", errors)
    metric_ids = _ensure_unique(metrics, "metric_id", "metric", errors, allow_str=True)

    _validate_role_bindings(role_adapters, backend_ids, prompt_ids, errors)
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


def _validate_role_bindings(
    role_adapters: List[dict],
    backend_ids: List[str],
    prompt_ids: List[str],
    errors: List[str],
) -> None:
    backend_set = set(backend_ids)
    prompt_set = set(prompt_ids)
    for adapter in role_adapters:
        backend_id = adapter.get("backend_id")
        inline_backend = adapter.get("backend")
        prompt_id = adapter.get("prompt_id")
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
        if prompt_id and prompt_id not in prompt_set:
            errors.append(
                f"role adapter '{adapter_id}' references unknown prompt '{prompt_id}'"
            )


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
