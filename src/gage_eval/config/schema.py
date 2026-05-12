"""Lightweight schema validation helpers for PipelineConfig payloads."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List, Sequence

from gage_eval.pipeline.step_contracts import collect_step_sequence_issues

_VALID_FAILURE_POLICIES = {"fail_fast", "graceful", "best_effort"}
_VALID_SHUFFLE_STRATEGIES = {"auto", "in_memory", "reservoir", "external_index"}


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
    environments = _ensure_list(data.get("environments"), "environments", errors)
    role_adapters = _ensure_list(data.get("role_adapters"), "role_adapters", errors)
    metrics = _ensure_list(data.get("metrics"), "metrics", errors)
    tasks = _ensure_list(data.get("tasks"), "tasks", errors)
    summary_generators = _ensure_list(data.get("summary_generators"), "summary_generators", errors)

    builtin = data.get("builtin")
    custom = data.get("custom")
    agentkit_v2_compat = _is_agentkit_v2_compat_payload(data)

    if not datasets and not agentkit_v2_compat:
        errors.append("at least one dataset must be declared")
    if not role_adapters:
        errors.append("at least one role adapter must be declared")
    # NOTE: In the TaskOrchestrator mode, a config that declares `tasks` does not
    # need to explicitly provide `builtin`/`custom`. We only treat the payload as
    # incomplete when both `builtin/custom` and `tasks` are missing.
    if not (custom or builtin) and not tasks and not agentkit_v2_compat:
        errors.append("either 'builtin' or 'custom' pipeline must be provided when 'tasks' is empty")

    dataset_ids = _ensure_unique(datasets, "dataset_id", "dataset", errors)
    model_ids = _ensure_unique(models, "model_id", "model", errors)
    backend_ids = _ensure_unique(backends, "backend_id", "backend", errors)
    agent_backend_ids = _ensure_unique(agent_backends, "agent_backend_id", "agent backend", errors)
    _normalize_sandbox_profile_ids(sandbox_profiles, errors)
    _ensure_unique(sandbox_profiles, "sandbox_id", "sandbox profile", errors)
    mcp_client_ids = _ensure_unique(mcp_clients, "mcp_client_id", "mcp client", errors)
    prompt_ids = _ensure_unique(prompts, "prompt_id", "prompt", errors)
    environment_ids = _ensure_unique(environments, "env_id", "environment", errors)
    adapter_ids = _ensure_unique(role_adapters, "adapter_id", "role adapter", errors)
    metric_ids = _ensure_unique(metrics, "metric_id", "metric", errors, allow_str=True)

    _validate_role_bindings(role_adapters, backend_ids, agent_backend_ids, prompt_ids, mcp_client_ids, errors)
    _validate_steps(custom, adapter_ids=adapter_ids, errors=errors)
    _validate_tasks(
        tasks=tasks,
        dataset_ids=dataset_ids,
        adapter_ids=adapter_ids,
        metric_ids=metric_ids,
        has_custom=bool(custom),
        has_builtin=bool(builtin),
        errors=errors,
    )
    _validate_external_harness_pipeline(
        data=data,
        datasets=datasets,
        dataset_ids=dataset_ids,
        backends=backends,
        role_adapters=role_adapters,
        adapter_ids=adapter_ids,
        tasks=tasks,
        environments=environments,
        environment_ids=environment_ids,
        errors=errors,
    )

    if errors:
        raise SchemaValidationError(errors)

    data["datasets"] = datasets
    data["models"] = models
    data["backends"] = backends
    _set_optional_list(data, "agent_backends", agent_backends)
    _set_optional_list(data, "sandbox_profiles", sandbox_profiles)
    data["mcp_clients"] = mcp_clients
    data["prompts"] = prompts
    data["role_adapters"] = role_adapters
    data["metrics"] = metrics
    data["tasks"] = tasks
    data["summary_generators"] = summary_generators
    return data


def _set_optional_list(data: Dict[str, Any], key: str, value: List[dict]) -> None:
    if value:
        data[key] = value
    else:
        data.pop(key, None)


def _is_agentkit_v2_compat_payload(data: Dict[str, Any]) -> bool:
    return all(data.get(section) for section in ("agents", "benchmarks", "environments", "dut_agents"))


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
        agent_runtime_id = adapter.get("agent_runtime_id")
        prompt_id = adapter.get("prompt_id")
        mcp_client_id = adapter.get("mcp_client_id")
        role_type = adapter.get("role_type")
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
        if _is_installed_client_dut_agent(role_type=role_type, agent_runtime_id=agent_runtime_id):
            if agent_backend_id:
                errors.append(
                    f"role adapter '{adapter_id}' uses installed_client runtime '{agent_runtime_id}' and must not declare 'agent_backend_id'"
                )
            if inline_agent_backend is not None:
                errors.append(
                    f"role adapter '{adapter_id}' uses installed_client runtime '{agent_runtime_id}' and must not declare inline 'agent_backend'"
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
        from gage_eval.registry import load_default_manifest_repository, registry
    except Exception:
        return False
    try:
        registry.get("prompts", prompt_id)
        return True
    except KeyError:
        pass
    except Exception:
        return False

    try:
        return load_default_manifest_repository().resolve("prompts", prompt_id) is not None
    except Exception:
        return False


def _is_installed_client_dut_agent(*, role_type: Any, agent_runtime_id: Any) -> bool:
    if role_type != "dut_agent":
        return False
    if not isinstance(agent_runtime_id, str):
        return False
    return "installed_client" in agent_runtime_id


def _validate_steps(
    custom: Any,
    *,
    adapter_ids: List[str],
    errors: List[str],
) -> None:
    if not custom:
        return
    steps = custom.get("steps")
    if steps is None:
        errors.append("custom pipeline must declare 'steps'")
        return
    _validate_step_sequence(
        steps,
        adapter_ids=adapter_ids,
        owner_label="custom steps",
        errors=errors,
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
    metric_set = set(metric_ids)
    for task in tasks:
        task_id = task.get("task_id", "<unknown>")
        execution_mode = task.get("execution_mode", "sample_loop")
        if execution_mode not in {"sample_loop", "task_batch_harness"}:
            errors.append(
                f"external_harness.config.invalid_steps: task '{task_id}' "
                f"has unsupported execution_mode '{execution_mode}'"
            )
        overrides = task.get("metric_overrides") or []
        for metric in overrides:
            metric_id = metric.get("metric_id")
            if metric_id and metric_id not in metric_set:
                errors.append(
                    f"task '{task_id}' overrides metric '{metric_id}' which is not defined globally"
                )
        failure_policy = task.get("failure_policy")
        if failure_policy is not None and str(failure_policy).strip().lower() not in _VALID_FAILURE_POLICIES:
            errors.append(
                f"task '{task_id}' declares unsupported failure_policy '{failure_policy}'"
            )
        shuffle_strategy = task.get("shuffle_strategy")
        if shuffle_strategy is not None and str(shuffle_strategy).strip().lower() not in _VALID_SHUFFLE_STRATEGIES:
            errors.append(
                f"task '{task_id}' declares unsupported shuffle_strategy '{shuffle_strategy}'"
            )
        support_payload_policy = task.get("support_payload_policy")
        if support_payload_policy is not None and not isinstance(support_payload_policy, dict):
            errors.append(
                f"task '{task_id}' field 'support_payload_policy' must be a mapping"
            )
        if execution_mode == "task_batch_harness":
            continue
        dataset_id = task.get("dataset_id") or task.get("dataset_ref")
        if dataset_id not in dataset_set:
            errors.append(f"task '{task_id}' references unknown dataset '{dataset_id}'")
        steps = task.get("steps") or []
        if not steps and not (has_custom or has_builtin):
            errors.append(
                f"task '{task_id}' must declare steps when pipeline has no builtin/custom definition"
            )
        _validate_step_sequence(
            steps,
            adapter_ids=adapter_ids,
            owner_label=f"task '{task_id}' step",
            errors=errors,
            )


def _validate_external_harness_pipeline(
    *,
    data: Dict[str, Any],
    datasets: List[dict],
    dataset_ids: List[str],
    backends: List[dict],
    role_adapters: List[dict],
    adapter_ids: List[str],
    tasks: List[dict],
    environments: List[dict],
    environment_ids: List[str],
    errors: List[str],
) -> None:
    external_adapters = [
        adapter
        for adapter in role_adapters
        if isinstance(adapter, dict) and _is_external_harness_adapter(adapter)
    ]
    task_batch_tasks = [
        task
        for task in tasks
        if isinstance(task, dict) and task.get("execution_mode") == "task_batch_harness"
    ]
    active = bool(external_adapters or task_batch_tasks)
    if not active:
        return

    for key in ("agents", "prompts", "benchmarks", "dut_agents"):
        if data.get(key):
            errors.append(
                f"external_harness.config.forbidden_top_level: top-level '{key}' is not allowed with external_harness"
            )

    dataset_by_id = {
        str(dataset_id): dataset
        for dataset_id, dataset in zip(dataset_ids, datasets)
        if isinstance(dataset, dict)
    }
    backend_by_id = {
        str(backend.get("backend_id")): backend
        for backend in backends
        if isinstance(backend, dict) and backend.get("backend_id")
    }
    environment_by_id = {
        str(env_id): environment
        for env_id, environment in zip(environment_ids, environments)
        if isinstance(environment, dict)
    }

    _validate_external_harness_tasks(
        tasks=task_batch_tasks,
        dataset_by_id=dataset_by_id,
        adapter_ids=adapter_ids,
        errors=errors,
    )
    _validate_external_harness_adapters(
        adapters=external_adapters,
        backend_by_id=backend_by_id,
        environment_by_id=environment_by_id,
        tasks=task_batch_tasks,
        errors=errors,
    )


def _is_external_harness_adapter(adapter: dict) -> bool:
    if adapter.get("role_type") == "external_harness":
        return True
    capabilities = adapter.get("capabilities") or []
    return "task_batch_harness" in capabilities


def _validate_external_harness_tasks(
    *,
    tasks: List[dict],
    dataset_by_id: Dict[str, dict],
    adapter_ids: List[str],
    errors: List[str],
) -> None:
    valid_loaders = {"harbor_registry", "harbor_local_path"}
    for task in tasks:
        task_id = task.get("task_id", "<unknown>")
        steps = task.get("steps") or []
        if not steps:
            errors.append(
                f"external_harness.config.invalid_steps: task '{task_id}' must declare non-empty steps"
            )
        dataset_id = task.get("dataset_id") or task.get("dataset_ref")
        dataset = dataset_by_id.get(str(dataset_id))
        if not dataset_id or dataset is None:
            errors.append(
                f"external_harness.config.invalid_dataset_params: task '{task_id}' references unknown dataset '{dataset_id}'"
            )
        else:
            loader = dataset.get("loader") or dataset.get("type")
            if loader not in valid_loaders:
                errors.append(
                    f"external_harness.config.invalid_loader: task '{task_id}' dataset '{dataset_id}' uses loader '{loader}'"
                )
            else:
                _validate_harbor_dataset_params(
                    task_id=task_id,
                    dataset_id=str(dataset_id),
                    loader=loader,
                    dataset=dataset,
                    task=task,
                    errors=errors,
                )
        if task.get("shuffle") is not None or task.get("shuffle_seed") is not None:
            errors.append(
                f"external_harness.config.invalid_dataset_params: task '{task_id}' must not set shuffle or shuffle_seed"
            )
        concurrency = task.get("concurrency")
        if concurrency is not None and _coerce_int(concurrency) < 1:
            errors.append(
                f"external_harness.config.invalid_concurrency: task '{task_id}' concurrency must be >= 1"
            )
        for issue in collect_step_sequence_issues(
            steps,
            owner_label=f"task '{task_id}' step",
            adapter_ids=adapter_ids,
            execution_mode="task_batch_harness",
        ):
            code = (
                "external_harness.config.unknown_adapter"
                if issue.code == "unknown_adapter"
                else "external_harness.config.invalid_steps"
            )
            errors.append(f"{code}: {issue.message}")


def _validate_external_harness_adapters(
    *,
    adapters: List[dict],
    backend_by_id: Dict[str, dict],
    environment_by_id: Dict[str, dict],
    tasks: List[dict],
    errors: List[str],
) -> None:
    allowed_providers = {"docker", "e2b"}
    for adapter in adapters:
        adapter_id = adapter.get("adapter_id", "<unknown>")
        backend_id = adapter.get("backend_id")
        env_id = adapter.get("env_id")
        params = adapter.get("params") if isinstance(adapter.get("params"), dict) else {}
        harness = params.get("harness") if isinstance(params.get("harness"), dict) else {}
        agent = harness.get("agent") if isinstance(harness.get("agent"), dict) else {}

        if not backend_id:
            errors.append(
                f"external_harness.config.missing_backend_id: role adapter '{adapter_id}' must declare backend_id"
            )
        if not env_id or str(env_id) not in environment_by_id:
            errors.append(
                f"external_harness.config.missing_env_id: role adapter '{adapter_id}' references unknown env_id '{env_id}'"
            )

        environment = environment_by_id.get(str(env_id)) if env_id else None
        provider = environment.get("provider") if isinstance(environment, dict) else None
        if isinstance(environment, dict) and provider not in allowed_providers:
            errors.append(
                f"external_harness.config.invalid_environment_provider: environment '{env_id}' provider '{provider}' is unsupported"
            )

        launcher = harness.get("launcher") if isinstance(harness.get("launcher"), dict) else {}
        launcher_mode = launcher.get("mode")
        if launcher_mode is not None and launcher_mode != "python_subprocess":
            errors.append(
                f"external_harness.config.invalid_launcher: role adapter '{adapter_id}' launcher mode '{launcher_mode}' is unsupported"
            )

        if _has_legacy_environment_override(adapter=adapter, params=params, harness=harness):
            errors.append(
                f"external_harness.config.invalid_environment_override: role adapter '{adapter_id}' uses legacy environment override keys"
            )

        environment_override = _environment_override_for(adapter=adapter, harness=harness)
        override_type = environment_override.get("type") if isinstance(environment_override, dict) else None
        if override_type is not None and _provider_conflicts_with_override(provider, override_type):
            errors.append(
                f"external_harness.config.provider_mismatch: role adapter '{adapter_id}' override type '{override_type}' conflicts with provider '{provider}'"
            )

        _validate_harness_agent(
            adapter_id=adapter_id,
            agent=agent,
            errors=errors,
        )
        _validate_adapter_concurrency(
            adapter_id=adapter_id,
            harness=harness,
            tasks=tasks,
            errors=errors,
        )
        _validate_backend_agent_overlap(
            adapter_id=adapter_id,
            backend=backend_by_id.get(str(backend_id)) if backend_id else None,
            agent=agent,
            errors=errors,
        )


def _has_legacy_environment_override(*, adapter: dict, params: dict, harness: dict) -> bool:
    legacy_keys = {"harbor_environment", "harbor_environment_override", "environment_binding"}
    return any(key in adapter or key in params or key in harness for key in legacy_keys)


def _environment_override_for(*, adapter: dict, harness: dict) -> dict:
    for container in (harness, adapter):
        value = container.get("environment_override")
        if isinstance(value, dict):
            return value
    return {}


def _provider_conflicts_with_override(provider: Any, override_type: Any) -> bool:
    if provider == "docker":
        return override_type != "docker"
    if provider == "e2b":
        return override_type != "e2b"
    if provider == "local_process":
        return override_type is not None
    return False


def _validate_harness_agent(*, adapter_id: Any, agent: dict, errors: List[str]) -> None:
    kind = agent.get("kind")
    if kind is not None and kind not in {"base_agent", "installed_client"}:
        errors.append(
            f"external_harness.config.invalid_agent: role adapter '{adapter_id}' harness.agent.kind '{kind}' is unsupported"
        )
    if not (agent.get("name") or agent.get("import_path")):
        errors.append(
            f"external_harness.config.invalid_agent: role adapter '{adapter_id}' harness.agent must declare name or import_path"
        )
    if "env" in agent:
        errors.append(
            f"external_harness.config.invalid_agent: role adapter '{adapter_id}' must use harness.agent.extra_env instead of deprecated harness.agent.env"
        )
    extra_env = agent.get("extra_env")
    if isinstance(extra_env, dict) and _contains_forbidden_agent_env(extra_env):
        errors.append(
            f"external_harness.config.secret_agent_env_forbidden: role adapter '{adapter_id}' harness.agent.extra_env must not contain secrets or template references"
        )


def _validate_adapter_concurrency(
    *,
    adapter_id: Any,
    harness: dict,
    tasks: List[dict],
    errors: List[str],
) -> None:
    harness_concurrency = harness.get("n_concurrent")
    if harness_concurrency is None:
        return
    harness_value = _coerce_int(harness_concurrency)
    for task in tasks:
        task_concurrency = task.get("concurrency")
        if task_concurrency is None:
            continue
        if _coerce_int(task_concurrency) != harness_value:
            errors.append(
                f"external_harness.config.invalid_concurrency: role adapter '{adapter_id}' harness.n_concurrent does not match task '{task.get('task_id', '<unknown>')}' concurrency"
            )


def _validate_backend_agent_overlap(
    *,
    adapter_id: Any,
    backend: dict | None,
    agent: dict,
    errors: List[str],
) -> None:
    if not backend:
        return
    backend_config = backend.get("config") if isinstance(backend.get("config"), dict) else {}
    generation_parameters = backend_config.get("generation_parameters")
    if not isinstance(generation_parameters, dict):
        generation_parameters = {}
    kwargs = agent.get("kwargs") if isinstance(agent.get("kwargs"), dict) else {}
    sampling_keys = {"temperature", "max_tokens", "max_new_tokens", "stop", "top_p"}
    direct_duplicates = sampling_keys & generation_parameters.keys() & kwargs.keys()
    llm_call_kwargs = kwargs.get("llm_call_kwargs") if isinstance(kwargs.get("llm_call_kwargs"), dict) else {}
    nested_duplicates = sampling_keys & generation_parameters.keys() & llm_call_kwargs.keys()
    if direct_duplicates or nested_duplicates:
        duplicates = sorted(direct_duplicates | nested_duplicates)
        errors.append(
            f"external_harness.config.invalid_agent: role adapter '{adapter_id}' duplicates generation parameters {duplicates}"
        )

    backend_model_info = backend_config.get("model_info")
    agent_model_info = kwargs.get("model_info")
    if isinstance(backend_model_info, dict) and isinstance(agent_model_info, dict):
        for path in _model_info_conflict_paths(backend_model_info, agent_model_info):
            errors.append(
                f"external_harness.translate.model_info_conflict: role adapter '{adapter_id}' model_info key '{path}' differs between backend and harness.agent.kwargs"
            )


def _validate_harbor_dataset_params(
    *,
    task_id: Any,
    dataset_id: str,
    loader: str,
    dataset: dict,
    task: dict,
    errors: List[str],
) -> None:
    params = dataset.get("params") if isinstance(dataset.get("params"), dict) else {}
    if loader == "harbor_registry":
        if not params.get("ref"):
            errors.append(
                f"external_harness.config.invalid_dataset_params: task '{task_id}' dataset '{dataset_id}' missing params.ref"
            )
        if params.get("registry_url") and params.get("registry_path"):
            errors.append(
                f"external_harness.config.invalid_dataset_params: task '{task_id}' dataset '{dataset_id}' must not set both registry_url and registry_path"
            )
    if loader == "harbor_local_path":
        if not params.get("path"):
            errors.append(
                f"external_harness.config.invalid_dataset_params: task '{task_id}' dataset '{dataset_id}' missing params.path"
            )
        if params.get("path_kind", "auto") not in {"auto", "dataset", "task"}:
            errors.append(
                f"external_harness.config.invalid_dataset_params: task '{task_id}' dataset '{dataset_id}' path_kind must be auto, dataset, or task"
            )
        if params.get("path_scope", "host") not in {"host", "launcher"}:
            errors.append(
                f"external_harness.config.invalid_dataset_params: task '{task_id}' dataset '{dataset_id}' path_scope must be host or launcher"
            )
        if params.get("path_kind") == "task":
            if params.get("task_names") or params.get("exclude_task_names"):
                errors.append(
                    f"external_harness.config.invalid_dataset_params: task '{task_id}' dataset '{dataset_id}' path_kind=task must not set task filters"
                )
            max_samples = task.get("max_samples")
            if max_samples is not None and _coerce_int(max_samples) > 1:
                errors.append(
                    f"external_harness.config.invalid_dataset_params: task '{task_id}' dataset '{dataset_id}' path_kind=task requires max_samples <= 1"
                )


def _model_info_conflict_paths(
    backend_model_info: Dict[str, Any],
    agent_model_info: Dict[str, Any],
    *,
    prefix: str = "",
) -> List[str]:
    conflicts: List[str] = []
    for key, backend_value in backend_model_info.items():
        if key not in agent_model_info:
            continue
        path = f"{prefix}.{key}" if prefix else str(key)
        agent_value = agent_model_info[key]
        if isinstance(backend_value, dict) and isinstance(agent_value, dict):
            conflicts.extend(
                _model_info_conflict_paths(
                    backend_value,
                    agent_value,
                    prefix=path,
                )
            )
        elif backend_value != agent_value:
            conflicts.append(path)
    return conflicts


def _contains_forbidden_agent_env(extra_env: Dict[Any, Any]) -> bool:
    for key, value in extra_env.items():
        if _looks_secret_like(key):
            return True
        if isinstance(value, str):
            if "${" in value and "}" in value:
                return True
            if _looks_secret_like(value):
                return True
    return False


def _looks_secret_like(value: Any) -> bool:
    lowered = str(value).lower()
    return any(token in lowered for token in ("secret", "token", "api_key", "apikey", "password", "credential"))


def _coerce_int(value: Any) -> int:
    if isinstance(value, bool):
        return int(value)
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _validate_step_sequence(
    steps: List[dict],
    *,
    adapter_ids: List[str],
    owner_label: str,
    errors: List[str],
) -> None:
    errors.extend(
        issue.message
        for issue in collect_step_sequence_issues(
            steps,
            owner_label=owner_label,
            adapter_ids=adapter_ids,
        )
    )
