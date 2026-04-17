"""Static-evaluation smart-default rules."""

from __future__ import annotations

from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path
from typing import Any

from gage_eval.config.smart_defaults.registry import smart_default
from gage_eval.config.smart_defaults.types import DefaultTrace, RuleContext

_DATASET_HUB_PARAM_KEYS = ("hub_id", "split", "subset", "revision", "data_files")
_LITELLM_API_BASES = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
}


def _iter_indexed_dicts(value: Any) -> Iterator[tuple[int, dict[str, Any]]]:
    if not isinstance(value, list):
        return
    for index, item in enumerate(value):
        if isinstance(item, dict):
            yield index, item


def _iter_dicts(value: Any) -> Iterator[dict[str, Any]]:
    for _, item in _iter_indexed_dicts(value):
        yield item


def _backend_ids(payload: dict[str, Any]) -> list[str]:
    return [
        str(item.get("backend_id"))
        for item in _iter_dicts(payload.get("backends"))
        if item.get("backend_id")
    ]


def _metadata_name(payload: dict[str, Any]) -> str:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return str(metadata.get("name") or "static_task")


@smart_default(
    scene="static",
    phase="dataset",
    priority=10,
    name="dataset_hub_from_hub_id",
    description="Fill dataset.hub when dataset.hub_id is used as sugar.",
)
def dataset_hub_from_hub_id(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, dataset in _iter_indexed_dicts(payload.get("datasets")):
        if dataset.get("hub_id"):
            ctx.fill(dataset, key="hub", value="huggingface", path=f"datasets[{index}].hub")


@smart_default(
    scene="static",
    phase="dataset",
    priority=10,
    name="dataset_loader_from_hub_id",
    description="Fill dataset.loader=hf_hub when dataset.hub_id is used as sugar.",
)
def dataset_loader_from_hub_id(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, dataset in _iter_indexed_dicts(payload.get("datasets")):
        if dataset.get("hub_id"):
            ctx.fill(dataset, key="loader", value="hf_hub", path=f"datasets[{index}].loader")


@smart_default(
    scene="static",
    phase="dataset",
    priority=20,
    name="dataset_loader_from_path",
    description="Infer json/jsonl loader from local dataset path.",
)
def dataset_loader_from_path(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, dataset in _iter_indexed_dicts(payload.get("datasets")):
        if dataset.get("loader"):
            continue
        params = dataset.get("params") if isinstance(dataset.get("params"), dict) else {}
        raw_path = dataset.get("path") or params.get("path")
        if not raw_path:
            continue
        suffix = Path(str(raw_path)).suffix.lower()
        if suffix == ".jsonl":
            ctx.fill(dataset, key="loader", value="jsonl", path=f"datasets[{index}].loader")
        elif suffix == ".json":
            ctx.fill(dataset, key="loader", value="json", path=f"datasets[{index}].loader")


@smart_default(
    scene="static",
    phase="dataset",
    priority=30,
    name="dataset_hub_params_gather",
    description="Move dataset hub sugar keys into hub_params.",
)
def dataset_hub_params_gather(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, dataset in _iter_indexed_dicts(payload.get("datasets")):
        present_keys = [key for key in _DATASET_HUB_PARAM_KEYS if key in dataset]
        if not present_keys:
            continue

        hub_params = dataset.get("hub_params")
        if hub_params is None:
            hub_params = {}
            dataset["hub_params"] = hub_params
        if not isinstance(hub_params, dict):
            ctx.fail("dataset.hub_params must be a mapping", path=f"datasets[{index}].hub_params")

        for key in present_keys:
            ctx.migrate(
                dataset,
                source_key=key,
                target=hub_params,
                target_key=key,
                path=f"datasets[{index}].hub_params.{key}",
            )


@smart_default(
    scene="static",
    phase="dataset",
    priority=40,
    name="dataset_preprocess_kwargs_default",
    description="Fill params.preprocess_kwargs when params.preprocess is declared.",
)
def dataset_preprocess_kwargs_default(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, dataset in _iter_indexed_dicts(payload.get("datasets")):
        params = dataset.get("params")
        if isinstance(params, dict) and params.get("preprocess"):
            ctx.fill(
                params,
                key="preprocess_kwargs",
                value={},
                path=f"datasets[{index}].params.preprocess_kwargs",
            )


@smart_default(
    scene="static",
    phase="backend",
    priority=10,
    name="vllm_tokenizer_path_from_model_path",
    description="Fill tokenizer_path from model_path for vLLM configs.",
)
def vllm_tokenizer_path_from_model_path(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, backend in _iter_indexed_dicts(payload.get("backends")):
        if backend.get("type") != "vllm" or not isinstance(backend.get("config"), dict):
            continue
        config = backend["config"]
        if config.get("model_path"):
            ctx.fill(
                config,
                key="tokenizer_path",
                value=config["model_path"],
                path=f"backends[{index}].config.tokenizer_path",
            )


@smart_default(
    scene="static",
    phase="backend",
    priority=20,
    name="vllm_force_tokenize_prompt_default",
    description="Fill vLLM force_tokenize_prompt default.",
)
def vllm_force_tokenize_prompt_default(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, backend in _iter_indexed_dicts(payload.get("backends")):
        if backend.get("type") == "vllm" and isinstance(backend.get("config"), dict):
            ctx.fill(
                backend["config"],
                key="force_tokenize_prompt",
                value=True,
                path=f"backends[{index}].config.force_tokenize_prompt",
            )


@smart_default(
    scene="static",
    phase="backend",
    priority=20,
    name="vllm_tokenizer_trust_remote_code_default",
    description="Fill vLLM tokenizer_trust_remote_code default.",
)
def vllm_tokenizer_trust_remote_code_default(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, backend in _iter_indexed_dicts(payload.get("backends")):
        if backend.get("type") == "vllm" and isinstance(backend.get("config"), dict):
            ctx.fill(
                backend["config"],
                key="tokenizer_trust_remote_code",
                value=True,
                path=f"backends[{index}].config.tokenizer_trust_remote_code",
            )


@smart_default(
    scene="static",
    phase="backend",
    priority=10,
    name="litellm_provider_from_api_base",
    description="Infer LiteLLM provider from api_base.",
)
def litellm_provider_from_api_base(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, backend in _iter_indexed_dicts(payload.get("backends")):
        if backend.get("type") != "litellm" or not isinstance(backend.get("config"), dict):
            continue
        config = backend["config"]
        api_base = str(config.get("api_base") or "")
        if "api.openai.com" in api_base:
            ctx.fill(
                config,
                key="provider",
                value="openai",
                path=f"backends[{index}].config.provider",
            )
        elif "api.deepseek.com" in api_base:
            ctx.fill(
                config,
                key="provider",
                value="deepseek",
                path=f"backends[{index}].config.provider",
            )


@smart_default(
    scene="static",
    phase="backend",
    priority=10,
    name="litellm_api_base_from_provider",
    description="Fill LiteLLM api_base from known providers.",
)
def litellm_api_base_from_provider(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, backend in _iter_indexed_dicts(payload.get("backends")):
        if backend.get("type") != "litellm" or not isinstance(backend.get("config"), dict):
            continue
        config = backend["config"]
        provider = str(config.get("provider") or "").lower()
        api_base = _LITELLM_API_BASES.get(provider)
        if api_base:
            ctx.fill(
                config,
                key="api_base",
                value=api_base,
                path=f"backends[{index}].config.api_base",
            )


@smart_default(
    scene="static",
    phase="backend",
    priority=30,
    name="litellm_streaming_default",
    description="Fill LiteLLM streaming default.",
)
def litellm_streaming_default(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, backend in _iter_indexed_dicts(payload.get("backends")):
        if backend.get("type") == "litellm" and isinstance(backend.get("config"), dict):
            ctx.fill(
                backend["config"],
                key="streaming",
                value=False,
                path=f"backends[{index}].config.streaming",
            )


@smart_default(
    scene="static",
    phase="backend",
    priority=30,
    name="litellm_max_retries_default",
    description="Fill LiteLLM max_retries default.",
)
def litellm_max_retries_default(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, backend in _iter_indexed_dicts(payload.get("backends")):
        if backend.get("type") == "litellm" and isinstance(backend.get("config"), dict):
            ctx.fill(
                backend["config"],
                key="max_retries",
                value=6,
                path=f"backends[{index}].config.max_retries",
            )


@smart_default(
    scene="static",
    phase="role_adapter",
    priority=20,
    name="auto_dut_role_adapters",
    description="Generate DUT role adapters from backends when adapters are omitted.",
)
def auto_dut_role_adapters(payload: dict[str, Any], ctx: RuleContext) -> None:
    existing = payload.get("role_adapters")
    if existing:
        return

    adapters = [
        {
            "adapter_id": f"dut_{backend_id}",
            "role_type": "dut_model",
            "backend_id": backend_id,
            "capabilities": ["chat_completion"],
        }
        for backend_id in _backend_ids(payload)
    ]
    if adapters:
        payload["role_adapters"] = adapters
        ctx.traces.append(
            DefaultTrace(rule=ctx.current_rule, action="replace_subtree", path="role_adapters")
        )


@smart_default(
    scene="static",
    phase="custom_steps",
    priority=20,
    name="auto_custom_steps",
    description="Generate inference/auto_eval custom steps for pure DUT configs.",
)
def auto_custom_steps(payload: dict[str, Any], ctx: RuleContext) -> None:
    custom = payload.get("custom")
    if custom is not None and not isinstance(custom, dict):
        ctx.fail("custom must be a mapping", path="custom")
    custom = custom or {}
    if "steps" in custom:
        return

    adapters = list(_iter_dicts(payload.get("role_adapters")))
    if not adapters or any(adapter.get("role_type") != "dut_model" for adapter in adapters):
        return

    custom["steps"] = [{"step": "inference"}, {"step": "auto_eval"}]
    payload["custom"] = custom
    ctx.traces.append(DefaultTrace(rule=ctx.current_rule, action="fill", path="custom.steps"))


@smart_default(
    scene="static",
    phase="task",
    priority=5,
    name="single_task_fallback",
    description="Generate a single task for one dataset and one DUT adapter.",
)
def single_task_fallback(payload: dict[str, Any], ctx: RuleContext) -> None:
    if "tasks" in payload or "task" in payload:
        return

    datasets = list(_iter_dicts(payload.get("datasets")))
    adapters = [
        item
        for item in _iter_dicts(payload.get("role_adapters"))
        if item.get("role_type") == "dut_model"
    ]
    if len(datasets) == 1 and len(adapters) == 1:
        payload["tasks"] = [
            {"task_id": _metadata_name(payload), "dataset_id": datasets[0].get("dataset_id")}
        ]
        ctx.traces.append(
            DefaultTrace(rule=ctx.current_rule, action="replace_subtree", path="tasks")
        )


@smart_default(
    scene="static",
    phase="task",
    priority=6,
    name="task_singular_alias",
    description="Lower top-level task to tasks list.",
)
def task_singular_alias(payload: dict[str, Any], ctx: RuleContext) -> None:
    if "task" in payload and "tasks" in payload:
        ctx.fail("task and tasks cannot both be declared")
    if "task" not in payload:
        return

    task = payload.pop("task")
    if not isinstance(task, dict):
        ctx.fail("task must be a mapping", path="task")
    payload["tasks"] = [task]
    ctx.traces.append(DefaultTrace(rule=ctx.current_rule, action="replace_subtree", path="tasks"))


@smart_default(
    scene="static",
    phase="task",
    priority=10,
    name="task_implicit_ids",
    description="Fill task_id and dataset_id for single-task static configs.",
)
def task_implicit_ids(payload: dict[str, Any], ctx: RuleContext) -> None:
    datasets = list(_iter_dicts(payload.get("datasets")))
    for index, task in _iter_indexed_dicts(payload.get("tasks")):
        ctx.fill(task, key="task_id", value=_metadata_name(payload), path=f"tasks[{index}].task_id")
        if "dataset_id" in task:
            continue
        if len(datasets) != 1:
            ctx.fail(
                "task omitted dataset_id but config does not have exactly one dataset",
                path=f"tasks[{index}]",
            )
        task["dataset_id"] = datasets[0].get("dataset_id")
        ctx.traces.append(
            DefaultTrace(rule=ctx.current_rule, action="fill", path=f"tasks[{index}].dataset_id")
        )


@smart_default(
    scene="static",
    phase="task",
    priority=15,
    name="task_backend_from_single_dut",
    description="Fill task.backend when there is exactly one DUT adapter.",
)
def task_backend_from_single_dut(payload: dict[str, Any], ctx: RuleContext) -> None:
    if getattr(ctx.cli_intent, "backend_id", None):
        return

    dut_adapters = [
        item
        for item in _iter_dicts(payload.get("role_adapters"))
        if item.get("role_type") == "dut_model"
    ]
    for index, task in _iter_indexed_dicts(payload.get("tasks")):
        if task.get("backend"):
            continue
        if len(dut_adapters) != 1:
            ctx.fail(
                "task omitted backend but config does not have exactly one DUT adapter",
                path=f"tasks[{index}]",
            )
        backend_id = dut_adapters[0].get("backend_id")
        if not backend_id:
            ctx.fail("single DUT adapter is missing backend_id", path=f"tasks[{index}]")
        task["backend"] = backend_id
        ctx.traces.append(
            DefaultTrace(rule=ctx.current_rule, action="fill", path=f"tasks[{index}].backend")
        )


def _find_unique_dut_adapter(
    payload: dict[str, Any], backend_id: str, ctx: RuleContext, *, path: str
) -> str:
    matches = [
        adapter.get("adapter_id")
        for adapter in _iter_dicts(payload.get("role_adapters"))
        if adapter.get("role_type") == "dut_model" and adapter.get("backend_id") == backend_id
    ]
    adapter_ids = [str(item) for item in matches if item]
    if len(adapter_ids) != 1:
        ctx.fail(f"cannot find unique DUT adapter for backend '{backend_id}'", path=path)
    return adapter_ids[0]


@smart_default(
    scene="static",
    phase="task",
    priority=20,
    name="task_backend_expand",
    description="Lower task.backend or CLI backend_id to inference adapter binding.",
)
def task_backend_expand(payload: dict[str, Any], ctx: RuleContext) -> None:
    custom = payload.get("custom") if isinstance(payload.get("custom"), dict) else {}
    custom_steps = custom.get("steps") if isinstance(custom, dict) else []

    for task_index, task in _iter_indexed_dicts(payload.get("tasks")):
        cli_backend = getattr(ctx.cli_intent, "backend_id", None)
        if cli_backend:
            target_backend = cli_backend
            target_backend_path = "cli.backend_id"
        else:
            target_backend = task.get("backend")
            target_backend_path = f"tasks[{task_index}].backend"
        if not target_backend:
            continue

        target_adapter = _find_unique_dut_adapter(
            payload, str(target_backend), ctx, path=target_backend_path
        )
        if not task.get("steps"):
            if not custom_steps:
                ctx.fail(
                    "task specified backend but no custom.steps or task.steps are available",
                    path=f"tasks[{task_index}]",
            )
            task["steps"] = deepcopy(custom_steps)
        inference_steps = [
            step
            for step in task.get("steps") or []
            if isinstance(step, dict) and step.get("step") == "inference"
        ]
        if len(inference_steps) != 1:
            ctx.fail(
                "task must have exactly one inference step to bind backend",
                path=f"tasks[{task_index}].steps",
            )

        inference = inference_steps[0]
        existing = inference.get("adapter_id")
        if existing and existing != target_adapter:
            ctx.fail(
                "inference step is already bound to a different adapter",
                path=f"tasks[{task_index}].steps",
            )
        inference["adapter_id"] = target_adapter
        task.pop("backend", None)


@smart_default(
    scene="static",
    phase="task",
    priority=30,
    name="task_reporting_default",
    description="Fill standard console/file reporting sinks.",
)
def task_reporting_default(payload: dict[str, Any], ctx: RuleContext) -> None:
    for index, task in _iter_indexed_dicts(payload.get("tasks")):
        ctx.fill(
            task,
            key="reporting",
            value={"sinks": [{"type": "console"}, {"type": "file"}]},
            path=f"tasks[{index}].reporting",
        )
