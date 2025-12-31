"""Static task planning helpers derived from PipelineConfig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

from gage_eval.config.pipeline_config import (
    CustomPipelineStep,
    DatasetSpec,
    MetricSpec,
    PipelineConfig,
    RoleAdapterSpec,
    TaskSpec,
)


@dataclass(frozen=True)
class TaskRuntimePolicy:
    max_samples: Optional[int]
    shuffle: Optional[bool]
    shuffle_seed: Optional[int]
    concurrency: Optional[int]
    prefetch_factor: Optional[int]
    max_inflight: Optional[int]


@dataclass(frozen=True)
class TaskPlanSpec:
    task_id: str
    dataset_id: str
    dataset_schema: Optional[Dict[str, Any]]
    steps: Sequence[CustomPipelineStep]
    role_bindings: Dict[str, RoleAdapterSpec]
    metrics: Sequence[MetricSpec]
    runtime_policy: TaskRuntimePolicy
    reporting: Dict[str, Any]


def build_task_plan_specs(config: PipelineConfig) -> Sequence[TaskPlanSpec]:
    """Derive TaskPlanSpec entries from the validated PipelineConfig."""

    dataset_map: Dict[str, DatasetSpec] = {dataset.dataset_id: dataset for dataset in config.datasets}
    role_map: Dict[str, RoleAdapterSpec] = {adapter.adapter_id: adapter for adapter in config.role_adapters}
    default_steps: Sequence[CustomPipelineStep] = config.custom.steps if config.custom else ()
    plan_specs: List[TaskPlanSpec] = []
    for task in config.tasks:
        plan_specs.append(
            _build_task_plan(
                task=task,
                dataset_map=dataset_map,
                role_map=role_map,
                default_steps=default_steps,
                default_metrics=config.metrics,
            )
        )
    return tuple(plan_specs)


def _build_task_plan(
    *,
    task: TaskSpec,
    dataset_map: Dict[str, DatasetSpec],
    role_map: Dict[str, RoleAdapterSpec],
    default_steps: Sequence[CustomPipelineStep],
    default_metrics: Sequence[MetricSpec],
) -> TaskPlanSpec:
    dataset = dataset_map.get(task.dataset_id)
    if dataset is None:
        raise KeyError(f"Task '{task.task_id}' references unknown dataset '{task.dataset_id}'")
    steps = task.steps or default_steps
    if not steps:
        raise ValueError(
            f"Task '{task.task_id}' must declare steps explicitly or via custom pipeline"
        )

    # NOTE: Auto-fill missing `adapter_id` for common cases (currently focused on inference steps).
    resolved_steps = _infer_step_bindings(steps, role_map, task_id=task.task_id)

    role_bindings: Dict[str, RoleAdapterSpec] = {}
    for step in resolved_steps:
        adapter_id = step.adapter_id
        if not adapter_id:
            continue
        if adapter_id not in role_map:
            raise KeyError(
                f"Task '{task.task_id}' references role adapter '{adapter_id}' which is not defined"
            )
        role_bindings[adapter_id] = role_map[adapter_id]
    metrics = task.metric_overrides or default_metrics
    runtime_policy = TaskRuntimePolicy(
        max_samples=task.max_samples,
        shuffle=task.shuffle,
        shuffle_seed=task.shuffle_seed,
        concurrency=task.concurrency,
        prefetch_factor=task.prefetch_factor,
        max_inflight=task.max_inflight,
    )
    return TaskPlanSpec(
        task_id=task.task_id,
        dataset_id=task.dataset_id,
        dataset_schema=dataset.schema,
        steps=resolved_steps,
        role_bindings=role_bindings,
        metrics=metrics,
        runtime_policy=runtime_policy,
        reporting=task.reporting or {},
    )


def _infer_step_bindings(
    steps: Sequence[CustomPipelineStep],
    role_map: Dict[str, RoleAdapterSpec],
    *,
    task_id: str,
) -> Sequence[CustomPipelineStep]:
    """Infer adapter bindings for steps that omit adapter_id.

    The current implementation targets the most common case: if there is exactly
    one DUT role adapter, the `inference` step may omit `adapter_id` and will be
    automatically bound to that DUT adapter.
    """

    # STEP 1: If all steps specify adapter_id explicitly, keep them unchanged.
    if all(step.adapter_id for step in steps):
        return steps

    resolved: List[CustomPipelineStep] = []
    # STEP 2: Resolve the unique DUT role adapter (role_type='dut_model') if possible.
    dut_adapters = [spec for spec in role_map.values() if spec.role_type == "dut_model"]
    inferred_dut_id: Optional[str] = None
    if len(dut_adapters) == 1:
        inferred_dut_id = dut_adapters[0].adapter_id

    # STEP 3: Walk steps and infer bindings only for `inference` when safe.
    for step in steps:
        if step.adapter_id:
            resolved.append(step)
            continue
        # Only infer for `inference`; keep other step types unchanged.
        if step.step_type == "inference":
            if inferred_dut_id is None:
                if not dut_adapters:
                    raise ValueError(
                        f"Task '{task_id}' cannot infer adapter for inference step: "
                        "no RoleAdapter with role_type='dut_model' found"
                    )
                adapter_ids = ", ".join(spec.adapter_id for spec in dut_adapters)
                raise ValueError(
                    f"Task '{task_id}' inference step is ambiguous: multiple DUT role adapters "
                    f"found for role_type='dut_model' ({adapter_ids}); please set adapter_id explicitly"
                )
            resolved.append(
                CustomPipelineStep(
                    step_type=step.step_type,
                    adapter_id=inferred_dut_id,
                    params=step.params,
                )
            )
        else:
            resolved.append(step)

    # STEP 4: Return the resolved step list.
    return tuple(resolved)
