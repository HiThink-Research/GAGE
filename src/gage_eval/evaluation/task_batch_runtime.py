"""Task-batch runtime primitives for external harness tasks."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass, field
import inspect
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

from loguru import logger

from gage_eval.config.pipeline_config import (
    CustomPipelineStep,
    DatasetSpec,
    PipelineConfig,
    TaskSpec,
)
from gage_eval.evaluation.cache import EvalCache
from gage_eval.evaluation.sample_loop import SampleLoop
from gage_eval.evaluation.task_plan import TaskPlanSpec
from gage_eval.evaluation.task_planner import TaskPlanner
from gage_eval.external_harness_kits.archive import safe_archive_segment
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.registry import import_asset_from_manifest, registry
from gage_eval.role.role_manager import RoleManager


@dataclass
class SampleLoopRuntimeEntry:
    """Runtime entry for the original per-sample path."""

    task_id: str
    dataset_id: str
    dataset_metadata: Dict[str, Any]
    sample_loop: SampleLoop
    task_planner: TaskPlanner
    reporting: Dict[str, Any]

    @property
    def processed_count(self) -> int:
        return int(self.sample_loop.processed_count)

    @property
    def shuffle_summary(self) -> Any:
        return self.sample_loop.shuffle_summary

    @property
    def report_partial_on_failure(self) -> bool:
        return bool(self.sample_loop.report_partial_on_failure)

    def shutdown(self) -> None:
        self.sample_loop.shutdown()


@dataclass(frozen=True)
class TaskBatchExecutionOutcome:
    """Summary of one task-batch task execution."""

    status: str
    task_id: str
    sample_count: int = 0
    metrics: Sequence[Dict[str, Any]] = field(default_factory=tuple)
    failed_step: Optional[str] = None
    failed_step_index: Optional[int] = None
    failure: Optional[Dict[str, Any]] = None

    def to_summary_payload(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "status": self.status,
            "sample_count": self.sample_count,
        }
        if self.failed_step is not None:
            payload["failed_step"] = self.failed_step
        if self.failed_step_index is not None:
            payload["failed_step_index"] = self.failed_step_index
        if self.failure is not None:
            payload["failure"] = dict(self.failure)
        return payload


class TaskBatchExecutionError(RuntimeError):
    """Raised when a task-batch step aborts its task."""

    def __init__(
        self,
        message: str,
        *,
        outcome: TaskBatchExecutionOutcome,
        cause: BaseException,
    ) -> None:
        self.outcome = outcome
        self.__cause__ = cause
        super().__init__(message)


@dataclass
class TaskBatchExecutionContext:
    """Task-scoped context passed to task-level steps."""

    config: PipelineConfig
    task_plan: TaskPlanSpec
    task_spec: TaskSpec
    dataset_spec: DatasetSpec
    role_manager: RoleManager
    trace: ObservabilityTrace
    cache_store: EvalCache
    registry: Any = None
    state: Dict[str, Any] = field(default_factory=dict)

    @property
    def task_id(self) -> str:
        return self.task_plan.task_id

    @property
    def dataset_id(self) -> str:
        return self.task_plan.dataset_id

    @property
    def workdir(self) -> Path:
        return self.cache_store.run_dir / "external_harness" / self.task_id

    @property
    def external_harness_root(self) -> Path:
        return self.cache_store.run_dir / "external_harness"

    @property
    def external_harness_manifest_path(self) -> Path:
        return self.external_harness_root / "manifest.json"

    def adapter_workdir(self, adapter_id: str) -> Path:
        return self.workdir / safe_archive_segment(adapter_id)

    def store(self, key: str, value: Any) -> None:
        self.state[str(key)] = value

    def load(self, key: str, default: Any = None) -> Any:
        return self.state.get(str(key), default)

    def get_task_batch_harness_adapter(self, adapter_id: str):
        adapter = self.role_manager.get_task_batch_harness_adapter(adapter_id)
        if adapter is None:
            raise KeyError(
                f"Role adapter '{adapter_id}' is not registered as a task-batch harness adapter"
            )
        return adapter

    def request_payload(self, *, adapter_id: Optional[str] = None) -> Dict[str, Any]:
        role_adapter = None
        if adapter_id:
            role_adapter = next(
                (
                    item
                    for item in self.config.role_adapters
                    if item.adapter_id == adapter_id
                ),
                None,
            )
        workdir = self.adapter_workdir(adapter_id) if adapter_id else self.workdir
        return {
            "run_id": self.trace.run_id,
            "python": None,
            "external_harness_root": str(self.external_harness_root),
            "external_harness_manifest_path": str(self.external_harness_manifest_path),
            "workdir": str(workdir),
            "jobs_dir": str(workdir / "jobs"),
            "job_config_path": str(workdir / "harbor_job.json"),
            "task": _dataclass_to_dict(self.task_spec),
            "dataset": _dataclass_to_dict(self.dataset_spec),
            "role_adapter": _dataclass_to_dict(role_adapter) if role_adapter else {},
            "datasets": [_dataclass_to_dict(item) for item in self.config.datasets],
            "backends": [_dataclass_to_dict(item) for item in self.config.backends],
            "environments": [_environment_to_dict(item) for item in self.config.environments],
        }


@dataclass
class TaskBatchRuntimeEntry:
    """Runtime entry that executes task-level steps without DataManager/SampleLoop."""

    task_id: str
    dataset_id: str
    dataset_metadata: Dict[str, Any]
    task_plan: TaskPlanSpec
    task_spec: TaskSpec
    dataset_spec: DatasetSpec
    steps: Sequence[CustomPipelineStep]
    reporting: Dict[str, Any]
    config: PipelineConfig
    registry: Any
    cache_store: EvalCache
    _processed_count: int = 0
    _last_outcome: Optional[TaskBatchExecutionOutcome] = None

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def shuffle_summary(self) -> None:
        return None

    @property
    def report_partial_on_failure(self) -> bool:
        value = self.task_plan.runtime_policy.report_partial_on_failure
        return True if value is None else bool(value)

    @property
    def failure_policy(self) -> Optional[str]:
        return self.task_plan.runtime_policy.failure_policy

    def run(
        self,
        *,
        role_manager: RoleManager,
        trace: ObservabilityTrace,
    ) -> TaskBatchExecutionOutcome:
        context = TaskBatchExecutionContext(
            config=self.config,
            task_plan=self.task_plan,
            task_spec=self.task_spec,
            dataset_spec=self.dataset_spec,
            role_manager=role_manager,
            trace=trace,
            cache_store=self.cache_store,
            registry=self.registry,
        )
        sample_count_before = self.cache_store.sample_count
        metrics: list[Dict[str, Any]] = []
        for step_index, step in enumerate(self.steps):
            step_type = step.step_type
            step_start = time.perf_counter()
            trace.emit(
                "step_execution_started",
                {
                    "step_type": step_type,
                    "task_id": self.task_id,
                    "step_index": step_index,
                    "step_kind": "task",
                    "expected_trial_count": _expected_trial_count(
                        self.task_spec,
                        adapter_spec=_role_adapter_spec(self.config, step.adapter_id),
                    ),
                },
                sample_id=None,
            )
            try:
                result = self._execute_step(step, context, step_index=step_index)
                metrics.extend(_metrics_from_step_result(result))
            except Exception as exc:
                elapsed_s = time.perf_counter() - step_start
                trace.emit(
                    "step_execution_failed",
                    {
                        "step_type": step_type,
                        "task_id": self.task_id,
                        "step_index": step_index,
                        "step_kind": "task",
                        "elapsed_s": elapsed_s,
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                    sample_id=None,
                )
                outcome = self._failure_outcome(
                    exc,
                    failed_step=step_type,
                    failed_step_index=step_index,
                    sample_count_before=sample_count_before,
                    metrics=metrics,
                )
                self._last_outcome = outcome
                if _is_best_effort(self.failure_policy):
                    logger.warning(
                        "Task-batch task '{}' failed at step '{}' but failure_policy=best_effort; continuing.",
                        self.task_id,
                        step_type,
                    )
                    return outcome
                raise TaskBatchExecutionError(
                    f"Task-batch task '{self.task_id}' failed at step '{step_type}': {exc}",
                    outcome=outcome,
                    cause=exc,
                ) from exc
            else:
                elapsed_s = time.perf_counter() - step_start
                payload = {
                    "step_type": step_type,
                    "task_id": self.task_id,
                    "step_index": step_index,
                    "step_kind": "task",
                    "elapsed_s": elapsed_s,
                }
                payload.update(_completion_fields_from_step_result(result))
                trace.emit("step_execution_completed", payload, sample_id=None)
        self._processed_count = max(0, self.cache_store.sample_count - sample_count_before)
        outcome = TaskBatchExecutionOutcome(
            status="completed",
            task_id=self.task_id,
            sample_count=self._processed_count,
            metrics=tuple(metrics),
        )
        self._last_outcome = outcome
        return outcome

    def shutdown(self) -> None:
        return None

    def _execute_step(
        self,
        step: CustomPipelineStep,
        context: TaskBatchExecutionContext,
        *,
        step_index: int,
    ) -> Any:
        step_obj = _instantiate_task_step(
            _resolve_task_step_class(self.registry, step.step_type),
            step=step,
            cache_store=self.cache_store,
            registry_view=getattr(self.registry, "registry_view", None),
        )
        execute = getattr(step_obj, "execute_task", None) or getattr(step_obj, "execute", None)
        if not callable(execute):
            raise TypeError(f"Task step '{step.step_type}' does not define execute_task()")
        result = _invoke_with_supported_kwargs(
            execute,
            context,
            step=step,
            step_spec=step,
            step_index=step_index,
            trace=context.trace,
        )
        if inspect.isawaitable(result):
            return asyncio.run(result)
        return result

    def _failure_outcome(
        self,
        exc: BaseException,
        *,
        failed_step: str,
        failed_step_index: int,
        sample_count_before: int,
        metrics: Sequence[Dict[str, Any]],
    ) -> TaskBatchExecutionOutcome:
        self._processed_count = max(0, self.cache_store.sample_count - sample_count_before)
        return TaskBatchExecutionOutcome(
            status="failed",
            task_id=self.task_id,
            sample_count=self._processed_count,
            metrics=tuple(metrics),
            failed_step=failed_step,
            failed_step_index=failed_step_index,
            failure={
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
        )


def _resolve_task_step_class(config_registry: Any, step_type: str):
    lookup = getattr(config_registry, "registry_view", None) or registry
    try:
        return lookup.get("pipeline_steps", step_type)
    except KeyError:
        if lookup is registry:
            import_asset_from_manifest(
                "pipeline_steps",
                step_type,
                registry=registry,
                source=f"task_step:{step_type}",
            )
            return registry.get("pipeline_steps", step_type)
        raise


def _instantiate_task_step(
    step_cls,
    *,
    step: CustomPipelineStep,
    cache_store: EvalCache,
    registry_view: Any,
):
    signature = inspect.signature(step_cls)
    kwargs: Dict[str, Any] = {}
    accepts_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    candidates = {
        "adapter_id": step.adapter_id,
        "params": dict(step.params or {}),
        "step_type": step.step_type,
        "cache_store": cache_store,
        "registry_view": registry_view,
    }
    for key, value in candidates.items():
        if value is not None and (accepts_kwargs or key in signature.parameters):
            kwargs[key] = value
    return step_cls(**kwargs)


def _invoke_with_supported_kwargs(method, context: TaskBatchExecutionContext, **kwargs):
    signature = inspect.signature(method)
    accepts_kwargs = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    call_kwargs = {
        key: value
        for key, value in kwargs.items()
        if accepts_kwargs or key in signature.parameters
    }
    return method(context, **call_kwargs)


def _dataclass_to_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}
    if hasattr(value, "__dataclass_fields__"):
        return asdict(value)
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _environment_to_dict(value: Any) -> Dict[str, Any]:
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        result = to_dict()
        return dict(result) if isinstance(result, Mapping) else {}
    return _dataclass_to_dict(value)


def _metrics_from_step_result(result: Any) -> Sequence[Dict[str, Any]]:
    if not isinstance(result, Mapping):
        return ()
    metrics = result.get("metrics")
    if isinstance(metrics, Sequence) and not isinstance(metrics, (str, bytes)):
        return tuple(dict(item) for item in metrics if isinstance(item, Mapping))
    return ()


def _completion_fields_from_step_result(result: Any) -> Dict[str, Any]:
    if not isinstance(result, Mapping):
        return {}
    fields: Dict[str, Any] = {}
    for key in ("job_name", "produced_sample_count", "sample_count"):
        if key in result:
            fields[key] = result[key]
    return fields


def _role_adapter_spec(config: PipelineConfig, adapter_id: Optional[str]):
    if not adapter_id:
        return None
    return next(
        (
            spec
            for spec in config.role_adapters
            if spec.adapter_id == adapter_id
        ),
        None,
    )


def _expected_trial_count(task: TaskSpec, *, adapter_spec: Any = None) -> Optional[int]:
    if task.max_samples is None:
        return None
    trial_policy = getattr(adapter_spec, "trial_policy", None) or {}
    trials = trial_policy.get("trials") if isinstance(trial_policy, Mapping) else None
    try:
        return int(task.max_samples) * int(trials or 1)
    except (TypeError, ValueError):
        return None


def _is_best_effort(value: Optional[str]) -> bool:
    return str(value or "").strip().lower() == "best_effort"


__all__ = [
    "SampleLoopRuntimeEntry",
    "TaskBatchExecutionContext",
    "TaskBatchExecutionError",
    "TaskBatchExecutionOutcome",
    "TaskBatchRuntimeEntry",
]
