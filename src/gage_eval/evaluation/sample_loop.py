"""Sample loop implementation modelled after llm-eval's data server."""

from __future__ import annotations

import os
from pathlib import Path
import random
import threading
import time
from concurrent.futures import ALL_COMPLETED, FIRST_COMPLETED, Future, wait
from contextvars import copy_context
from queue import Empty, Full, Queue
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Set,
    Union,
)

from loguru import logger

from gage_eval.evaluation.execution_controller import (
    FailurePolicy,
    SampleLoopExecutionError,
    SampleLoopOutcome,
    TaskExecutionController,
)
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.pipeline.step_contracts import get_step_contract_catalog
from gage_eval.evaluation.sample_ingress import resolve_runtime_sample_id
from gage_eval.evaluation.shuffle_store import (
    iter_external_shuffle_samples,
    iter_reservoir_samples,
    try_resolve_length,
)
from gage_eval.evaluation.task_planner import TaskPlanner, TaskPlan
from gage_eval.role.role_manager import RoleManager
from gage_eval.role.runtime.invocation import (
    RoleSessionStore,
    SampleExecutionContext,
)
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.session_router import SandboxSessionRouter
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope
from gage_eval.assets.datasets.sample import Sample, Message, MessageContent

_DEFAULT_SHUFFLE_STRATEGY = "auto"
_DEFAULT_SHUFFLE_SMALL_DATASET_THRESHOLD = 20_000


class SampleLoop:
    """Iterate over samples and invoke TaskPlanner/RoleManager per sample."""

    def __init__(
        self,
        samples: Iterable[Union[dict, Sample]],
        *,
        shuffle: Optional[bool] = None,
        shuffle_seed: Optional[int] = None,
        shuffle_strategy: Optional[str] = None,
        shuffle_small_dataset_threshold: Optional[int] = None,
        keep_shuffle_artifacts: Optional[bool] = None,
        max_samples: Optional[int] = None,
        concurrency: Optional[int] = None,
        streaming: bool = False,
        task_id: Optional[str] = None,
        shuffle_artifact_root: Optional[Path] = None,
        prefetch_factor: Optional[int] = None,
        max_inflight: Optional[int] = None,
        failure_policy: Optional[str] = None,
        report_partial_on_failure: Optional[bool] = None,
        sandbox_manager: Optional[SandboxManager] = None,
        sandbox_profiles: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> None:
        self._samples = samples
        self._hooks: List[Callable[[dict], None]] = []
        self._custom_steps: Sequence[dict] = ()
        self._shuffle = (
            shuffle
            if shuffle is not None
            else _env_flag("GAGE_EVAL_SHUFFLE", default=False)
        )
        self._shuffle_seed = shuffle_seed or int(
            os.environ.get("GAGE_EVAL_SHUFFLE_SEED", "123")
        )
        self._shuffle_strategy = _normalize_shuffle_strategy(
            shuffle_strategy or os.environ.get("GAGE_EVAL_SHUFFLE_STRATEGY")
        )
        env_shuffle_threshold = _env_int("GAGE_EVAL_SHUFFLE_SMALL_DATASET_THRESHOLD")
        self._shuffle_small_dataset_threshold = max(
            1,
            shuffle_small_dataset_threshold
            if shuffle_small_dataset_threshold is not None
            else (
                env_shuffle_threshold
                if env_shuffle_threshold is not None
                else _DEFAULT_SHUFFLE_SMALL_DATASET_THRESHOLD
            ),
        )
        self._keep_shuffle_artifacts = (
            keep_shuffle_artifacts
            if keep_shuffle_artifacts is not None
            else _env_flag("GAGE_EVAL_KEEP_SHUFFLE_ARTIFACTS", default=False)
        )
        self._shuffle_artifact_root = (
            Path(shuffle_artifact_root).expanduser()
            if shuffle_artifact_root is not None
            else None
        )
        env_max_samples = os.environ.get("GAGE_EVAL_MAX_SAMPLES")
        self._max_samples = (
            max_samples
            if max_samples is not None
            else (int(env_max_samples) if env_max_samples else None)
        )
        env_threads = os.environ.get("GAGE_EVAL_THREADS")
        default_threads = min(os.cpu_count() or 1, 4)
        self._concurrency = max(
            1,
            concurrency
            if concurrency is not None
            else (int(env_threads) if env_threads else default_threads),
        )
        env_prefetch = _env_int("GAGE_EVAL_PREFETCH_FACTOR")
        env_inflight = _env_int("GAGE_EVAL_MAX_INFLIGHT")
        raw_prefetch = (
            prefetch_factor
            if prefetch_factor is not None
            else (env_prefetch if env_prefetch is not None else 2)
        )
        raw_max_inflight = (
            max_inflight
            if max_inflight is not None
            else (env_inflight if env_inflight is not None else self._concurrency)
        )
        self._prefetch_factor = max(1, raw_prefetch)
        self._max_inflight = max(1, raw_max_inflight)
        self._buffer_capacity = max(
            self._max_inflight, self._concurrency * self._prefetch_factor
        )
        self._sequential_override = _env_flag("GAGE_EVAL_SEQUENTIAL", default=False)
        self._legacy_ff_mode = (
            failure_policy is None
            and _env_flag("GAGE_EVAL_FF_MODE", default=False)
        )
        if failure_policy is not None:
            self._failure_policy = failure_policy
        elif self._legacy_ff_mode:
            self._failure_policy = "best_effort"
        else:
            self._failure_policy = None
        self._report_partial_on_failure = (
            True
            if report_partial_on_failure is None
            else bool(report_partial_on_failure)
        )
        self._materialized_samples: Optional[List[dict]] = None
        self._streaming = streaming
        self._task_id = task_id
        self._processed_count = 0
        self._processed_lock = threading.Lock()
        self._execution_controller: Optional[TaskExecutionController] = None
        self._shuffle_summary: Dict[str, Any] = {
            "enabled": bool(self._shuffle),
            "requested": self._shuffle_strategy if self._shuffle else "disabled",
            "resolved": "pending" if self._shuffle else "sequential",
            "seed": self._shuffle_seed if self._shuffle else None,
            "max_samples": self._max_samples,
            "small_dataset_threshold": (
                self._shuffle_small_dataset_threshold if self._shuffle else None
            ),
        }
        self._sandbox_profiles = sandbox_profiles or {}
        self._sandbox_manager = sandbox_manager or SandboxManager(
            profiles=self._sandbox_profiles
        )
        logger.info(
            "SampleLoop initialized (shuffle={}, strategy={}, max_samples={}, concurrency={}, streaming={}, task_id={})",
            self._shuffle,
            self._shuffle_strategy,
            self._max_samples,
            self._concurrency,
            self._streaming,
            self._task_id,
        )

    def register_hook(self, hook: Callable[[dict], None]) -> None:
        self._hooks.append(hook)

    def configure_custom_steps(self, steps: Sequence[dict]) -> None:
        self._custom_steps = steps

    def attach_execution_controller(
        self, controller: Optional[TaskExecutionController]
    ) -> None:
        self._execution_controller = controller

    def run(
        self, planner: TaskPlanner, role_manager: RoleManager, trace: ObservabilityTrace
    ) -> SampleLoopOutcome:
        work = self._iter_samples(trace)
        controller = self._get_execution_controller(planner)

        logger.info(
            "SampleLoop running with bounded buffer (workers={}, max_inflight={}, prefetch_factor={}, buffer_capacity={}, failure_policy={})",
            controller.sample_workers,
            self._max_inflight,
            self._prefetch_factor,
            self._buffer_capacity,
            controller.failure_policy.value,
        )
        sentinel = object()
        sample_queue: Queue = Queue(maxsize=self._buffer_capacity)
        stop_event = threading.Event()
        producer_errors: List[BaseException] = []
        producer_context = copy_context()
        producer = threading.Thread(
            target=producer_context.run,
            name="SamplePrefetcher",
            args=(self._prefetch_samples, work, sample_queue, sentinel, stop_event, trace, producer_errors),
            daemon=True,
        )
        producer.start()

        futures: Set[Future[Any]] = set()
        try:
            while True:
                if controller.should_stop_submitting():
                    stop_event.set()
                    break
                item = sample_queue.get()
                if item is sentinel:
                    break
                if controller.should_stop_submitting():
                    controller.record_queue_cancellations(1)
                    stop_event.set()
                    break
                logical_idx, sample = item
                preview_sample_id = self._preview_sample_id(logical_idx, sample)
                future = controller.submit_sample(
                    copy_context().run,
                    self._process_sample,
                    logical_idx,
                    sample,
                    planner,
                    role_manager,
                    trace,
                    sample_id=preview_sample_id,
                )
                futures.add(future)
                self._emit_buffer_state(trace, sample_queue, len(futures))
                if len(futures) >= self._max_inflight:
                    self._drain_futures(futures, sample_queue, trace)
            if controller.should_stop_submitting():
                if controller.failure_policy is not FailurePolicy.BEST_EFFORT:
                    controller.cancel_pending_samples()
                producer.join()
                controller.record_queue_cancellations(
                    self._drain_pending_queue(sample_queue, sentinel)
                )
            self._drain_futures(futures, sample_queue, trace, drain_all=True)
        finally:
            stop_event.set()
            try:
                sample_queue.put_nowait(sentinel)
            except Exception:
                pass
            producer.join()

        if producer_errors:
            controller.record_failure(None, producer_errors[0])

        outcome = controller.snapshot(
            processed_samples=self.processed_count,
            max_inflight=self._max_inflight,
        )
        if outcome.error_type is not None and controller.first_error is not None:
            raise SampleLoopExecutionError(outcome, controller.first_error)
        return outcome

    def shutdown(self) -> None:
        if self._execution_controller is not None:
            self._execution_controller.shutdown()
        if self._sandbox_manager:
            self._sandbox_manager.shutdown()

    def _execute_plan(
        self,
        sample: dict,
        plan: TaskPlan,
        role_manager: RoleManager,
        trace: ObservabilityTrace,
        sample_identifier: str,
    ) -> None:
        # STEP 1: Build sample-scoped execution context.
        execution_context = self._build_execution_context(
            plan,
            sample,
            trace,
            sample_identifier,
        )
        # STEP 2: Execute plan within a per-sample session.
        try:
            per_sample_ctx = plan.create_context(
                sample,
                trace,
                role_manager,
                execution_context=execution_context,
            )
            with (
                trace.use_sample(sample_identifier),
                role_manager.per_sample_session(per_sample_ctx) as session,
            ):
                if plan.steps:
                    for step in plan.steps:
                        self._dispatch_step(
                            session,
                            step,
                            plan,
                            role_manager,
                            trace,
                            sample_identifier,
                        )
                else:
                    if plan.support_steps:
                        session.execute_support()
                    if plan.inference_role:
                        session.execute_inference()
                    if plan.arena_role:
                        session.execute_arena()
                    if plan.judge_role:
                        session.execute_judge()
                    if plan.auto_eval_enabled:
                        session.execute_auto_eval(sample_identifier)
        finally:
            # STEP 3: Release all sample-scoped runtime resources.
            execution_context.close()

    def _build_execution_context(
        self,
        plan: TaskPlan,
        sample: dict,
        trace: ObservabilityTrace,
        sample_identifier: str,
    ) -> SampleExecutionContext:
        return SampleExecutionContext(
            sample=sample,
            sample_id=sample_identifier,
            run_id=trace.run_id,
            task_id=plan.metadata.get("task_id") or self._task_id,
            trace=trace,
            session_store=RoleSessionStore(sample),
            sandbox_router=SandboxSessionRouter(
                self._sandbox_manager,
                run_id=trace.run_id,
                task_id=plan.metadata.get("task_id") or self._task_id,
                sample_id=sample_identifier,
                trace=trace,
            ),
        )

    def _dispatch_step(
        self,
        session,
        step,
        plan: TaskPlan,
        role_manager: RoleManager,
        trace: ObservabilityTrace,
        sample_identifier: str,
    ) -> None:
        step_type = _resolve_step_type(step)
        catalog = get_step_contract_catalog()
        contract = catalog.get(step_type)
        payload = self._build_step_event_payload(
            step_type=step_type,
            plan=plan,
            step=step,
            sample_identifier=sample_identifier,
        )
        if contract is None:
            payload["error_type"] = "unsupported_step"
            trace.emit("step_execution_failed", payload, sample_id=sample_identifier)
            raise ValueError(f"Unsupported configured step '{step_type}'")
        if contract.step_kind.value != "sample":
            payload["error_type"] = "invalid_step_kind"
            trace.emit("step_execution_failed", payload, sample_id=sample_identifier)
            raise ValueError(
                f"Configured step '{step_type}' is global and cannot run inside sample execution"
            )
        executor_name = contract.executor_name
        if executor_name is None or not hasattr(session, executor_name):
            payload["error_type"] = "missing_executor"
            trace.emit("step_execution_failed", payload, sample_id=sample_identifier)
            raise ValueError(
                f"Configured step '{step_type}' has no sample executor mapping"
            )
        adapter_id = payload.get("adapter_id")
        if contract.requires_adapter:
            try:
                self._assert_runtime_adapter_binding(
                    step_type=step_type,
                    adapter_id=adapter_id,
                    role_manager=role_manager,
                )
            except Exception as exc:
                failure_payload = dict(payload)
                failure_payload["error_type"] = exc.__class__.__name__
                failure_payload["error"] = str(exc)
                trace.emit(
                    "step_execution_failed",
                    failure_payload,
                    sample_id=sample_identifier,
                )
                raise
        trace.emit("step_execution_started", payload, sample_id=sample_identifier)
        started_at = time.perf_counter()
        try:
            executor = getattr(session, executor_name)
            if step_type == "support":
                executor(step)
            elif step_type == "auto_eval":
                executor(sample_identifier)
            else:
                executor()
        except Exception as exc:
            failure_payload = dict(payload)
            failure_payload["error_type"] = exc.__class__.__name__
            failure_payload["error"] = str(exc)
            trace.emit(
                "step_execution_failed",
                failure_payload,
                sample_id=sample_identifier,
            )
            raise
        completed_payload = dict(payload)
        completed_payload["duration_ms"] = int(
            (time.perf_counter() - started_at) * 1000
        )
        trace.emit(
            "step_execution_completed",
            completed_payload,
            sample_id=sample_identifier,
        )

    def _assert_runtime_adapter_binding(
        self,
        *,
        step_type: Optional[str],
        adapter_id: Optional[str],
        role_manager: RoleManager,
    ) -> None:
        if not adapter_id:
            raise RuntimeError(
                f"Configured step '{step_type}' requires a resolved adapter_id at runtime"
            )
        if role_manager.get_adapter(adapter_id) is None:
            raise KeyError(
                f"Configured step '{step_type}' references unregistered adapter '{adapter_id}'"
            )

    def _build_step_event_payload(
        self,
        *,
        step_type: Optional[str],
        plan: TaskPlan,
        step,
        sample_identifier: str,
    ) -> Dict[str, Any]:
        return {
            "task_id": plan.metadata.get("task_id") or self._task_id,
            "sample_id": sample_identifier,
            "step_type": step_type,
            "adapter_id": _resolve_runtime_adapter_id(step, step_type, plan),
        }

    def _iter_samples(self, trace: ObservabilityTrace) -> Iterator[Tuple[int, dict]]:
        if self._shuffle:
            strategy = self._resolve_shuffle_strategy(trace)
            if strategy == "reservoir":
                yield from iter_reservoir_samples(
                    self._samples,
                    max_samples=self._max_samples or 0,
                    seed=self._shuffle_seed,
                )
                return
            if strategy == "external_index":
                for logical_idx, sample in iter_external_shuffle_samples(
                    self._samples,
                    seed=self._shuffle_seed,
                    artifact_root=self._shuffle_artifact_root,
                    keep_artifacts=bool(self._keep_shuffle_artifacts),
                ):
                    if self._max_samples is not None and logical_idx >= self._max_samples:
                        break
                    yield logical_idx, sample
                return
            samples = self._materialize_samples()
            indices = list(range(len(samples)))
            random.Random(self._shuffle_seed).shuffle(indices)
            if self._max_samples is not None:
                indices = indices[: self._max_samples]
            for logical_idx, sample_idx in enumerate(indices):
                yield logical_idx, samples[sample_idx]
            return

        self._record_shuffle_decision(
            trace,
            requested="disabled",
            resolved="sequential",
            reason="shuffle_disabled",
            size_hint=try_resolve_length(self._samples),
        )

        for idx, sample in enumerate(self._samples):
            if self._max_samples is not None and idx >= self._max_samples:
                break
            yield idx, sample

    def _materialize_samples(self) -> List[dict]:
        if self._streaming:
            raise RuntimeError("Cannot materialize streaming datasets")
        if self._materialized_samples is None:
            self._materialized_samples = list(self._samples)
            logger.debug(
                "Materialized {} samples into memory", len(self._materialized_samples)
            )
        return self._materialized_samples

    def _resolve_shuffle_strategy(self, trace: ObservabilityTrace) -> str:
        requested = self._shuffle_strategy
        size_hint = None if self._streaming else try_resolve_length(self._samples)
        reason = None
        resolved = requested

        if requested == "reservoir":
            if self._max_samples is None:
                raise RuntimeError(
                    "shuffle_strategy='reservoir' requires max_samples to be set"
                )
            reason = "explicit_reservoir"
        elif requested == "in_memory":
            if self._streaming:
                raise RuntimeError(
                    "shuffle_strategy='in_memory' is not supported for streaming datasets"
                )
            resolved = "in_memory"
            reason = "explicit_in_memory"
        elif requested == "external_index":
            if (
                not self._streaming
                and size_hint is not None
                and size_hint <= self._shuffle_small_dataset_threshold
            ):
                resolved = "in_memory"
                reason = "small_dataset_threshold"
            else:
                resolved = "external_index"
                reason = "explicit_external_index"
        elif requested == "auto":
            if self._max_samples is not None:
                resolved = "reservoir"
                reason = "max_samples_present"
            elif (
                not self._streaming
                and size_hint is not None
                and size_hint <= self._shuffle_small_dataset_threshold
            ):
                resolved = "in_memory"
                reason = "small_dataset_threshold"
            else:
                resolved = "external_index"
                reason = "streaming_or_large_dataset"
        else:
            raise RuntimeError(f"Unsupported shuffle strategy '{requested}'")

        self._record_shuffle_decision(
            trace,
            requested=requested,
            resolved=resolved,
            reason=reason,
            size_hint=size_hint,
        )
        return resolved

    def _record_shuffle_decision(
        self,
        trace: ObservabilityTrace,
        *,
        requested: str,
        resolved: str,
        reason: Optional[str],
        size_hint: Optional[int],
    ) -> None:
        summary: Dict[str, Any] = {
            "enabled": bool(self._shuffle),
            "requested": requested,
            "resolved": resolved,
            "seed": self._shuffle_seed if self._shuffle else None,
            "max_samples": self._max_samples,
            "small_dataset_threshold": (
                self._shuffle_small_dataset_threshold if self._shuffle else None
            ),
            "size_hint": size_hint,
        }
        if reason:
            summary["reason"] = reason
        if self._shuffle_artifact_root is not None and resolved == "external_index":
            summary["artifact_root"] = str(self._shuffle_artifact_root)
        self._shuffle_summary = summary
        trace.emit(
            "shuffle_strategy_selected",
            {
                **summary,
                "task_id": self._task_id,
            },
        )
        if requested != resolved:
            trace.emit(
                "shuffle_fallback",
                {
                    "task_id": self._task_id,
                    "requested": requested,
                    "resolved": resolved,
                    "reason": reason,
                    "size_hint": size_hint,
                },
            )

    def _process_sample(
        self,
        logical_idx: int,
        sample: dict,
        planner: TaskPlanner,
        role_manager: RoleManager,
        trace: ObservabilityTrace,
    ) -> None:
        for hook in self._hooks:
            hook(sample)
        plan = planner.prepare_plan(
            sample,
            metadata=self._compose_metadata(sample),
        )
        sample_id = plan.metadata.get("sample_id") or resolve_runtime_sample_id(
            sample,
            task_id=self._task_id,
            logical_idx=logical_idx,
        )
        logger.debug(
            "Processing sample logical_idx={} sample_id={}", logical_idx, sample_id
        )
        self._execute_plan(sample, plan, role_manager, trace, sample_id)
        with self._processed_lock:
            self._processed_count += 1

    def _preview_sample_id(self, logical_idx: int, sample: dict) -> str:
        return resolve_runtime_sample_id(
            sample,
            task_id=self._task_id,
            logical_idx=logical_idx,
        )

    def _get_execution_controller(
        self, planner: TaskPlanner
    ) -> TaskExecutionController:
        if self._execution_controller is not None:
            return self._execution_controller
        auto_eval_step = planner.get_auto_eval_step()
        metric_count = auto_eval_step.metric_count() if auto_eval_step is not None else 0
        metric_workers = _default_metric_workers(metric_count)
        if self._legacy_ff_mode:
            logger.warning(
                "GAGE_EVAL_FF_MODE is deprecated; mapping SampleLoop to failure_policy='best_effort'"
            )
        self._execution_controller = TaskExecutionController(
            sample_workers=1 if self._should_run_sequentially() else self._concurrency,
            metric_workers=metric_workers,
            failure_policy=self._failure_policy,
            legacy_ff_mode=self._legacy_ff_mode,
            report_partial_on_failure=self._report_partial_on_failure,
        )
        planner.attach_execution_controller(self._execution_controller)
        return self._execution_controller

    def _prefetch_samples(
        self,
        iterator: Iterator[Tuple[int, dict]],
        sample_queue: Queue,
        sentinel: object,
        stop_event: threading.Event,
        trace: ObservabilityTrace,
        errors: List[BaseException],
    ) -> None:
        wait_logged = False
        try:
            for item in iterator:
                if stop_event.is_set():
                    break
                while not stop_event.is_set():
                    try:
                        sample_queue.put(item, timeout=0.1)
                        if wait_logged:
                            wait_logged = False
                        break
                    except Full:
                        if not wait_logged:
                            trace.emit(
                                "sample_prefetch_wait",
                                {
                                    "buffer_capacity": self._buffer_capacity,
                                    "task_id": self._task_id,
                                },
                            )
                            logger.debug(
                                "SampleLoop prefetch blocked (buffer_capacity={})",
                                self._buffer_capacity,
                            )
                            wait_logged = True
                        continue
            logger.debug(
                "SamplePrefetcher thread completed (task_id={})", self._task_id
            )
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)
            logger.exception(
                "SamplePrefetcher thread failed (task_id={})", self._task_id
            )
        finally:
            while True:
                try:
                    sample_queue.put(sentinel, timeout=0.1)
                    break
                except Full:
                    if stop_event.is_set():
                        break
                    continue

    def _drain_futures(
        self,
        futures: Set,
        sample_queue: Queue,
        trace: ObservabilityTrace,
        *,
        drain_all: bool = False,
    ) -> None:
        if not futures:
            return
        return_when = ALL_COMPLETED if drain_all else FIRST_COMPLETED
        done, _ = wait(futures, return_when=return_when)
        for future in done:
            futures.discard(future)
            if future.cancelled():
                continue
            try:
                future.result()
            except Exception as exc:
                logger.exception(
                    "SampleLoop worker failed (task_id={}): {}", self._task_id, exc
                )
        self._emit_buffer_state(trace, sample_queue, len(futures))

    def _drain_pending_queue(self, sample_queue: Queue, sentinel: object) -> int:
        cancelled = 0
        while True:
            try:
                item = sample_queue.get_nowait()
            except Empty:
                break
            if item is sentinel:
                continue
            cancelled += 1
        return cancelled

    def _emit_buffer_state(
        self, trace: ObservabilityTrace, sample_queue: Queue, inflight: int
    ) -> None:
        trace.emit(
            "sample_buffer_state",
            {
                "buffer_size": sample_queue.qsize(),
                "buffer_capacity": self._buffer_capacity,
                "inflight": inflight,
                "max_inflight": self._max_inflight,
                "task_id": self._task_id,
            },
        )

    def _should_run_sequentially(self) -> bool:
        return self._sequential_override or self._concurrency <= 1

    @property
    def processed_count(self) -> int:
        return self._processed_count

    @property
    def concurrency(self) -> int:
        return self._concurrency

    @property
    def report_partial_on_failure(self) -> bool:
        controller = self._execution_controller
        if controller is not None:
            return controller.report_partial_on_failure
        return self._report_partial_on_failure

    @property
    def shuffle_summary(self) -> Dict[str, Any]:
        return dict(self._shuffle_summary)

    def _compose_metadata(self, sample: dict) -> dict:
        metadata = {
            "sample_id": resolve_runtime_sample_id(sample, task_id=self._task_id)
        }
        if self._task_id:
            metadata["task_id"] = self._task_id
        return metadata

    def _build_sandbox_provider(
        self,
        plan: TaskPlan,
        sample: dict,
        role_manager: RoleManager,
        trace: ObservabilityTrace,
        sample_identifier: str,
    ) -> Optional[SandboxProvider]:
        config, sandbox_adapter_id = self._resolve_sandbox_target(
            plan, sample, role_manager
        )
        if not config:
            return None
        arena_id = None
        if sandbox_adapter_id and sandbox_adapter_id == plan.arena_role:
            arena_id = f"{sandbox_adapter_id}_{sample_identifier}"
        scope = SandboxScope(
            run_id=trace.run_id,
            task_id=plan.metadata.get("task_id") or self._task_id,
            sample_id=sample_identifier,
            arena_id=arena_id,
        )
        return SandboxProvider(self._sandbox_manager, config, scope, trace=trace)

    def _resolve_sandbox_target(
        self,
        plan: TaskPlan,
        sample: dict,
        role_manager: RoleManager,
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        sample_config = (
            sample.get("sandbox") if isinstance(sample.get("sandbox"), dict) else None
        )
        adapter_config, adapter_id = self._resolve_adapter_sandbox_config(
            plan, role_manager
        )
        if not adapter_config and not sample_config:
            return None, None
        return self._sandbox_manager.resolve_config(
            adapter_config or {}, sample_config
        ), adapter_id

    def _resolve_adapter_sandbox_config(
        self, plan: TaskPlan, role_manager: RoleManager
    ) -> tuple[Dict[str, Any], Optional[str]]:
        adapter_ids: List[str] = []
        if plan.inference_role:
            adapter_ids.append(plan.inference_role)
        if plan.arena_role:
            adapter_ids.append(plan.arena_role)
        if plan.judge_role:
            adapter_ids.append(plan.judge_role)
        for step in plan.support_steps or []:
            if isinstance(step, dict):
                adapter_id = step.get("adapter_id") or step.get("role_ref")
                if adapter_id:
                    adapter_ids.append(str(adapter_id))
        for adapter_id in adapter_ids:
            adapter = role_manager.get_adapter(adapter_id)
            if adapter is None:
                continue
            sandbox_config = getattr(adapter, "sandbox_config", None)
            if isinstance(sandbox_config, dict) and sandbox_config:
                return dict(sandbox_config), adapter_id
        return {}, None


def _env_flag(var: str, *, default: bool) -> bool:
    value = os.environ.get(var)
    if value is None:
        return default
    return value.lower() in {"1", "true", "yes", "on"}


def _env_int(var: str) -> Optional[int]:
    value = os.environ.get(var)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        logger.warning("Invalid integer for {}={}; ignoring override", var, value)
        return None


def _normalize_shuffle_strategy(value: Optional[str]) -> str:
    normalized = str(value or _DEFAULT_SHUFFLE_STRATEGY).strip().lower()
    if normalized in {"auto", "in_memory", "reservoir", "external_index"}:
        return normalized
    return _DEFAULT_SHUFFLE_STRATEGY


def _default_metric_workers(metric_count: int) -> int:
    if metric_count <= 0:
        return 0
    env_workers = _env_int("GAGE_EVAL_AUTOEVAL_WORKERS")
    if env_workers is not None and env_workers > 0:
        return max(1, min(env_workers, metric_count))
    return max(1, min(metric_count, 2))


def _resolve_step_type(step) -> Optional[str]:
    if hasattr(step, "step_type"):
        return getattr(step, "step_type")
    if hasattr(step, "get"):
        return step.get("step") or step.get("step_type")
    return None


def _resolve_runtime_adapter_id(step, step_type: Optional[str], plan: TaskPlan) -> Optional[str]:
    if step_type == "support" and hasattr(step, "get"):
        return step.get("adapter_id") or step.get("role_ref")
    if step_type == "inference":
        return plan.inference_role
    if step_type == "arena":
        return plan.arena_role
    if step_type == "judge":
        return plan.judge_role
    return None
