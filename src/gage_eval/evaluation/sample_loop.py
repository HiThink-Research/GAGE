"""Sample loop implementation modelled after llm-eval's data server."""

from __future__ import annotations

import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED, ALL_COMPLETED
from queue import Queue, Full
from typing import Callable, Iterable, Iterator, List, Optional, Sequence, Tuple, Set

from loguru import logger

from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.evaluation.task_planner import TaskPlanner, TaskPlan
from gage_eval.role.role_manager import RoleManager


class SampleLoop:
    """Iterate over samples and invoke TaskPlanner/RoleManager per sample."""

    def __init__(
        self,
        samples: Iterable[dict],
        *,
        shuffle: Optional[bool] = None,
        shuffle_seed: Optional[int] = None,
        max_samples: Optional[int] = None,
        concurrency: Optional[int] = None,
        streaming: bool = False,
        task_id: Optional[str] = None,
        prefetch_factor: Optional[int] = None,
        max_inflight: Optional[int] = None,
    ) -> None:
        self._samples = samples
        self._hooks: List[Callable[[dict], None]] = []
        self._custom_steps: Sequence[dict] = ()
        self._shuffle = shuffle if shuffle is not None else _env_flag("GAGE_EVAL_SHUFFLE", default=False)
        self._shuffle_seed = shuffle_seed or int(os.environ.get("GAGE_EVAL_SHUFFLE_SEED", "123"))
        env_max_samples = os.environ.get("GAGE_EVAL_MAX_SAMPLES")
        self._max_samples = max_samples if max_samples is not None else (int(env_max_samples) if env_max_samples else None)
        env_threads = os.environ.get("GAGE_EVAL_THREADS")
        default_threads = min(os.cpu_count() or 1, 4)
        self._concurrency = max(
            1,
            concurrency if concurrency is not None else (int(env_threads) if env_threads else default_threads),
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
        self._buffer_capacity = max(self._max_inflight, self._concurrency * self._prefetch_factor)
        self._sequential_override = _env_flag("GAGE_EVAL_SEQUENTIAL", default=False)
        self._materialized_samples: Optional[List[dict]] = None
        self._streaming = streaming
        if self._streaming and self._shuffle:
            logger.warning("Streaming datasets do not support shuffling; disabling shuffle")
            self._shuffle = False
        self._task_id = task_id
        self._processed_count = 0
        self._processed_lock = threading.Lock()
        logger.info(
            "SampleLoop initialized (shuffle={}, max_samples={}, concurrency={}, streaming={}, task_id={})",
            self._shuffle,
            self._max_samples,
            self._concurrency,
            self._streaming,
            self._task_id,
        )

    def register_hook(self, hook: Callable[[dict], None]) -> None:
        self._hooks.append(hook)

    def configure_custom_steps(self, steps: Sequence[dict]) -> None:
        self._custom_steps = steps

    def run(self, planner: TaskPlanner, role_manager: RoleManager, trace: ObservabilityTrace) -> None:
        work = self._iter_samples()
        if self._should_run_sequentially():
            logger.info("SampleLoop running sequentially over samples")
            for idx, sample in work:
                self._process_sample(idx, sample, planner, role_manager, trace)
            return

        ff_mode = _env_flag("GAGE_EVAL_FF_MODE", default=False)

        logger.info(
            "SampleLoop running with bounded buffer (workers=%s, max_inflight=%s, prefetch_factor=%s, buffer_capacity=%s)",
            self._concurrency,
            self._max_inflight,
            self._prefetch_factor,
            self._buffer_capacity,
        )
        sentinel = object()
        sample_queue: Queue = Queue(maxsize=self._buffer_capacity)
        stop_event = threading.Event()
        producer_errors: List[BaseException] = []
        producer = threading.Thread(
            target=self._prefetch_samples,
            name="SamplePrefetcher",
            args=(work, sample_queue, sentinel, stop_event, trace, producer_errors),
            daemon=True,
        )
        producer.start()

        if ff_mode:
            self._run_fire_and_forget(sample_queue, sentinel, stop_event, planner, role_manager, trace, producer_errors)
        else:
            futures: Set = set()
            try:
                with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
                    while True:
                        item = sample_queue.get()
                        if item is sentinel:
                            break
                        logical_idx, sample = item
                        future = executor.submit(
                            self._process_sample,
                            logical_idx,
                            sample,
                            planner,
                            role_manager,
                            trace,
                        )
                        futures.add(future)
                        self._emit_buffer_state(trace, sample_queue, len(futures))
                        if len(futures) >= self._max_inflight:
                            self._drain_futures(futures, sample_queue, trace)
                    self._drain_futures(futures, sample_queue, trace, drain_all=True)
            except Exception:
                stop_event.set()
                raise
            finally:
                stop_event.set()
                try:
                    sample_queue.put_nowait(sentinel)
                except Exception:
                    pass
                producer.join()
            if producer_errors:
                raise producer_errors[0]

    def _execute_plan(
        self,
        sample: dict,
        plan: TaskPlan,
        role_manager: RoleManager,
        trace: ObservabilityTrace,
        sample_identifier: str,
    ) -> None:
        per_sample_ctx = plan.create_context(sample, trace, role_manager)
        with trace.use_sample(sample_identifier), role_manager.per_sample_session(per_sample_ctx) as session:
            if plan.steps:
                for step in plan.steps:
                    step_type = _resolve_step_type(step)
                    if step_type == "support":
                        if plan.support_steps:
                            session.execute_support_step(step)
                        continue
                    if step_type == "inference":
                        if plan.inference_role:
                            session.execute_inference()
                        continue
                    if step_type == "judge":
                        if plan.judge_role:
                            session.execute_judge()
                        continue
                    if step_type == "auto_eval":
                        if plan.auto_eval_enabled:
                            session.execute_auto_eval(sample_identifier)
                        continue
                    logger.debug("Unknown step type '{}' skipped during execution", step_type)
            else:
                if plan.support_steps:
                    session.execute_support()
                if plan.inference_role:
                    session.execute_inference()
                if plan.judge_role:
                    session.execute_judge()
                if plan.auto_eval_enabled:
                    session.execute_auto_eval(sample_identifier)

    def _iter_samples(self) -> Iterator[Tuple[int, dict]]:
        if self._shuffle:
            samples = self._materialize_samples()
            indices = list(range(len(samples)))
            random.Random(self._shuffle_seed).shuffle(indices)
            if self._max_samples is not None:
                indices = indices[: self._max_samples]
            for logical_idx, sample_idx in enumerate(indices):
                yield logical_idx, samples[sample_idx]
            return

        for idx, sample in enumerate(self._samples):
            if self._max_samples is not None and idx >= self._max_samples:
                break
            yield idx, sample

    def _materialize_samples(self) -> List[dict]:
        if self._streaming:
            raise RuntimeError("Cannot materialize streaming datasets")
        if self._materialized_samples is None:
            self._materialized_samples = list(self._samples)
            logger.debug("Materialized {} samples into memory", len(self._materialized_samples))
        return self._materialized_samples

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
        sample_id = plan.metadata.get("sample_id") or str(sample.get("id") or logical_idx)
        logger.debug("Processing sample logical_idx={} sample_id={}", logical_idx, sample_id)
        self._execute_plan(sample, plan, role_manager, trace, sample_id)
        with self._processed_lock:
            self._processed_count += 1

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
            logger.debug("SamplePrefetcher thread completed (task_id={})", self._task_id)
        except BaseException as exc:  # pragma: no cover - defensive
            errors.append(exc)
            logger.exception("SamplePrefetcher thread failed (task_id={})", self._task_id)
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
            try:
                future.result()
            except Exception as exc:
                # 确保 worker 异常不会被静默吞掉，便于上层感知并触发 stop_event 逻辑。
                logger.exception("SampleLoop worker failed (task_id=%s): %s", self._task_id, exc)
                raise
        self._emit_buffer_state(trace, sample_queue, len(futures))

    def _run_fire_and_forget(
        self,
        sample_queue: Queue,
        sentinel: object,
        stop_event: threading.Event,
        planner: TaskPlanner,
        role_manager: RoleManager,
        trace: ObservabilityTrace,
        producer_errors: List[BaseException],
    ) -> None:
        """Fire-and-forget 模式：使用信号量控制 max_inflight，不再维护 futures 集合。"""

        logger.info(
            "SampleLoop running in fire-and-forget mode (workers=%s, max_inflight=%s, buffer_capacity=%s)",
            self._concurrency,
            self._max_inflight,
            self._buffer_capacity,
        )
        sem = threading.Semaphore(self._max_inflight)
        worker_errors: List[BaseException] = []
        try:
            with ThreadPoolExecutor(max_workers=self._concurrency) as executor:
                while True:
                    item = sample_queue.get()
                    if item is sentinel:
                        break
                    logical_idx, sample = item
                    sem.acquire()

                    def _run_one(idx=logical_idx, s=sample):
                        try:
                            self._process_sample(idx, s, planner, role_manager, trace)
                        except BaseException as exc:
                            worker_errors.append(exc)
                            stop_event.set()
                            logger.exception("SampleLoop worker failed in FF mode (task_id=%s): %s", self._task_id, exc)
                        finally:
                            sem.release()

                    executor.submit(_run_one)

                # 等待所有 in-flight 任务完成：当可以连续获取 max_inflight 次时，说明没有任务持有信号量
                for _ in range(self._max_inflight):
                    sem.acquire()
        finally:
            stop_event.set()
            try:
                sample_queue.put_nowait(sentinel)
            except Exception:
                pass
            producer.join()
        if producer_errors:
            raise producer_errors[0]
        if worker_errors:
            # 抛出第一个 worker 错误，细节已在日志中记录
            raise worker_errors[0]

    def _emit_buffer_state(self, trace: ObservabilityTrace, sample_queue: Queue, inflight: int) -> None:
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

    def _compose_metadata(self, sample: dict) -> dict:
        metadata = {"sample_id": str(sample.get("id", "unknown"))}
        if self._task_id:
            metadata["task_id"] = self._task_id
        return metadata


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
        logger.warning("Invalid integer for %s=%s; ignoring override", var, value)
        return None


def _resolve_step_type(step) -> Optional[str]:
    if hasattr(step, "step_type"):
        return getattr(step, "step_type")
    if hasattr(step, "get"):
        return step.get("step") or step.get("step_type")
    return None
