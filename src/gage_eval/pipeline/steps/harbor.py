"""Task-level Harbor external harness steps."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
import inspect
from pathlib import Path
from typing import Any, Optional

from gage_eval.external_harness_kits.archive import write_raw_archive_entry
from gage_eval.external_harness_kits.secret_redaction import (
    SecretRedactionContext,
    redact_for_artifact,
)
from gage_eval.pipeline.sample_artifact_writer import SampleArtifactWriter
from gage_eval.pipeline.steps.base import TaskStep
from gage_eval.registry import registry
from gage_eval.external_harness_kits.base import (
    TaskBatchHarnessHandle,
    TaskBatchHarnessRequest,
    TaskBatchHarnessResult,
)


@dataclass(frozen=True)
class HarborJobHandle:
    job_name: str
    jobs_dir: Path
    job_dir: Path
    job_config_path: Path
    launcher_result_path: Path
    workdir: Path
    environment: dict[str, Any]
    invocation_metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        context = SecretRedactionContext()
        return {
            "job_name": self.job_name,
            "jobs_dir": str(self.jobs_dir),
            "job_dir": str(self.job_dir),
            "job_config_path": str(self.job_config_path),
            "launcher_result_path": str(self.launcher_result_path),
            "workdir": str(self.workdir),
            "environment": redact_for_artifact(self.environment, context=context),
            "invocation_metadata": redact_for_artifact(
                self.invocation_metadata,
                context=context,
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "HarborJobHandle":
        missing = [
            key
            for key in _HANDLE_KEYS
            if key not in payload
        ]
        if missing:
            raise ValueError(f"HarborJobHandle missing required fields: {', '.join(missing)}")
        return cls(
            job_name=str(payload["job_name"]),
            jobs_dir=Path(str(payload["jobs_dir"])),
            job_dir=Path(str(payload["job_dir"])),
            job_config_path=Path(str(payload["job_config_path"])),
            launcher_result_path=Path(str(payload["launcher_result_path"])),
            workdir=Path(str(payload["workdir"])),
            environment=dict(_mapping(payload.get("environment"))),
            invocation_metadata=dict(_mapping(payload.get("invocation_metadata"))),
        )

    @classmethod
    def from_sources(
        cls,
        *,
        plan: Any,
        launch_handle: Any,
        result: Any,
    ) -> "HarborJobHandle":
        for candidate in (
            _payload_mapping(result).get("harbor_job_handle"),
            _payload_mapping(result).get("handle"),
            _payload_mapping(launch_handle).get("harbor_job_handle"),
            _payload_mapping(launch_handle).get("handle"),
            result,
            launch_handle,
        ):
            handle = _coerce_handle(candidate)
            if handle is not None:
                return handle
        invocation = _payload_mapping(plan).get("invocation")
        if invocation is None:
            raise ValueError("HarborRunStep could not derive HarborJobHandle from adapter output")
        return cls.from_invocation(invocation, result=result)

    @classmethod
    def from_invocation(cls, invocation: Any, *, result: Any = None) -> "HarborJobHandle":
        payload = _object_mapping(invocation)
        job_name = str(payload.get("job_name") or "")
        if not job_name:
            raise ValueError("Harbor invocation missing job_name")
        jobs_dir = Path(str(payload.get("jobs_dir") or "."))
        workdir = Path(str(payload.get("workdir") or jobs_dir.parent))
        job_config_path = Path(str(payload.get("job_config_path") or workdir / "harbor_job.json"))
        launcher_result_path = _launcher_result_path(payload, workdir=workdir)
        result_payload = _payload_mapping(result)
        job_dir = Path(str(result_payload.get("job_dir") or jobs_dir / job_name))
        job_config = _mapping(payload.get("job_config"))
        environment = dict(_mapping(job_config.get("environment")))
        if hasattr(invocation, "to_artifact_dict"):
            invocation_metadata = dict(invocation.to_artifact_dict())
        else:
            invocation_metadata = {
                key: value
                for key, value in payload.items()
                if key
                in {
                    "job_name",
                    "jobs_dir",
                    "job_config_path",
                    "launcher_mode",
                    "launcher_argv",
                    "workdir",
                    "expected_total_trials",
                }
            }
        return cls(
            job_name=job_name,
            jobs_dir=jobs_dir,
            job_dir=job_dir,
            job_config_path=job_config_path,
            launcher_result_path=launcher_result_path,
            workdir=workdir,
            environment=environment,
            invocation_metadata=invocation_metadata,
        )


@registry.asset(
    "pipeline_steps",
    "harbor_run",
    desc="Task-level step that launches a Harbor external harness job",
    tags=("external_harness", "harbor"),
    step_kind="task",
    requires_adapter=True,
    allow_multiple=False,
)
class HarborRunStep(TaskStep):
    def __init__(
        self,
        adapter_id: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__("HarborRunStep")
        self.adapter_id = adapter_id
        self.params = dict(params or {})

    def execute_task(self, context, *, step=None, step_index: int = 0):
        del step_index
        adapter_id = self._resolve_adapter_id(step)
        adapter = context.get_task_batch_harness_adapter(adapter_id)
        request = TaskBatchHarnessRequest(
            adapter_id=adapter_id,
            payload=context.request_payload(adapter_id=adapter_id),
        )
        plan = adapter.translate(request)
        _initialize_adapter(adapter, plan)
        launch_handle = adapter.launch(plan)
        result = adapter.poll_until_done(launch_handle)
        handle = HarborJobHandle.from_sources(
            plan=plan,
            launch_handle=launch_handle,
            result=result,
        )
        _write_raw_archive_manifest(context, adapter_id=adapter_id, handle=handle)
        context.store("harbor_job_handle", handle)
        context.store("harbor_task_batch_plan", plan)
        context.store("harbor_task_batch_launch_handle", launch_handle)
        context.store("harbor_task_batch_result", result)
        return {
            "job_name": handle.job_name,
            "harbor_job_handle": handle.to_dict(),
        }

    def _resolve_adapter_id(self, step: Any = None) -> str:
        adapter_id = self.adapter_id or getattr(step, "adapter_id", None)
        if not adapter_id:
            raise ValueError("harbor_run requires adapter_id")
        return str(adapter_id)


@registry.asset(
    "pipeline_steps",
    "harbor_result",
    desc="Task-level step that imports Harbor external harness results",
    tags=("external_harness", "harbor"),
    step_kind="task",
    requires_adapter=True,
    allow_multiple=False,
)
class HarborResultStep(TaskStep):
    def __init__(
        self,
        adapter_id: Optional[str] = None,
        params: Optional[Mapping[str, Any]] = None,
        artifact_writer_factory: Optional[Callable[[Any], Any]] = None,
        **kwargs,
    ) -> None:
        del kwargs
        super().__init__("HarborResultStep")
        self.adapter_id = adapter_id
        self.params = dict(params or {})
        self._artifact_writer_factory = artifact_writer_factory or SampleArtifactWriter.from_context

    def execute_task(self, context, *, step=None, step_index: int = 0):
        del step_index
        adapter_id = self._resolve_adapter_id(step)
        adapter = context.get_task_batch_harness_adapter(adapter_id)
        handle = self._resolve_handle(context)
        result = context.load("harbor_task_batch_result")
        if result is None:
            result = TaskBatchHarnessResult(
                adapter_id=adapter_id,
                payload={"handle": handle.to_dict()},
            )
        records = list(_parse_results(adapter, result, context=context, handle=handle))
        writer = self._artifact_writer_factory(context)
        _write_records(writer, records, context=context, handle=handle)
        context.store("harbor_result_records", records)
        return {
            "job_name": handle.job_name,
            "produced_sample_count": len(records),
            "sample_count": len(records),
        }

    def _resolve_adapter_id(self, step: Any = None) -> str:
        adapter_id = self.adapter_id or getattr(step, "adapter_id", None)
        if not adapter_id:
            raise ValueError("harbor_result requires adapter_id")
        return str(adapter_id)

    def _resolve_handle(self, context) -> HarborJobHandle:
        handle = context.load("harbor_job_handle")
        coerced = _coerce_handle(handle)
        if coerced is not None:
            return coerced
        if "handle" in self.params:
            coerced = _coerce_handle(self.params["handle"])
            if coerced is not None:
                return coerced
        if "jobs_dir" in self.params and "job_name" in self.params:
            jobs_dir = Path(str(self.params["jobs_dir"]))
            job_name = str(self.params["job_name"])
            workdir = Path(str(self.params.get("workdir") or jobs_dir.parent))
            return HarborJobHandle(
                job_name=job_name,
                jobs_dir=jobs_dir,
                job_dir=Path(str(self.params.get("job_dir") or jobs_dir / job_name)),
                job_config_path=Path(str(self.params.get("job_config_path") or workdir / "harbor_job.json")),
                launcher_result_path=Path(str(self.params.get("launcher_result_path") or workdir / "launcher_result.json")),
                workdir=workdir,
                environment=dict(_mapping(self.params.get("environment"))),
                invocation_metadata=dict(_mapping(self.params.get("invocation_metadata"))),
            )
        raise ValueError("harbor_result requires a HarborJobHandle from harbor_run or params")


class _NoopArtifactWriter:
    def __init__(self, context) -> None:
        self.context = context

    def write(self, records: Iterable[Any], *, context, handle: HarborJobHandle) -> dict[str, Any]:
        del context, handle
        return {"written": len(list(records))}


_HANDLE_KEYS = (
    "job_name",
    "jobs_dir",
    "job_dir",
    "job_config_path",
    "launcher_result_path",
    "workdir",
    "environment",
    "invocation_metadata",
)


def _coerce_handle(value: Any) -> HarborJobHandle | None:
    if isinstance(value, HarborJobHandle):
        return value
    if isinstance(value, Mapping) and all(key in value for key in _HANDLE_KEYS):
        return HarborJobHandle.from_dict(value)
    return None


def _payload_mapping(value: Any) -> Mapping[str, Any]:
    payload = getattr(value, "payload", None)
    if isinstance(payload, Mapping):
        return payload
    if isinstance(value, Mapping):
        return value
    return {}


def _object_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dataclass_fields__"):
        return {
            key: getattr(value, key)
            for key in value.__dataclass_fields__
        }
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _launcher_result_path(payload: Mapping[str, Any], *, workdir: Path) -> Path:
    argv = payload.get("launcher_argv")
    if isinstance(argv, list) and "--result-file" in argv:
        idx = argv.index("--result-file")
        if idx + 1 < len(argv):
            return Path(str(argv[idx + 1]))
    return workdir / "launcher_result.json"


def _write_records(writer: Any, records: list[Any], *, context, handle: HarborJobHandle) -> Any:
    write = getattr(writer, "write", None)
    if callable(write):
        return write(records, context=context, handle=handle)
    write_many = getattr(writer, "write_many", None)
    if callable(write_many):
        return write_many(records, context=context, handle=handle)
    raise TypeError("HarborResultStep artifact writer must define write() or write_many()")


def _write_raw_archive_manifest(context: Any, *, adapter_id: str, handle: HarborJobHandle) -> None:
    cache_store = getattr(context, "cache_store", None)
    run_dir = getattr(cache_store, "run_dir", None)
    if run_dir is None:
        return
    archive_root = Path(run_dir) / "external_harness"
    write_raw_archive_entry(
        archive_root=archive_root,
        run_id=str(getattr(cache_store, "run_id", getattr(getattr(context, "trace", None), "run_id", ""))),
        task_id=str(getattr(context, "task_id", "")),
        adapter_id=adapter_id,
        provider="harbor",
        job_name=handle.job_name,
        artifacts={
            "workdir_ref": handle.workdir,
            "invocation_ref": handle.workdir / "invocation.json",
            "job_config_ref": handle.workdir / "job_config.json",
            "launcher_input_ref": handle.job_config_path,
            "launcher_result_ref": handle.launcher_result_path,
            "jobs_dir_ref": handle.jobs_dir,
            "job_dir_ref": handle.job_dir,
        },
        metadata={
            "expected_total_trials": handle.invocation_metadata.get("expected_total_trials"),
        },
    )


def _parse_results(adapter: Any, result: TaskBatchHarnessResult, *, context, handle: HarborJobHandle) -> Iterable[Any]:
    parse_results = getattr(adapter, "parse_results")
    try:
        signature = inspect.signature(parse_results)
    except (TypeError, ValueError):
        return parse_results(result)
    kwargs: dict[str, Any] = {}
    if "context" in signature.parameters:
        kwargs["context"] = context
    if "handle" in signature.parameters:
        kwargs["handle"] = handle
    return parse_results(result, **kwargs)


def _initialize_adapter(adapter: Any, plan: Any) -> None:
    initializer = getattr(adapter, "_initialize", None)
    if callable(initializer):
        if _callable_accepts_positional(initializer):
            initializer(plan)
        else:
            initializer()
        return
    initializer = getattr(adapter, "initialize", None)
    if callable(initializer):
        initializer()


def _callable_accepts_positional(func: Callable[..., Any]) -> bool:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return False
    return any(
        parameter.kind
        in {
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.VAR_POSITIONAL,
        }
        for parameter in signature.parameters.values()
    )


__all__ = [
    "HarborJobHandle",
    "HarborResultStep",
    "HarborRunStep",
]
