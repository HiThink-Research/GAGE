"""Shared writer for imported sample records and Agent-Kit style artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.agent_runtime.trace_schema import ArtifactRef, SampleRecord
from gage_eval.evaluation.cache import EvalCache
from gage_eval.reporting.privacy import SecretFilter


@dataclass
class WrittenSampleArtifacts:
    sample_id: str
    sample_record_ref: ArtifactRef
    trial_aggregate_ref: ArtifactRef
    sample_cache_path: str
    artifact_refs: list[ArtifactRef]


class SampleArtifactWriter:
    """Write imported samples through the same cache/artifact schema as Path A."""

    def __init__(
        self,
        *,
        cache_store: EvalCache,
        artifact_sink: RuntimeArtifactSink | None = None,
        dut_id: str = "external_harness",
    ) -> None:
        self.cache_store = cache_store
        self.artifact_sink = artifact_sink or RuntimeArtifactSink(
            base_dir=str(cache_store.run_dir.parent)
        )
        self.dut_id = dut_id

    @classmethod
    def from_context(cls, context) -> "SampleArtifactWriter":
        return cls(cache_store=context.cache_store)

    def write(self, records: Iterable[Any], *, context, handle=None) -> dict[str, Any]:
        written = [
            self.write_sample_record(record, context=context, handle=handle)
            for record in records
        ]
        return {"written": len(written), "samples": written}

    def write_sample_record(self, record: Any, *, context=None, handle=None) -> WrittenSampleArtifacts:
        payload = _record_payload(record)
        run_id = str(payload.get("run_id") or self.cache_store.run_id)
        task_id = str(payload["task_id"])
        sample_id = str(payload["sample_id"])
        sample = _report_safe_value(dict(payload["sample"]))
        trial_results = _report_safe_value(list(payload["trial_results"]))
        aggregate = _report_safe_value(payload["aggregate"])
        environment_descriptor = _report_safe_value(dict(payload.get("environment_descriptor") or {}))
        if handle is not None and not environment_descriptor:
            environment_descriptor = dict(getattr(handle, "environment", {}) or {})
        artifacts: list[ArtifactRef] = list(payload.get("artifacts") or [])

        effective_config_ref = self.artifact_sink.write_effective_config(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            final_config=dict(payload.get("effective_config") or {"sample_id": sample_id, "task_id": task_id}),
            source_layers=list(payload.get("source_layers") or []),
        )
        trial_aggregate_ref = self.artifact_sink.write_trial_aggregate(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            aggregate=aggregate,
        )
        infra_refs = self._write_sample_infra_artifacts(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            specs=payload.get("infra_artifacts") or [],
        )
        artifacts.extend(
            ref
            for ref in (effective_config_ref, trial_aggregate_ref, *infra_refs)
            if ref is not None
        )
        sample_record_payload = {
            "run_id": run_id,
            "task_id": task_id,
            "sample_id": sample_id,
            "dut_id": str(payload.get("dut_id") or self.dut_id),
            "input_ref": dict(payload.get("input_ref") or {"sample_id": sample_id}),
            "trial_policy": dict(payload.get("trial_policy") or {"trials": len(trial_results)}),
            "trial_results": [
                result.model_dump(mode="python") if hasattr(result, "model_dump") else dict(result)
                for result in trial_results
            ],
            "aggregate_result": _aggregate_payload(aggregate),
            "scheduler_result": _report_safe_value(
                dict(payload.get("scheduler_result") or _primary_attr(trial_results, "scheduler_result"))
            ),
            "verifier_result": _report_safe_value(
                dict(payload.get("verifier_result") or _primary_attr(trial_results, "verifier_result"))
            ),
            "environment_descriptor": environment_descriptor,
            "effective_config_ref": effective_config_ref.model_dump(mode="python"),
            "artifacts": [ref.model_dump(mode="python") for ref in artifacts],
            "status": str(payload.get("status") or _status_from_trials(trial_results)),
            "failure": _report_safe_value(payload.get("failure")),
        }
        sample_record = SampleRecord.model_validate(sample_record_payload)
        sample_record_ref = self.artifact_sink.write_artifact(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            owner="infra",
            name="sample_record.json",
            content=sample_record.model_dump(mode="python"),
            mime_type="application/json",
        )
        if not isinstance(sample_record_ref, ArtifactRef):  # pragma: no cover
            raise TypeError("sample_record write must return ArtifactRef")
        artifacts.append(sample_record_ref)
        sample_cache_path = self.cache_store.write_sample(
            sample_id,
            {
                "sample": sample,
                "model_output": _report_safe_value(payload.get("model_output") or _primary_predict_result(sample)),
                "judge_output": _report_safe_value(payload.get("judge_output") or sample.get("eval_result") or {}),
                "trial_results": sample_record_payload["trial_results"],
                "aggregate_result": sample_record_payload["aggregate_result"],
                "artifact_refs": [ref.model_dump(mode="python") for ref in artifacts],
            },
            namespace=f"task/{task_id}",
        )
        return WrittenSampleArtifacts(
            sample_id=sample_id,
            sample_record_ref=sample_record_ref,
            trial_aggregate_ref=trial_aggregate_ref,
            sample_cache_path=str(sample_cache_path),
            artifact_refs=artifacts,
        )

    def _write_sample_infra_artifacts(
        self,
        *,
        run_id: str,
        task_id: str,
        sample_id: str,
        specs: Iterable[Any],
    ) -> list[ArtifactRef]:
        refs: list[ArtifactRef] = []
        for spec in specs:
            ref = self._write_sample_infra_artifact(
                run_id=run_id,
                task_id=task_id,
                sample_id=sample_id,
                spec=spec,
            )
            if ref is not None:
                refs.append(ref)
        return refs

    def _write_sample_infra_artifact(
        self,
        *,
        run_id: str,
        task_id: str,
        sample_id: str,
        spec: Any,
    ) -> ArtifactRef | None:
        if not isinstance(spec, Mapping):
            raise TypeError("sample infra artifact spec must be a mapping")
        name = str(spec.get("name") or "")
        if not name:
            raise ValueError("sample infra artifact spec requires name")
        content = spec.get("content", spec.get("payload"))
        if not content:
            return None
        ref = self.artifact_sink.write_artifact(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            owner=str(spec.get("owner") or "infra"),
            name=name,
            content=to_json_compatible(content),
            mime_type=str(spec.get("mime_type") or "application/json"),
            sample_level=True,
        )
        if not isinstance(ref, ArtifactRef):  # pragma: no cover
            raise TypeError(f"{name} write must return ArtifactRef")
        return ref


def _record_payload(record: Any) -> dict[str, Any]:
    if isinstance(record, Mapping):
        return dict(record)
    if hasattr(record, "to_writer_payload"):
        return dict(record.to_writer_payload())
    if hasattr(record, "__dict__"):
        return dict(vars(record))
    raise TypeError("sample artifact record must be a mapping or expose to_writer_payload()")


def _aggregate_payload(aggregate: Any) -> dict[str, Any]:
    if hasattr(aggregate, "to_dict"):
        return dict(aggregate.to_dict())
    if isinstance(aggregate, Mapping):
        return dict(aggregate)
    return dict(to_json_compatible(aggregate))


def _primary_attr(items: Sequence[Any], attr: str) -> dict[str, Any]:
    if not items:
        return {}
    value = getattr(items[0], attr, None)
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(items[0], Mapping):
        value = items[0].get(attr)
        return dict(value) if isinstance(value, Mapping) else {}
    return {}


def _primary_predict_result(sample: Mapping[str, Any]) -> dict[str, Any]:
    predict_result = sample.get("predict_result")
    if isinstance(predict_result, list) and predict_result:
        first = predict_result[0]
        return dict(first) if isinstance(first, Mapping) else {"value": first}
    return {}


def _status_from_trials(trial_results: Sequence[Any]) -> str:
    if not trial_results:
        return "failed"
    statuses = [
        getattr(result, "status", result.get("status") if isinstance(result, Mapping) else None)
        for result in trial_results
    ]
    if any(status == "completed" for status in statuses):
        return "completed"
    if any(status == "aborted" for status in statuses):
        return "aborted"
    return "failed"


_REPORT_SECRET_FILTER = SecretFilter()


def _report_safe_value(value: Any) -> Any:
    return _REPORT_SECRET_FILTER.redact(value).value


__all__ = ["SampleArtifactWriter", "WrittenSampleArtifacts"]
