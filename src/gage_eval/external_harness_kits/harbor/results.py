"""Import Harbor job output into GAGE Agent-Kit-V2 records."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
import time
from typing import Any, Iterable, Literal, Mapping

from gage_eval.agent_runtime import trials as trial_aggregation
from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.agent_runtime.trace_schema import ArtifactRef, TrialResult
from gage_eval.assets.datasets.sample import (
    Message,
    MessageContent,
    PredictResult,
    SCHEMA_VERSION,
    Sample,
    sample_to_dict,
)
from gage_eval.external_harness_kits.errors import ExternalHarnessParseError
from gage_eval.external_harness_kits.secret_redaction import redact_for_artifact
from gage_eval.external_harness_kits.trace_translation import AgentTraceTranslator


HARBOR_LAUNCHER_FAILED = "harbor.launcher_failed"
HARBOR_JOB_RESULT_MISSING = "harbor.job_result_missing"
HARBOR_TRIAL_EXCEPTION = "harbor.trial_exception"
HARBOR_VERIFIER_RESULT_MISSING = "harbor.verifier_result_missing"
EXTERNAL_HARNESS_CANCELLED = "external_harness.cancelled.subprocess_aborted"

MISSING_RESULT_WARNING = "external_harness.parse.missing_result"
MALFORMED_RESULT_WARNING = "external_harness.parse.malformed_output"
TRIAL_COUNT_MISMATCH_WARNING = "external_harness.parse.trial_count_mismatch"
PARTIAL_JOB_WARNING = "external_harness.parse.job_result_missing_partial"
CANCELLED_JOB_WARNING = "external_harness.cancelled.job_marked_cancelled"


@dataclass(frozen=True)
class HarborParseWarning:
    code: str
    message: str
    path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": self.message, "path": self.path}


@dataclass
class HarborImportedSample:
    sample_id: str
    task_id: str
    dataset_id: str
    sample: dict[str, Any]
    trial_results: list[TrialResult]
    aggregate: Any
    artifacts: list[ArtifactRef] = field(default_factory=list)
    harbor_invocation: dict[str, Any] = field(default_factory=dict)
    harbor_job_result: dict[str, Any] = field(default_factory=dict)
    environment_descriptor: dict[str, Any] = field(default_factory=dict)
    warnings: list[HarborParseWarning] = field(default_factory=list)
    trial_policy: dict[str, Any] = field(default_factory=dict)
    input_ref: dict[str, Any] = field(default_factory=dict)
    dut_id: str = "external_harness"

    def to_writer_payload(self) -> dict[str, Any]:
        primary = self.trial_results[0] if self.trial_results else None
        return {
            "task_id": self.task_id,
            "dataset_id": self.dataset_id,
            "sample_id": self.sample_id,
            "sample": self.sample,
            "trial_results": self.trial_results,
            "aggregate": self.aggregate,
            "artifacts": self.artifacts,
            "infra_artifacts": [
                {
                    "name": "harbor_invocation.json",
                    "content": self.harbor_invocation,
                },
                {
                    "name": "harbor_job_result.json",
                    "content": self.harbor_job_result,
                },
            ],
            "effective_config": {
                "external_harness": "harbor",
                "sample_id": self.sample_id,
                "task_id": self.task_id,
            },
            "source_layers": [{"source": "external_harness_kits.harbor"}],
            "environment_descriptor": self.environment_descriptor,
            "trial_policy": self.trial_policy or {"trials": len(self.trial_results)},
            "input_ref": self.input_ref or {"sample_id": self.sample_id, "dataset_id": self.dataset_id},
            "dut_id": self.dut_id,
            "scheduler_result": primary.scheduler_result if primary is not None else {},
            "verifier_result": primary.verifier_result if primary is not None else {},
            "status": _sample_status(self.trial_results),
            "failure": _sample_failure(self.trial_results),
        }


@dataclass
class HarborResultBundle:
    samples: list[HarborImportedSample]
    task_metrics: dict[str, Any]
    warnings: list[HarborParseWarning]
    elapsed_s: float

    def __iter__(self):
        return iter(self.samples)


@dataclass(frozen=True)
class _ParserContext:
    run_id: str
    task_id: str
    dataset_id: str
    artifact_sink: RuntimeArtifactSink
    reward_key: str
    aggregation: str | None
    expected_trials: int | None
    dut_id: str


@dataclass(frozen=True)
class _HarborHandle:
    job_name: str
    jobs_dir: Path
    job_dir: Path
    job_config_path: Path | None
    launcher_result_path: Path | None
    workdir: Path
    environment: dict[str, Any]
    invocation_metadata: dict[str, Any]


def parse_harbor_results(
    result: Any,
    *,
    context: Any = None,
    handle: Any = None,
    reward_key: str | None = None,
    aggregation: str | None = None,
    expected_trials: int | None = None,
    dut_id: str = "external_harness",
    trace_translator: AgentTraceTranslator | None = None,
) -> HarborResultBundle:
    """Parse a Harbor job tree and return sample records ready for the shared writer."""

    started = time.perf_counter()
    payload = _payload_mapping(result)
    resolved_handle = _resolve_handle(handle=handle, result_payload=payload)
    parser_context = _parser_context(
        context=context,
        handle=resolved_handle,
        reward_key=reward_key or str(payload.get("reward_key") or "reward"),
        aggregation=aggregation or _payload_aggregation(payload),
        expected_trials=expected_trials or _payload_expected_trials(payload, resolved_handle),
        dut_id=dut_id,
    )
    warnings: list[HarborParseWarning] = []

    cancelled_marker, cancelled_marker_path = _read_cancelled_marker(resolved_handle)
    if cancelled_marker:
        warnings.append(
            HarborParseWarning(
                CANCELLED_JOB_WARNING,
                "Harbor job was marked cancelled; importing available aborted evidence",
                str(cancelled_marker_path) if cancelled_marker_path else None,
            )
        )

    launcher_result = _read_launcher_result(resolved_handle, payload)
    if _launcher_failed(launcher_result) and not cancelled_marker:
        raise ExternalHarnessParseError(
            HARBOR_LAUNCHER_FAILED,
            f"Harbor launcher failed for job {resolved_handle.job_name}",
        )

    job_result, job_result_missing = _read_job_result(resolved_handle)
    raw_trials = _scan_trial_results(resolved_handle.job_dir, warnings)
    if cancelled_marker and not raw_trials:
        raw_trials = [_synthetic_cancelled_trial(resolved_handle, cancelled_marker)]
    if job_result_missing:
        if not raw_trials:
            raise ExternalHarnessParseError(
                HARBOR_JOB_RESULT_MISSING,
                f"Harbor job result is missing: {resolved_handle.job_dir / 'result.json'}",
            )
        warnings.append(
            HarborParseWarning(
                PARTIAL_JOB_WARNING,
                (
                    "Harbor job result is missing; continuing with cancelled marker"
                    if cancelled_marker
                    else "Harbor job result is missing; continuing with trial result files"
                ),
                str(resolved_handle.job_dir / "result.json"),
            )
        )

    grouped = _group_trials(raw_trials, parser_context, warnings)
    if not grouped:
        raise ExternalHarnessParseError(
            MALFORMED_RESULT_WARNING,
            "No Harbor trial result could be parsed",
        )

    samples: list[HarborImportedSample] = []
    for task_key, parsed_trials in grouped.items():
        if not parsed_trials:
            continue
        if parser_context.expected_trials is not None and len(parsed_trials) != parser_context.expected_trials:
            warnings.append(
                HarborParseWarning(
                    TRIAL_COUNT_MISMATCH_WARNING,
                    f"Parsed {len(parsed_trials)} trials for {task_key}; expected {parser_context.expected_trials}",
                    str(resolved_handle.job_dir),
                )
            )
        trial_results = [
            _build_trial_result(
                raw=raw,
                trial_id=f"trial_{index:04d}",
                parser_context=parser_context,
                handle=resolved_handle,
            )
            for index, raw in enumerate(parsed_trials, start=1)
        ]
        if not trial_results:
            continue
        aggregate = trial_aggregation.aggregate_trial_results(
            trial_results,
            aggregation=parser_context.aggregation or ("single" if len(trial_results) == 1 else "all"),
        )
        sample_id = _safe_segment(task_key)
        sample = _build_sample(
            sample_id=sample_id,
            raw_trials=parsed_trials,
            trial_results=trial_results,
            aggregate=aggregate,
            job_result=job_result,
            handle=resolved_handle,
            warnings=warnings,
            parser_context=parser_context,
            trace_translator=trace_translator,
        )
        primary_raw = parsed_trials[0].payload
        samples.append(
            HarborImportedSample(
                sample_id=sample_id,
                task_id=parser_context.task_id,
                dataset_id=parser_context.dataset_id,
                sample=sample,
                trial_results=trial_results,
                aggregate=aggregate,
                artifacts=_unique_artifact_refs(trial_results),
                harbor_invocation=redact_for_artifact(resolved_handle.invocation_metadata),
                harbor_job_result=redact_for_artifact(job_result),
                environment_descriptor=_environment_descriptor(primary_raw, resolved_handle),
                warnings=list(warnings),
                trial_policy={"trials": len(trial_results), "aggregation": aggregate.aggregation},
                input_ref=_input_ref(primary_raw, sample_id),
                dut_id=parser_context.dut_id,
            )
        )

    if not samples:
        raise ExternalHarnessParseError(
            MALFORMED_RESULT_WARNING,
            "No GAGE sample could be built from Harbor trial results",
        )

    return HarborResultBundle(
        samples=samples,
        task_metrics=_task_metrics(samples),
        warnings=warnings,
        elapsed_s=time.perf_counter() - started,
    )


@dataclass(frozen=True)
class _RawTrial:
    path: Path
    trial_dir: Path
    task_key: str
    payload: dict[str, Any]
    malformed: bool = False
    malformed_reason: str | None = None


def _scan_trial_results(job_dir: Path, warnings: list[HarborParseWarning]) -> list[_RawTrial]:
    trials: list[_RawTrial] = []
    if not job_dir.exists():
        return trials
    for trial_dir in sorted(path for path in job_dir.iterdir() if path.is_dir()):
        result_path = trial_dir / "result.json"
        if not result_path.exists():
            warnings.append(
                HarborParseWarning(
                    MISSING_RESULT_WARNING,
                    "Harbor trial directory has no result.json",
                    str(result_path),
                )
            )
            continue
        try:
            payload = _read_json(result_path)
        except (OSError, json.JSONDecodeError) as exc:
            task_key = _task_key_from_trial_dir(trial_dir)
            warnings.append(
                HarborParseWarning(
                    MALFORMED_RESULT_WARNING,
                    f"Harbor trial result is malformed: {exc}",
                    str(result_path),
                )
            )
            if task_key is not None:
                trials.append(
                    _RawTrial(
                        path=result_path,
                        trial_dir=trial_dir,
                        task_key=task_key,
                        payload=_synthetic_malformed_payload(trial_dir, result_path, exc),
                        malformed=True,
                        malformed_reason=str(exc),
                    )
                )
            continue
        if not isinstance(payload, Mapping):
            task_key = _task_key_from_trial_dir(trial_dir)
            warnings.append(
                HarborParseWarning(
                    MALFORMED_RESULT_WARNING,
                    "Harbor trial result is not a JSON object",
                    str(result_path),
                )
            )
            if task_key is not None:
                trials.append(
                    _RawTrial(
                        path=result_path,
                        trial_dir=trial_dir,
                        task_key=task_key,
                        payload=_synthetic_malformed_payload(
                            trial_dir,
                            result_path,
                            TypeError("trial result is not a JSON object"),
                        ),
                        malformed=True,
                        malformed_reason="not_object",
                    )
                )
            continue
        trial = dict(payload)
        task_key = _harbor_task_key(trial) or _task_key_from_trial_dir(trial_dir)
        if task_key is None:
            warnings.append(
                HarborParseWarning(
                    MALFORMED_RESULT_WARNING,
                    "Harbor trial result does not expose task identity",
                    str(result_path),
                )
            )
            continue
        if "trial_name" not in trial:
            trial["trial_name"] = trial_dir.name
        if "trial_uri" not in trial:
            trial["trial_uri"] = trial_dir.as_uri()
        trials.append(_RawTrial(path=result_path, trial_dir=trial_dir, task_key=task_key, payload=trial))
    return trials


def _group_trials(
    raw_trials: list[_RawTrial],
    parser_context: _ParserContext,
    warnings: list[HarborParseWarning],
) -> dict[str, list[_RawTrial]]:
    del parser_context
    grouped: dict[str, list[_RawTrial]] = {}
    for raw in raw_trials:
        if raw.malformed:
            warnings.append(
                HarborParseWarning(
                    MALFORMED_RESULT_WARNING,
                    f"Constructing failed TrialResult for malformed Harbor trial {raw.trial_dir.name}",
                    str(raw.path),
                )
            )
        grouped.setdefault(raw.task_key, []).append(raw)
    return grouped


def _build_trial_result(
    *,
    raw: _RawTrial,
    trial_id: str,
    parser_context: _ParserContext,
    handle: _HarborHandle,
) -> TrialResult:
    payload = raw.payload
    exception_info = _mapping(payload.get("exception_info"))
    verifier_raw = _mapping(payload.get("verifier_result"))
    rewards = _mapping(verifier_raw.get("rewards")) if verifier_raw else {}
    score = _score_from_rewards(rewards, parser_context.reward_key)
    passed = bool(score and score > 0) if score is not None else None
    resolved = rewards.get("resolved", passed)
    failure: dict[str, Any] | None = None
    status: Literal["completed", "failed", "aborted"] = "completed"
    cancelled_marker = _mapping(payload.get("_cancelled_marker"))
    if cancelled_marker:
        status = "aborted"
        failure = _failure(
            EXTERNAL_HARNESS_CANCELLED,
            f"Harbor job was cancelled: {cancelled_marker.get('reason') or 'unknown'}",
            stage="adapter_shutdown",
            details={"cancelled": cancelled_marker, "trial_name": payload.get("trial_name")},
        )
    elif raw.malformed:
        status = "failed"
        failure = _failure(
            MALFORMED_RESULT_WARNING,
            f"Malformed Harbor trial result: {raw.malformed_reason}",
            stage="parse_results",
            details={"path": str(raw.path)},
        )
    elif exception_info:
        status = _status_from_exception(exception_info)
        failure = _failure(
            HARBOR_TRIAL_EXCEPTION,
            _exception_summary(exception_info),
            stage="execute_trial",
            details={"exception_info": exception_info},
        )
    elif not verifier_raw:
        status = "failed"
        failure = _failure(
            HARBOR_VERIFIER_RESULT_MISSING,
            "Harbor trial result has no verifier_result",
            stage="parse_results",
            details={"trial_name": payload.get("trial_name")},
        )

    artifact_refs: list[ArtifactRef] = []
    artifact_refs.append(
        parser_context.artifact_sink.write_artifact(
            run_id=parser_context.run_id,
            task_id=parser_context.task_id,
            sample_id=_safe_segment(raw.task_key),
            trial_id=trial_id,
            owner="infra",
            name="harbor_raw_result.json",
            content=redact_for_artifact(payload),
            mime_type="application/json",
        )
    )
    artifact_refs.extend(
        _copy_optional_trial_artifacts(
            raw=raw,
            trial_id=trial_id,
            parser_context=parser_context,
            sample_id=_safe_segment(raw.task_key),
        )
    )
    trace_ref = _append_trial_trace(
        raw=raw,
        trial_id=trial_id,
        parser_context=parser_context,
        sample_id=_safe_segment(raw.task_key),
        rewards=rewards,
        score=score,
        passed=passed,
        artifact_refs=artifact_refs,
        failure=failure,
    )

    verifier_result = {
        "status": status,
        "rewards": to_json_compatible(dict(rewards)),
        "reward_key": parser_context.reward_key,
        "reward": score,
        "score": score,
        "passed": passed,
        "resolved": resolved,
        "raw_verifier_result": to_json_compatible(verifier_raw) if verifier_raw else None,
    }
    scheduler_result = {
        "status": status,
        "agent_info": to_json_compatible(payload.get("agent_info")),
        "agent_result": to_json_compatible(payload.get("agent_result")),
        "usage": _usage_from_agent_result(_mapping(payload.get("agent_result"))),
        "harbor": {
            "job_name": handle.job_name,
            "task_key": raw.task_key,
            "task_name": payload.get("task_name"),
            "trial_name": payload.get("trial_name"),
            "trial_uri": payload.get("trial_uri"),
            "source": payload.get("source"),
            "task_checksum": payload.get("task_checksum"),
            "timings": _trial_timings(payload),
        },
    }
    result = TrialResult.model_validate(
        {
            "trial_id": trial_id,
            "status": status,
            "scheduler_result": to_json_compatible(scheduler_result),
            "verifier_result": to_json_compatible(verifier_result),
            "environment_descriptor": to_json_compatible(_environment_descriptor(payload, handle)),
            "artifact_refs": [ref.model_dump(mode="python") for ref in artifact_refs],
            "trace_ref": trace_ref.model_dump(mode="python"),
            "failure": to_json_compatible(failure) if failure is not None else None,
        }
    )
    record_ref = parser_context.artifact_sink.write_trial_record(
        run_id=parser_context.run_id,
        task_id=parser_context.task_id,
        sample_id=_safe_segment(raw.task_key),
        trial_id=trial_id,
        trial_result=result,
    )
    result.artifact_refs.append(record_ref)
    return result


def _copy_optional_trial_artifacts(
    *,
    raw: _RawTrial,
    trial_id: str,
    parser_context: _ParserContext,
    sample_id: str,
) -> list[ArtifactRef]:
    specs = (
        ("infra", "trial.log", raw.trial_dir / "trial.log", "text/plain"),
        ("agent", "trajectory.json", raw.trial_dir / "agent" / "trajectory.json", "application/json"),
        ("verifier", "reward.json", raw.trial_dir / "verifier" / "reward.json", "application/json"),
        ("verifier", "test-stdout.txt", raw.trial_dir / "verifier" / "test-stdout.txt", "text/plain"),
        ("verifier", "test-stderr.txt", raw.trial_dir / "verifier" / "test-stderr.txt", "text/plain"),
    )
    refs: list[ArtifactRef] = []
    for owner, name, source, mime_type in specs:
        if not source.exists():
            continue
        content: Any
        if source.suffix == ".json":
            try:
                content = _read_json(source)
            except (OSError, json.JSONDecodeError):
                content = source.read_text(encoding="utf-8", errors="replace")
        else:
            content = source.read_text(encoding="utf-8", errors="replace")
        ref = parser_context.artifact_sink.write_artifact(
            run_id=parser_context.run_id,
            task_id=parser_context.task_id,
            sample_id=sample_id,
            trial_id=trial_id,
            owner=owner,
            name=name,
            content=redact_for_artifact(content),
            mime_type=mime_type,
        )
        refs.append(ref)
    if not any(ref.owner == "verifier" and ref.name == "reward.json" for ref in refs):
        rewards = _mapping(_mapping(raw.payload.get("verifier_result")).get("rewards"))
        if rewards:
            refs.append(
                parser_context.artifact_sink.write_artifact(
                    run_id=parser_context.run_id,
                    task_id=parser_context.task_id,
                    sample_id=sample_id,
                    trial_id=trial_id,
                    owner="verifier",
                    name="reward.json",
                    content={"rewards": to_json_compatible(dict(rewards))},
                    mime_type="application/json",
                )
            )
    return refs


def _append_trial_trace(
    *,
    raw: _RawTrial,
    trial_id: str,
    parser_context: _ParserContext,
    sample_id: str,
    rewards: Mapping[str, Any],
    score: float | None,
    passed: bool | None,
    artifact_refs: list[ArtifactRef],
    failure: Mapping[str, Any] | None,
) -> ArtifactRef:
    if failure is not None:
        failure_code = str(failure.get("failure_code") or failure.get("code") or "")
        return parser_context.artifact_sink.append_trace_event(
            run_id=parser_context.run_id,
            task_id=parser_context.task_id,
            sample_id=sample_id,
            trial_id=trial_id,
            actor="runtime",
            event_type=(
                "external_harness.cancelled"
                if failure_code == EXTERNAL_HARNESS_CANCELLED
                else "external_harness.trial_exception"
            ),
            payload={
                "trial_name": raw.payload.get("trial_name"),
                "failure": to_json_compatible(dict(failure)),
            },
            artifact_refs=artifact_refs,
        )
    return parser_context.artifact_sink.append_trace_event(
        run_id=parser_context.run_id,
        task_id=parser_context.task_id,
        sample_id=sample_id,
        trial_id=trial_id,
        actor="verifier",
        event_type="verifier.result",
        payload={
            "metric": {
                "score": score,
                "passed": passed,
                "reward_key": parser_context.reward_key,
            },
            "rewards": to_json_compatible(dict(rewards)),
            "trial_name": raw.payload.get("trial_name"),
        },
        artifact_refs=artifact_refs,
    )


def _build_sample(
    *,
    sample_id: str,
    raw_trials: list[_RawTrial],
    trial_results: list[TrialResult],
    aggregate: Any,
    job_result: dict[str, Any],
    handle: _HarborHandle,
    warnings: list[HarborParseWarning],
    parser_context: _ParserContext,
    trace_translator: AgentTraceTranslator | None = None,
) -> dict[str, Any]:
    primary_raw = raw_trials[0].payload
    instruction, raw_sample = _raw_sample(primary_raw)
    trajectory_payload = _trajectory_payload(raw_trials[0])
    trajectory = _normalize_trajectory(trajectory_payload)
    agent_trace = (
        trace_translator.translate(
            trajectory_payload or trajectory,
            context={
                "trial_id": trial_results[0].trial_id,
                "agent_info": _mapping(primary_raw.get("agent_info")),
                "final_metrics": _mapping(_mapping(trajectory_payload).get("final_metrics")),
                "primary_raw_trial": primary_raw,
            },
        )
        if trace_translator is not None
        else []
    )
    final_answer = _final_answer(primary_raw, trajectory)
    pass_values = [_trial_pass_value(trial) for trial in trial_results]
    aggregate_payload = aggregate.to_dict() if hasattr(aggregate, "to_dict") else to_json_compatible(aggregate)
    cancelled_marker = _mapping(primary_raw.get("_cancelled_marker"))
    sample = Sample(
        schema_version=SCHEMA_VERSION,
        id=sample_id,
        messages=_messages(instruction, trajectory),
        instruction=instruction,
        prompt=instruction,
        text=instruction,
        inputs=_inputs(primary_raw),
        task_type="external_harness.harbor",
        references=[],
        sandbox={
            "launcher": {"mode": handle.invocation_metadata.get("launcher_mode")},
            "harbor_environment": redact_for_artifact(handle.environment),
        },
        metadata={
            "_harness": {
                "kit_id": "harbor",
                "job_name": handle.job_name,
                "harbor_task_key": sample_id,
                "harbor_trial_names": [str(raw.payload.get("trial_name") or raw.trial_dir.name) for raw in raw_trials],
            },
            "dataset_id": parser_context.dataset_id,
            "trial_count": len(trial_results),
            "primary_trial_id": trial_results[0].trial_id,
            "input_source": "task_dir" if raw_sample["available"] else "harbor_result",
            "warnings": [warning.to_dict() for warning in warnings],
        },
        data_tag={
            "harness": "harbor",
            "dataset_ref": _dataset_ref(primary_raw),
            "agent_name": _mapping(primary_raw.get("agent_info")).get("name"),
            "model": _mapping(_mapping(primary_raw.get("agent_info")).get("model_info")).get("name"),
        },
        raw_assets={
            "harbor_task": raw_sample,
            "job_config_path": str(handle.job_config_path) if handle.job_config_path else None,
            "job_result_path": str(handle.job_dir / "result.json"),
        },
        tools=_tools_from_trajectory(trajectory),
        predict_result=[
            PredictResult(
                index=0,
                message=Message(role="assistant", content=[MessageContent(type="text", text=final_answer or "")]),
                raw_response={
                    "prediction": {
                        "final_answer": final_answer,
                        "trajectory": trajectory,
                        "raw_agent_result": primary_raw.get("agent_result") or {},
                    }
                },
                usage=_usage_from_agent_result(_mapping(primary_raw.get("agent_result"))),
                agent_trace=agent_trace or None,
            )
        ],
        eval_result={
            **aggregate.samples_jsonl_projection,
            "_aggregate": aggregate_payload,
            "harbor_resolve_rate": aggregate.pass_rate,
            "harbor_score_mean": aggregate.score_mean,
            "harbor_cost_usd": _job_cost(job_result),
            "external_trial_pass_values": pass_values,
            "external_trial_metric_projection": aggregate.metric_projection,
            "external_harness_warnings": [warning.to_dict() for warning in warnings],
            **(
                {"external_harness_cancelled": to_json_compatible(dict(cancelled_marker))}
                if cancelled_marker
                else {}
            ),
        },
    )
    payload = sample_to_dict(sample)
    payload.update(
        {
            "sample_id": sample_id,
            "dataset_id": parser_context.dataset_id,
            "dataset_source": {
                "kind": "harbor",
                "benchmark": _benchmark(primary_raw),
                "version": _dataset_version(primary_raw),
                "task_name": primary_raw.get("task_name"),
                "task_id": to_json_compatible(primary_raw.get("task_id")),
                "task_checksum": primary_raw.get("task_checksum"),
            },
            "raw_sample": raw_sample,
            "prediction": {
                "final_answer": final_answer,
                "trajectory": trajectory,
                "raw_agent_result": to_json_compatible(primary_raw.get("agent_result") or {}),
            },
            "evaluation": {
                "passed": pass_values[0] if pass_values else None,
                "score": _trial_score(trial_results[0]) if trial_results else None,
                "rewards": _mapping(trial_results[0].verifier_result.get("rewards")) if trial_results else {},
                "raw_verifier_result": primary_raw.get("verifier_result") or {},
            },
            "trials": [
                {
                    "trial_id": trial.trial_id,
                    "trial_name": raw.payload.get("trial_name"),
                    "passed": _trial_pass_value(trial),
                    "score": _trial_score(trial),
                    "exception_info": raw.payload.get("exception_info"),
                }
                for raw, trial in zip(raw_trials, trial_results, strict=False)
            ],
            "provenance": {
                "external_harness": "harbor",
                "job_name": handle.job_name,
                "trial_uri": primary_raw.get("trial_uri"),
            },
        }
    )
    return to_json_compatible(payload)


def _read_job_result(handle: _HarborHandle) -> tuple[dict[str, Any], bool]:
    path = handle.job_dir / "result.json"
    if not path.exists():
        return {}, True
    payload = _read_json(path)
    if not isinstance(payload, Mapping):
        raise ExternalHarnessParseError(
            MALFORMED_RESULT_WARNING,
            f"Harbor job result is not a JSON object: {path}",
        )
    return dict(payload), False


def _read_launcher_result(handle: _HarborHandle, payload: Mapping[str, Any]) -> dict[str, Any]:
    launcher = payload.get("launcher_result")
    if isinstance(launcher, Mapping):
        return dict(launcher)
    if handle.launcher_result_path and handle.launcher_result_path.exists():
        try:
            data = _read_json(handle.launcher_result_path)
        except (OSError, json.JSONDecodeError):
            return {}
        return dict(data) if isinstance(data, Mapping) else {}
    return {}


def _read_cancelled_marker(handle: _HarborHandle) -> tuple[dict[str, Any], Path | None]:
    for path in (handle.job_dir / "cancelled.json", handle.workdir / "cancelled.json"):
        if not path.exists():
            continue
        try:
            payload = _read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(payload, Mapping) and str(payload.get("status") or "").lower() == "cancelled":
            return dict(payload), path
    return {}, None


def _synthetic_cancelled_trial(handle: _HarborHandle, marker: Mapping[str, Any]) -> _RawTrial:
    job_config = _mapping(handle.invocation_metadata.get("job_config"))
    task_config = _first_mapping(job_config.get("tasks"))
    datasets = job_config.get("datasets")
    dataset_config = _first_mapping(datasets)
    task_path = task_config.get("path")
    task_key = _safe_segment(
        str(
            marker.get("task_name")
            or marker.get("task_id")
            or (Path(str(task_path)).name if task_path else "")
            or handle.job_name
        )
    )
    trial_dir = handle.job_dir / f"{task_key}__cancelled"
    trial_name = trial_dir.name
    payload = {
        "task_name": task_key,
        "task_id": {"path": str(task_path)} if task_path else {"name": task_key},
        "source": task_config.get("source") or dataset_config.get("name") or "harbor",
        "task_checksum": None,
        "trial_name": trial_name,
        "trial_uri": _path_uri(trial_dir),
        "config": {
            "task": to_json_compatible(dict(task_config)),
            "environment": to_json_compatible(dict(handle.environment)),
        },
        "agent_result": {
            "final_answer": "",
            "metadata": {"cancelled": True},
        },
        "_cancelled_marker": to_json_compatible(dict(marker)),
    }
    return _RawTrial(
        path=trial_dir / "result.json",
        trial_dir=trial_dir,
        task_key=task_key,
        payload=payload,
    )


def _resolve_handle(*, handle: Any, result_payload: Mapping[str, Any]) -> _HarborHandle:
    candidate = handle or result_payload.get("handle") or result_payload.get("harbor_job_handle")
    if candidate is None:
        candidate = result_payload
    mapping = _object_mapping(candidate)
    if not mapping:
        raise ExternalHarnessParseError(MALFORMED_RESULT_WARNING, "Missing Harbor job handle")
    jobs_dir = Path(str(mapping.get("jobs_dir") or Path(mapping.get("job_dir", ".")).parent))
    job_name = str(mapping.get("job_name") or Path(str(mapping.get("job_dir", ""))).name)
    job_dir = Path(str(mapping.get("job_dir") or jobs_dir / job_name))
    if not job_name:
        job_name = job_dir.name
    workdir = Path(str(mapping.get("workdir") or jobs_dir.parent))
    return _HarborHandle(
        job_name=job_name,
        jobs_dir=jobs_dir,
        job_dir=job_dir,
        job_config_path=_optional_path(mapping.get("job_config_path")),
        launcher_result_path=_optional_path(mapping.get("launcher_result_path")),
        workdir=workdir,
        environment=dict(_mapping(mapping.get("environment"))),
        invocation_metadata=dict(_mapping(mapping.get("invocation_metadata"))),
    )


def _parser_context(
    *,
    context: Any,
    handle: _HarborHandle,
    reward_key: str,
    aggregation: str | None,
    expected_trials: int | None,
    dut_id: str,
) -> _ParserContext:
    cache_store = getattr(context, "cache_store", None)
    run_id = str(getattr(cache_store, "run_id", None) or getattr(context, "run_id", None) or "external_harness_run")
    task_id = _safe_segment(str(getattr(context, "task_id", None) or "external_harness_task"))
    dataset_id = _safe_segment(str(getattr(context, "dataset_id", None) or "external_harness_dataset"))
    if cache_store is not None:
        base_dir = str(cache_store.run_dir.parent)
    else:
        base_dir = str(handle.workdir / "runs")
    return _ParserContext(
        run_id=_safe_segment(run_id),
        task_id=task_id,
        dataset_id=dataset_id,
        artifact_sink=RuntimeArtifactSink(base_dir=base_dir),
        reward_key=reward_key,
        aggregation=aggregation,
        expected_trials=expected_trials,
        dut_id=dut_id,
    )


def _payload_mapping(value: Any) -> Mapping[str, Any]:
    payload = getattr(value, "payload", None)
    if isinstance(payload, Mapping):
        return payload
    return value if isinstance(value, Mapping) else {}


def _payload_aggregation(payload: Mapping[str, Any]) -> str | None:
    trial_policy = _mapping(payload.get("trial_policy"))
    value = trial_policy.get("aggregation") or payload.get("aggregation")
    return str(value) if value else None


def _payload_expected_trials(payload: Mapping[str, Any], handle: _HarborHandle) -> int | None:
    for value in (
        payload.get("expected_total_trials"),
        _mapping(payload.get("trial_policy")).get("trials"),
        handle.invocation_metadata.get("expected_total_trials"),
    ):
        try:
            if value is not None:
                return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _object_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "to_dict") and callable(value.to_dict):
        resolved = value.to_dict()
        if isinstance(resolved, Mapping):
            return resolved
    if hasattr(value, "__dataclass_fields__"):
        return {key: getattr(value, key) for key in value.__dataclass_fields__}
    if hasattr(value, "__dict__"):
        return vars(value)
    return {}


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, list):
        for item in value:
            if isinstance(item, Mapping):
                return item
    return {}


def _optional_path(value: Any) -> Path | None:
    return Path(str(value)) if value else None


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _path_uri(path: Path) -> str:
    try:
        return path.as_uri()
    except ValueError:
        return str(path)


def _launcher_failed(payload: Mapping[str, Any]) -> bool:
    if not payload:
        return False
    exit_code = payload.get("exit_code")
    return (exit_code is not None and int(exit_code) != 0) or bool(payload.get("launcher_error") or payload.get("error"))


def _harbor_task_key(payload: Mapping[str, Any]) -> str | None:
    task_name = payload.get("task_name")
    if task_name:
        return str(task_name)
    task_id = _mapping(payload.get("task_id"))
    path = task_id.get("path")
    if path:
        return Path(str(path)).name
    return None


def _task_key_from_trial_dir(trial_dir: Path) -> str | None:
    name = trial_dir.name
    if "__" in name:
        name = name.split("__", 1)[0]
    return name or None


def _synthetic_malformed_payload(trial_dir: Path, result_path: Path, error: BaseException) -> dict[str, Any]:
    return {
        "task_name": _task_key_from_trial_dir(trial_dir),
        "trial_name": trial_dir.name,
        "trial_uri": trial_dir.as_uri(),
        "exception_info": {
            "exception_type": error.__class__.__name__,
            "exception_message": str(error),
        },
        "_malformed_result_path": str(result_path),
    }


def _score_from_rewards(rewards: Mapping[str, Any], reward_key: str) -> float | None:
    for key in (reward_key, "reward"):
        value = rewards.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _status_from_exception(exception_info: Mapping[str, Any]) -> Literal["failed", "aborted"]:
    text = json.dumps(to_json_compatible(dict(exception_info)), ensure_ascii=False).lower()
    if "timeout" in text or "cancel" in text:
        return "aborted"
    return "failed"


def _exception_summary(exception_info: Mapping[str, Any]) -> str:
    exc_type = exception_info.get("exception_type") or exception_info.get("type") or "Exception"
    message = exception_info.get("exception_message") or exception_info.get("message") or ""
    return f"Harbor trial exception: {exc_type}: {message}".strip()


def _failure(
    code: str,
    summary: str,
    *,
    stage: str,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "failure_domain": "external_harness",
        "failure_stage": stage,
        "failure_code": code,
        "code": code,
        "summary": summary,
        "retryable": False,
        "details": to_json_compatible(dict(details or {})),
    }


def _usage_from_agent_result(agent_result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_input_tokens": agent_result.get("n_input_tokens"),
        "n_cache_tokens": agent_result.get("n_cache_tokens"),
        "n_output_tokens": agent_result.get("n_output_tokens"),
        "cost_usd": agent_result.get("cost_usd"),
        "iterations": _mapping(agent_result.get("metadata")).get("n_episodes"),
    }


def _trial_timings(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: to_json_compatible(payload.get(key))
        for key in ("started_at", "finished_at", "environment_setup", "agent_setup", "agent_execution", "verifier")
        if key in payload
    }


def _environment_descriptor(payload: Mapping[str, Any], handle: _HarborHandle) -> dict[str, Any]:
    config = _mapping(payload.get("config"))
    return {
        "external_harness": "harbor",
        "job_name": handle.job_name,
        "task_id": to_json_compatible(payload.get("task_id")),
        "source": payload.get("source"),
        "task_checksum": payload.get("task_checksum"),
        "harbor_environment": redact_for_artifact(handle.environment or _mapping(config.get("environment"))),
    }


def _raw_sample(payload: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    task_id = _mapping(payload.get("task_id"))
    task_path = task_id.get("path") or _mapping(_mapping(payload.get("config")).get("task")).get("path")
    if not task_path:
        return None, {"available": False, "reason": "missing_task_path", "payload": {}}
    task_dir = Path(str(task_path))
    instruction_path = task_dir / "instruction.md"
    task_toml = task_dir / "task.toml"
    payload_data: dict[str, Any] = {}
    if task_toml.exists():
        payload_data["task_toml"] = task_toml.read_text(encoding="utf-8", errors="replace")
    if instruction_path.exists():
        instruction = instruction_path.read_text(encoding="utf-8", errors="replace")
        payload_data["instruction"] = instruction
        return instruction, {"available": True, "source": str(task_dir), "payload": payload_data}
    return None, {"available": False, "source": str(task_dir), "reason": "instruction_missing", "payload": payload_data}


def _trajectory(raw: _RawTrial) -> list[dict[str, Any]]:
    return _normalize_trajectory(_trajectory_payload(raw))


def _trajectory_payload(raw: _RawTrial) -> Any:
    path = raw.trial_dir / "agent" / "trajectory.json"
    if not path.exists():
        return _trajectory_from_rollout(raw.payload)
    try:
        return _read_json(path)
    except (OSError, json.JSONDecodeError):
        return _trajectory_from_rollout(raw.payload)


def _trajectory_from_rollout(payload: Mapping[str, Any]) -> list[dict[str, Any]]:
    rollout_details = _mapping(payload.get("agent_result")).get("rollout_details")
    if isinstance(rollout_details, list):
        return _normalize_trajectory(rollout_details)
    return []


def _normalize_trajectory(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, Mapping):
        for key in ("trajectory", "messages", "steps", "events"):
            if isinstance(value.get(key), list):
                return [_normalize_trajectory_item(item) for item in value[key]]
        return [_normalize_trajectory_item(value)]
    if isinstance(value, list):
        return [_normalize_trajectory_item(item) for item in value]
    return []


def _normalize_trajectory_item(item: Any) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        return {"type": "text", "text": str(item)}
    payload = to_json_compatible(dict(item))
    tool_name = payload.get("tool_name") or payload.get("name")
    if payload.get("type") == "tool_call" or tool_name or payload.get("tool_calls"):
        return {
            "type": "tool_call",
            "tool_name": tool_name,
            "arguments": payload.get("arguments") or payload.get("args") or {},
            "output": payload.get("output") or payload.get("result") or payload.get("observation"),
            "raw": payload,
        }
    return payload


def _final_answer(payload: Mapping[str, Any], trajectory: list[dict[str, Any]]) -> str | None:
    agent_result = _mapping(payload.get("agent_result"))
    for key in ("final_answer", "answer", "output", "response"):
        if agent_result.get(key) is not None:
            return str(agent_result[key])
    for item in reversed(trajectory):
        for key in ("final_answer", "answer", "content", "text", "output"):
            value = item.get(key)
            if isinstance(value, str) and value:
                return value
    return None


def _messages(instruction: str | None, trajectory: list[dict[str, Any]]) -> list[Message]:
    if instruction:
        return [Message(role="user", content=[MessageContent(type="text", text=instruction)])]
    for item in trajectory:
        if item.get("role") in {"system", "user"} and item.get("content"):
            return [Message(role=str(item["role"]), content=[MessageContent(type="text", text=str(item["content"]))])]
    return []


def _tools_from_trajectory(trajectory: list[dict[str, Any]]) -> list[Any] | None:
    for item in trajectory:
        raw = _mapping(item.get("raw"))
        tools = raw.get("tool_definitions") or item.get("tool_definitions")
        if isinstance(tools, list):
            return tools
    return None


def _inputs(payload: Mapping[str, Any]) -> dict[str, Any]:
    config = _mapping(payload.get("config"))
    task_config = _mapping(config.get("task"))
    return {
        "source": payload.get("source"),
        "task_name": payload.get("task_name"),
        "task_id": to_json_compatible(payload.get("task_id")),
        "task_checksum": payload.get("task_checksum"),
        "task_config": to_json_compatible(task_config),
        "task_path_ref": _mapping(payload.get("task_id")).get("path") or task_config.get("path"),
        "git_url": task_config.get("git_url"),
        "git_commit_id": task_config.get("git_commit_id"),
    }


def _input_ref(payload: Mapping[str, Any], sample_id: str) -> dict[str, Any]:
    return {
        "sample_id": sample_id,
        "task_name": payload.get("task_name"),
        "task_id": to_json_compatible(payload.get("task_id")),
        "task_checksum": payload.get("task_checksum"),
    }


def _dataset_ref(payload: Mapping[str, Any]) -> str:
    source = payload.get("source") or _mapping(_mapping(payload.get("config")).get("task")).get("source")
    return str(source or "harbor")


def _benchmark(payload: Mapping[str, Any]) -> str:
    source = _dataset_ref(payload)
    if source == "terminal-bench":
        return source
    return source.split("/", 1)[0]


def _dataset_version(payload: Mapping[str, Any]) -> str | None:
    source = _dataset_ref(payload)
    if source == "terminal-bench":
        return "2.0"
    return None


def _job_cost(job_result: Mapping[str, Any]) -> Any:
    stats = _mapping(job_result.get("stats"))
    return stats.get("cost_usd")


def _trial_pass_value(trial: TrialResult) -> bool | None:
    for key in ("passed", "resolved", "pass"):
        value = trial.verifier_result.get(key)
        if isinstance(value, bool):
            return value
    return None


def _trial_score(trial: TrialResult) -> float | None:
    value = trial.verifier_result.get("score") or trial.verifier_result.get("reward")
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sample_status(trial_results: Iterable[TrialResult]) -> str:
    statuses = [trial.status for trial in trial_results]
    if any(status == "completed" for status in statuses):
        return "completed"
    if any(status == "aborted" for status in statuses):
        return "aborted"
    return "failed"


def _sample_failure(trial_results: Iterable[TrialResult]) -> dict[str, Any] | None:
    for trial in trial_results:
        if trial.failure:
            return trial.failure
    return None


def _task_metrics(samples: list[HarborImportedSample]) -> dict[str, Any]:
    pass_values: list[bool | None] = []
    scores: list[float] = []
    aborted_count = 0
    for sample in samples:
        aborted_count += sum(1 for trial in sample.trial_results if trial.status == "aborted")
        pass_values.extend(
            value
            for value in sample.sample.get("eval_result", {}).get("external_trial_pass_values", [])
            if value is not None
        )
        score = sample.sample.get("eval_result", {}).get("harbor_score_mean")
        if score is not None:
            scores.append(float(score))
    return {
        "sample_count": len(samples),
        "aborted_count": aborted_count,
        "harbor_resolve_rate": (sum(1 for value in pass_values if value) / len(pass_values)) if pass_values else None,
        "harbor_score_mean": (sum(scores) / len(scores)) if scores else None,
    }


def _unique_artifact_refs(trial_results: list[TrialResult]) -> list[ArtifactRef]:
    seen: set[str] = set()
    refs: list[ArtifactRef] = []
    for trial in trial_results:
        for ref in [trial.trace_ref, *trial.artifact_refs]:
            key = ref.path
            if key in seen:
                continue
            seen.add(key)
            refs.append(ref)
    return refs


def _safe_segment(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value or "sample")).strip("_")
    return safe or "sample"


__all__ = [
    "EXTERNAL_HARNESS_CANCELLED",
    "HARBOR_JOB_RESULT_MISSING",
    "HARBOR_LAUNCHER_FAILED",
    "HARBOR_TRIAL_EXCEPTION",
    "HARBOR_VERIFIER_RESULT_MISSING",
    "HarborImportedSample",
    "HarborParseWarning",
    "HarborResultBundle",
    "parse_harbor_results",
]
