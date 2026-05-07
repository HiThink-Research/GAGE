from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from gage_eval.agent_runtime.artifacts import RuntimeArtifactSink
from gage_eval.agent_runtime.serialization import to_json_compatible
from gage_eval.agent_runtime.trace_schema import ArtifactRef, TrialResult


TRIAL_POLICY_INVALID = "config.trial_policy.invalid"


class TrialPolicyError(ValueError):
    """Raised when a trial policy is invalid for AgentKit v2 Phase 1."""

    code = TRIAL_POLICY_INVALID

    def __init__(self, reason: str) -> None:
        super().__init__(f"{self.code}: {reason}")
        self.reason = reason


@dataclass(frozen=True)
class TrialPolicy:
    """Normalized AgentRuntime trial policy."""

    trials: int = 1
    environment_scope: Literal["per_trial"] = "per_trial"
    parallelism: int = 1
    aggregation: Literal["single", "all"] = "single"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any] | None) -> "TrialPolicy":
        raw = dict(payload or {})
        trials = _coerce_positive_int(raw.get("trials", 1), field_name="trials")
        environment_scope = str(raw.get("environment_scope", "per_trial"))
        parallelism = _coerce_positive_int(raw.get("parallelism", 1), field_name="parallelism")
        aggregation = str(raw.get("aggregation") or ("single" if trials == 1 else "all"))

        if environment_scope != "per_trial":
            raise TrialPolicyError("environment_scope must be per_trial in Phase 1")
        if parallelism != 1:
            raise TrialPolicyError("parallelism greater than 1 is not supported in Phase 1")
        if aggregation not in {"single", "all"}:
            raise TrialPolicyError("aggregation must be single or all")
        if aggregation == "single" and trials != 1:
            raise TrialPolicyError("aggregation=single is only valid when trials=1")

        return cls(
            trials=trials,
            environment_scope="per_trial",
            parallelism=parallelism,
            aggregation=aggregation,  # type: ignore[arg-type]
        )

    def trial_ids(self) -> list[str]:
        return [f"trial_{index:04d}" for index in range(1, self.trials + 1)]

    def to_dict(self) -> dict[str, Any]:
        return {
            "trials": self.trials,
            "environment_scope": self.environment_scope,
            "parallelism": self.parallelism,
            "aggregation": self.aggregation,
        }


@dataclass
class TrialAggregate:
    """Sample-level aggregate derived from one or more trial results."""

    aggregation: Literal["single", "all"]
    trial_count: int
    completed_trial_count: int
    failed_trial_count: int
    primary_trial_id: str
    score_mean: float | None
    score_min: float | None
    score_max: float | None
    pass_count: int | None
    pass_rate: float | None
    samples_jsonl_projection: dict[str, Any]
    metric_projection: dict[str, Any]
    failure_rollup: dict[str, Any]
    trial_result_refs: list[ArtifactRef]
    trial_results: list[TrialResult] = field(repr=False)

    def to_dict(self) -> dict[str, Any]:
        return {
            "aggregation": self.aggregation,
            "trial_count": self.trial_count,
            "completed_trial_count": self.completed_trial_count,
            "failed_trial_count": self.failed_trial_count,
            "primary_trial_id": self.primary_trial_id,
            "score_mean": self.score_mean,
            "score_min": self.score_min,
            "score_max": self.score_max,
            "pass_count": self.pass_count,
            "pass_rate": self.pass_rate,
            "samples_jsonl_projection": to_json_compatible(self.samples_jsonl_projection),
            "metric_projection": to_json_compatible(self.metric_projection),
            "failure_rollup": to_json_compatible(self.failure_rollup),
            "trial_result_refs": [ref.model_dump(mode="python") for ref in self.trial_result_refs],
        }


class TrialManager:
    """Runs sample-local trials sequentially and writes trial-scoped evidence."""

    def __init__(
        self,
        *,
        artifact_sink: RuntimeArtifactSink,
        scheduler_acquire: Callable[..., Any],
        scheduler_run: Callable[..., dict[str, Any]],
        verifier_run: Callable[..., dict[str, Any]],
        verifier_acquire: Callable[..., Any] | None = None,
        verifier_environment_policy: Literal["reuse", "fresh_from_profile"] = "reuse",
    ) -> None:
        self.artifact_sink = artifact_sink
        self.scheduler_acquire = scheduler_acquire
        self.scheduler_run = scheduler_run
        self.verifier_run = verifier_run
        self.verifier_acquire = verifier_acquire
        self.verifier_environment_policy = verifier_environment_policy

    def run(
        self,
        *,
        run_id: str,
        task_id: str,
        sample_id: str,
        trial_policy: dict[str, Any] | TrialPolicy | None,
        sample: dict[str, Any] | None = None,
        dut_id: str | None = None,
    ) -> TrialAggregate:
        policy = trial_policy if isinstance(trial_policy, TrialPolicy) else TrialPolicy.from_mapping(trial_policy)
        trial_results: list[TrialResult] = []

        for trial_index, trial_id in enumerate(policy.trial_ids(), start=1):
            trial_results.append(
                self._run_one_trial(
                    run_id=run_id,
                    task_id=task_id,
                    sample_id=sample_id,
                    sample=sample or {},
                    dut_id=dut_id,
                    policy=policy,
                    trial_id=trial_id,
                    trial_index=trial_index,
                )
            )

        aggregate = aggregate_trial_results(trial_results, aggregation=policy.aggregation)
        self.artifact_sink.write_trial_aggregate(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            aggregate=aggregate,
        )
        return aggregate

    def _run_one_trial(
        self,
        *,
        run_id: str,
        task_id: str,
        sample_id: str,
        sample: dict[str, Any],
        dut_id: str | None,
        policy: TrialPolicy,
        trial_id: str,
        trial_index: int,
    ) -> TrialResult:
        artifact_refs: list[ArtifactRef] = []
        trace_ref = self.artifact_sink.append_trace_event(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            trial_id=trial_id,
            actor="runtime",
            event_type="trial.start",
            payload={
                "trial_id": trial_id,
                "trial_index": trial_index,
                "trial_policy": policy.to_dict(),
            },
        )

        scheduler_result: dict[str, Any] = {}
        verifier_result: dict[str, Any] = {}
        failure: dict[str, Any] | None = None
        status: Literal["completed", "failed", "aborted"] = "completed"
        scheduler_lease: Any = None
        environment_descriptor: dict[str, Any] = {}

        try:
            scheduler_lease = self.scheduler_acquire(
                run_id=run_id,
                task_id=task_id,
                sample_id=sample_id,
                trial_id=trial_id,
                trial_index=trial_index,
                sample=sample,
                dut_id=dut_id,
            )
            environment_descriptor = _lease_descriptor(scheduler_lease)
            trace_ref = self.artifact_sink.append_trace_event(
                run_id=run_id,
                task_id=task_id,
                sample_id=sample_id,
                trial_id=trial_id,
                actor="environment",
                event_type="environment.acquire",
                payload={"role": "scheduler", "environment_descriptor": environment_descriptor},
            )
            scheduler_result = self.scheduler_run(
                run_id=run_id,
                task_id=task_id,
                sample_id=sample_id,
                trial_id=trial_id,
                trial_index=trial_index,
                sample=sample,
                scheduler_lease=scheduler_lease,
            )
            artifact_refs.append(
                self.artifact_sink.write_artifact(
                    run_id=run_id,
                    task_id=task_id,
                    sample_id=sample_id,
                    trial_id=trial_id,
                    owner="agent",
                    name="scheduler_result.json",
                    content=scheduler_result,
                    mime_type="application/json",
                )
            )

            verifier_lease = scheduler_lease
            if self.verifier_environment_policy == "fresh_from_profile":
                if self.verifier_acquire is None:
                    raise RuntimeError("verifier_acquire is required for fresh_from_profile")
                verifier_lease = self.verifier_acquire(
                    run_id=run_id,
                    task_id=task_id,
                    sample_id=sample_id,
                    trial_id=trial_id,
                    trial_index=trial_index,
                    sample=sample,
                    dut_id=dut_id,
                )
                trace_ref = self.artifact_sink.append_trace_event(
                    run_id=run_id,
                    task_id=task_id,
                    sample_id=sample_id,
                    trial_id=trial_id,
                    actor="environment",
                    event_type="environment.acquire",
                    payload={"role": "verifier", "environment_descriptor": _lease_descriptor(verifier_lease)},
                )

            verifier_result = self.verifier_run(
                run_id=run_id,
                task_id=task_id,
                sample_id=sample_id,
                trial_id=trial_id,
                trial_index=trial_index,
                sample=sample,
                scheduler_result=scheduler_result,
                scheduler_lease=scheduler_lease,
                verifier_lease=verifier_lease,
            )
            artifact_refs.append(
                self.artifact_sink.write_artifact(
                    run_id=run_id,
                    task_id=task_id,
                    sample_id=sample_id,
                    trial_id=trial_id,
                    owner="verifier",
                    name="verifier_result.json",
                    content=verifier_result,
                    mime_type="application/json",
                )
            )
            status = _trial_status(scheduler_result, verifier_result)
            failure = _extract_failure(scheduler_result, verifier_result)
        except Exception as exc:
            status = "failed"
            failure = _trial_execution_failure(exc)
            verifier_result = verifier_result or {"status": "failed", "failure": failure}

        result = TrialResult(
            trial_id=trial_id,
            status=status,
            scheduler_result=to_json_compatible(scheduler_result or {}),
            verifier_result=to_json_compatible(verifier_result or {}),
            environment_descriptor=to_json_compatible(environment_descriptor),
            artifact_refs=artifact_refs,
            trace_ref=trace_ref,
            failure=to_json_compatible(failure) if failure is not None else None,
        )
        record_ref = self.artifact_sink.write_trial_record(
            run_id=run_id,
            task_id=task_id,
            sample_id=sample_id,
            trial_id=trial_id,
            trial_result=result,
        )
        result.artifact_refs.append(record_ref)
        return result


def aggregate_trial_results(
    trial_results: list[TrialResult],
    *,
    aggregation: Literal["single", "all"] | str | None = None,
) -> TrialAggregate:
    if not trial_results:
        raise ValueError("trial.aggregate_failed: at least one TrialResult is required")

    resolved_aggregation = str(aggregation or ("single" if len(trial_results) == 1 else "all"))
    if resolved_aggregation not in {"single", "all"}:
        raise ValueError("trial.aggregate_failed: unsupported aggregation")
    if resolved_aggregation == "single" and len(trial_results) != 1:
        raise TrialPolicyError("aggregation=single is only valid when trials=1")

    primary = trial_results[0]
    completed = [result for result in trial_results if result.status == "completed"]
    failed = [result for result in trial_results if result.status != "completed"]
    scores = [_score_from_verifier(result.verifier_result) for result in completed]
    scores = [score for score in scores if score is not None]
    pass_values = [_pass_from_verifier(result.verifier_result) for result in completed]
    observed_pass_values = [value for value in pass_values if value is not None]

    score_mean = (sum(scores) / len(scores)) if scores else None
    pass_count = sum(1 for value in observed_pass_values if value) if observed_pass_values else None
    pass_rate = (pass_count / len(completed)) if pass_count is not None and completed else None
    primary_scalars = _primary_scalar_projection(primary)
    trial_ids = [result.trial_id for result in trial_results]
    skipped_failed_trials = [
        {
            "trial_id": result.trial_id,
            "failure_code": _failure_code(result),
            "status": result.status,
        }
        for result in failed
    ]

    return TrialAggregate(
        aggregation=resolved_aggregation,  # type: ignore[arg-type]
        trial_count=len(trial_results),
        completed_trial_count=len(completed),
        failed_trial_count=len(failed),
        primary_trial_id=primary.trial_id,
        score_mean=score_mean,
        score_min=min(scores) if scores else None,
        score_max=max(scores) if scores else None,
        pass_count=pass_count,
        pass_rate=pass_rate,
        samples_jsonl_projection={
            "primary_trial_id": primary.trial_id,
            **primary_scalars,
            "agent_eval.trial_aggregate": {
                "source": "aggregate",
                "fields": ["pass_rate", "pass_count", "trial_count", "score_mean", "score_min", "score_max"],
            },
        },
        metric_projection={
            "primary_trial_id": primary.trial_id,
            "trial_ids": trial_ids,
            "scalar_fields": sorted(primary_scalars),
            "aggregate_fields": ["pass_rate", "pass_count", "trial_count", "score_mean", "score_min", "score_max"],
            "skipped_failed_trials": skipped_failed_trials,
        },
        failure_rollup=_failure_rollup(trial_results),
        trial_result_refs=[_trial_result_ref(result) for result in trial_results],
        trial_results=list(trial_results),
    )


def _coerce_positive_int(value: Any, *, field_name: str) -> int:
    if isinstance(value, bool):
        raise TrialPolicyError(f"{field_name} must be an integer")
    try:
        number = int(value)
    except (TypeError, ValueError) as exc:
        raise TrialPolicyError(f"{field_name} must be an integer") from exc
    if number < 1:
        raise TrialPolicyError(f"{field_name} must be >= 1")
    return number


def _score_from_verifier(verifier_result: dict[str, Any]) -> float | None:
    for key in ("score", "reward"):
        value = verifier_result.get(key)
        if value is None or isinstance(value, bool):
            continue
        try:
            return float(value)
        except (TypeError, ValueError):
            continue
    return None


def _pass_from_verifier(verifier_result: dict[str, Any]) -> bool | None:
    for key in ("resolved", "passed", "pass"):
        value = verifier_result.get(key)
        if isinstance(value, bool):
            return value
    return None


def _primary_scalar_projection(primary: TrialResult) -> dict[str, Any]:
    projection: dict[str, Any] = {
        "status": {"value": primary.status, "source_trial_id": primary.trial_id},
    }
    for key, value in primary.verifier_result.items():
        if key == "status":
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            projection[key] = {"value": value, "source_trial_id": primary.trial_id}
    return projection


def _failure_rollup(trial_results: list[TrialResult]) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    failure_codes: dict[str, int] = {}
    failure_domains: dict[str, int] = {}
    for result in trial_results:
        status_counts[result.status] = status_counts.get(result.status, 0) + 1
        failure = result.failure if isinstance(result.failure, dict) else None
        if not failure:
            continue
        code = str(failure.get("failure_code") or failure.get("code") or "unknown")
        domain = str(failure.get("failure_domain") or "unknown")
        failure_codes[code] = failure_codes.get(code, 0) + 1
        failure_domains[domain] = failure_domains.get(domain, 0) + 1
    return {
        "status_counts": status_counts,
        "failure_codes": failure_codes,
        "failure_domains": failure_domains,
    }


def _failure_code(result: TrialResult) -> str | None:
    if isinstance(result.failure, dict):
        value = result.failure.get("failure_code") or result.failure.get("code")
        return str(value) if value else None
    return None


def _trial_result_ref(result: TrialResult) -> ArtifactRef:
    for ref in result.artifact_refs:
        if ref.name == "trial_result.json":
            return ref
    path = result.trace_ref.path
    if path.endswith("/trace.jsonl"):
        path = f"{path.rsplit('/', 1)[0]}/trial_result.json"
    else:
        path = f"{path.rsplit('/', 1)[0]}/trial_result.json"
    return ArtifactRef(
        owner="infra",
        name="trial_result.json",
        path=path,
        mime_type="application/json",
        size_bytes=0,
        sha256="0" * 64,
    )


def _lease_descriptor(lease: Any) -> dict[str, Any]:
    if lease is None:
        return {}
    if hasattr(lease, "to_dict"):
        value = lease.to_dict()
    elif isinstance(lease, dict):
        value = dict(lease)
    else:
        value = {
            key: getattr(lease, key)
            for key in ("lease_id", "profile_id", "resource_kind", "lifecycle")
            if hasattr(lease, key)
        }
    return to_json_compatible(value)


def _trial_status(
    scheduler_result: dict[str, Any],
    verifier_result: dict[str, Any],
) -> Literal["completed", "failed", "aborted"]:
    for payload in (verifier_result, scheduler_result):
        status = payload.get("status")
        if status in {"completed", "failed", "aborted"}:
            return status  # type: ignore[return-value]
    return "completed"


def _extract_failure(
    scheduler_result: dict[str, Any],
    verifier_result: dict[str, Any],
) -> dict[str, Any] | None:
    for payload in (verifier_result, scheduler_result):
        failure = payload.get("failure")
        if isinstance(failure, dict):
            return dict(failure)
        failure_code = payload.get("failure_code")
        if failure_code:
            return {"failure_code": str(failure_code), "failure_domain": payload.get("failure_domain") or "unknown"}
    return None


def _trial_execution_failure(error: BaseException) -> dict[str, Any]:
    return {
        "failure_domain": "trial",
        "failure_stage": "execute_trial",
        "failure_code": "trial.execution_failed",
        "summary": str(error),
        "details": {"error_type": error.__class__.__name__},
    }
