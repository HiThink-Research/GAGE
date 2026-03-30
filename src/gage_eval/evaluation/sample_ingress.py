"""Ingress helpers for canonical sample identity and validation summaries."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import Lock
from typing import Any, Callable, Dict, Iterable, Iterator, Mapping, Optional

from gage_eval.assets.datasets.validation import SampleValidator, ValidationFailure
from gage_eval.observability.trace import ObservabilityTrace


@dataclass(frozen=True)
class SampleIngressPolicy:
    """Policy knobs for validation visibility and sample identity resolution."""

    strict_sample_id: bool = False
    max_drop_ratio: Optional[float] = None
    min_valid_samples: Optional[int] = None
    drop_reasons_limit: int = 5


@dataclass(frozen=True)
class ResolvedSampleIdentity:
    """Canonical sample identity resolved at ingress time."""

    sample_id: str
    source: str


class SampleIdResolutionError(ValueError):
    """Raised when canonical sample identity cannot be resolved."""


class SampleValidationGateError(RuntimeError):
    """Raised when validation thresholds reject the current batch."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = str(code)


class ValidationLedger:
    """Accumulate validation totals and compact drop-reason summaries."""

    def __init__(
        self,
        *,
        drop_reasons_limit: int = 5,
        on_update: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self._lock = Lock()
        self._samples_total = 0
        self._samples_valid = 0
        self._samples_dropped = 0
        self._drop_reasons: Counter[str] = Counter()
        self._drop_reasons_limit = max(1, int(drop_reasons_limit))
        self._validation_gate_triggered = False
        self._validation_gate_error_code: Optional[str] = None
        self._validation_gate_error: Optional[str] = None
        self._on_update = on_update

    def update_drop_reasons_limit(self, limit: int) -> None:
        """Keep the largest configured top-k window seen by the current run."""

        with self._lock:
            self._drop_reasons_limit = max(self._drop_reasons_limit, max(1, int(limit)))
        self._publish_update()

    def record_seen(self) -> None:
        """Record one raw sample entering the ingress pipeline."""

        with self._lock:
            self._samples_total += 1
        self._publish_update()

    def record_valid(self) -> None:
        """Record one sample that survived ingress and entered execution."""

        with self._lock:
            self._samples_valid += 1
        self._publish_update()

    def record_failure(self, failure: ValidationFailure | str) -> None:
        """Record one dropped sample and bucket its reason."""

        reason = failure.reason if isinstance(failure, ValidationFailure) else str(failure)
        normalized = reason.strip() or "validation_error"
        with self._lock:
            self._samples_dropped += 1
            self._drop_reasons[normalized] += 1
        self._publish_update()

    def mark_gate_failure(self, *, code: str, message: str) -> None:
        """Persist the first gate failure in the summary payload."""

        with self._lock:
            if not self._validation_gate_triggered:
                self._validation_gate_triggered = True
                self._validation_gate_error_code = str(code)
                self._validation_gate_error = str(message)
        self._publish_update()

    def snapshot(self) -> Dict[str, Any]:
        """Build a JSON-safe summary payload."""

        with self._lock:
            total = self._samples_total
            dropped = self._samples_dropped
            reasons = sorted(
                self._drop_reasons.items(),
                key=lambda item: (-item[1], item[0]),
            )
            top_reasons = [
                {"reason": reason, "count": count}
                for reason, count in reasons[: self._drop_reasons_limit]
            ]
            return {
                "samples_total": total,
                "samples_valid": self._samples_valid,
                "samples_dropped": dropped,
                "samples_drop_ratio": (dropped / total) if total else 0.0,
                "drop_reasons_top": top_reasons,
                "validation_gate_triggered": self._validation_gate_triggered,
                "validation_gate_error_code": self._validation_gate_error_code,
                "validation_gate_error": self._validation_gate_error,
            }

    def _publish_update(self) -> None:
        if self._on_update is not None:
            self._on_update(self.snapshot())


class SampleIdentityResolver:
    """Resolve canonical sample ids once and reuse them everywhere else."""

    def __init__(self, *, strict_sample_id: bool = False) -> None:
        self._strict_sample_id = bool(strict_sample_id)

    def prepare_sample(
        self,
        sample: Mapping[str, Any],
        *,
        dataset_id: str,
        source_index: int,
        task_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return a prepared sample dict with canonical identity fields injected."""

        resolved = self.resolve(
            sample,
            dataset_id=dataset_id,
            source_index=source_index,
            task_id=task_id,
        )
        prepared = dict(sample)
        prepared["id"] = resolved.sample_id
        prepared["sample_id"] = resolved.sample_id
        prepared["_gage_source_index"] = int(source_index)
        prepared["_gage_sample_id_source"] = resolved.source
        prepared["_gage_dataset_id"] = str(dataset_id)
        if task_id:
            prepared["_gage_task_id"] = str(task_id)
        return prepared

    def resolve(
        self,
        sample: Mapping[str, Any],
        *,
        dataset_id: str,
        source_index: int,
        task_id: Optional[str] = None,
    ) -> ResolvedSampleIdentity:
        """Resolve the canonical sample id for one runtime sample."""

        prepared_id = _extract_prepared_sample_id(sample)
        if prepared_id is not None:
            source = str(sample.get("_gage_sample_id_source") or "prepared")
            return ResolvedSampleIdentity(sample_id=prepared_id, source=source)

        explicit_id = _extract_explicit_sample_id(sample)
        if explicit_id is not None:
            return ResolvedSampleIdentity(sample_id=explicit_id, source="explicit")

        if self._strict_sample_id:
            raise SampleIdResolutionError(
                "sample id is required when strict_sample_id=true"
            )

        prefix = _identity_prefix(task_id=task_id, dataset_id=dataset_id)
        return ResolvedSampleIdentity(
            sample_id=f"{prefix}:{int(source_index)}",
            source="synthetic_source_index",
        )


class SampleIngressCoordinator:
    """Coordinate envelope validation, canonical sample ids, and gate checks."""

    def __init__(
        self,
        *,
        dataset_id: str,
        validator: Optional[SampleValidator],
        policy: SampleIngressPolicy,
        trace: Optional[ObservabilityTrace] = None,
        task_id: Optional[str] = None,
        aggregate_ledger: Optional[ValidationLedger] = None,
    ) -> None:
        self._dataset_id = str(dataset_id)
        self._validator = validator
        self._policy = policy
        self._trace = trace
        self._task_id = str(task_id) if task_id else None
        self._resolver = SampleIdentityResolver(
            strict_sample_id=policy.strict_sample_id
        )
        self._local_ledger = ValidationLedger(
            drop_reasons_limit=policy.drop_reasons_limit
        )
        self._aggregate_ledger = aggregate_ledger
        if self._aggregate_ledger is not None:
            self._aggregate_ledger.update_drop_reasons_limit(policy.drop_reasons_limit)

    @property
    def requires_eager_gate_check(self) -> bool:
        """Return whether end-of-stream gates require eager materialization."""

        return (
            self._policy.max_drop_ratio is not None
            or self._policy.min_valid_samples is not None
        )

    def record_seen(self, source_index: int) -> None:  # noqa: ARG002
        """Count one raw sample entering ingress."""

        self._local_ledger.record_seen()
        if self._aggregate_ledger is not None:
            self._aggregate_ledger.record_seen()

    def record_failure(self, failure: ValidationFailure) -> None:
        """Count one dropped sample in both local and aggregate ledgers."""

        self._local_ledger.record_failure(failure)
        if self._aggregate_ledger is not None:
            self._aggregate_ledger.record_failure(failure)

    def prepare(self, raw_samples: Iterable[Dict[str, Any]]) -> Iterator[Dict[str, Any]]:
        """Yield prepared samples that are ready for TaskPlanner and SampleLoop."""

        for fallback_index, record in enumerate(raw_samples):
            sample = dict(record)
            source_index = _coerce_source_index(
                sample.get("_gage_source_index"),
                default=fallback_index,
            )
            sample_id_hint = resolve_runtime_sample_id(sample, task_id=self._task_id)
            if self._validator is not None:
                validated = self._validator.validate_envelope(
                    sample,
                    dataset_id=self._dataset_id,
                    sample_id=sample_id_hint,
                    trace=self._trace,
                    on_failure=self.record_failure,
                )
                if validated is None:
                    continue
                sample = dict(validated)
            try:
                prepared = self._resolver.prepare_sample(
                    sample,
                    dataset_id=self._dataset_id,
                    source_index=source_index,
                    task_id=self._task_id,
                )
            except SampleIdResolutionError as exc:
                message = (
                    f"{exc} dataset_id={self._dataset_id} "
                    f"task_id={self._task_id} source_index={source_index}"
                )
                self._mark_gate_failure("missing_sample_id", message)
                raise SampleValidationGateError("missing_sample_id", message) from exc
            self._local_ledger.record_valid()
            if self._aggregate_ledger is not None:
                self._aggregate_ledger.record_valid()
            yield prepared

        self._raise_if_needed()

    def _raise_if_needed(self) -> None:
        snapshot = self._local_ledger.snapshot()
        drop_ratio = float(snapshot["samples_drop_ratio"])
        if (
            self._policy.max_drop_ratio is not None
            and snapshot["samples_total"] > 0
            and drop_ratio > float(self._policy.max_drop_ratio)
        ):
            message = (
                f"samples drop ratio {drop_ratio:.4f} exceeded "
                f"max_drop_ratio={self._policy.max_drop_ratio:.4f}"
            )
            self._mark_gate_failure("drop_ratio_exceeded", message)
            raise SampleValidationGateError("drop_ratio_exceeded", message)

        if (
            self._policy.min_valid_samples is not None
            and snapshot["samples_valid"] < int(self._policy.min_valid_samples)
        ):
            message = (
                f"samples valid {snapshot['samples_valid']} fell below "
                f"min_valid_samples={int(self._policy.min_valid_samples)}"
            )
            self._mark_gate_failure("min_valid_samples_violation", message)
            raise SampleValidationGateError("min_valid_samples_violation", message)

    def _mark_gate_failure(self, code: str, message: str) -> None:
        self._local_ledger.mark_gate_failure(code=code, message=message)
        if self._aggregate_ledger is not None:
            self._aggregate_ledger.mark_gate_failure(code=code, message=message)


def build_sample_ingress_policy(
    config: Optional[Dict[str, Any]],
) -> SampleIngressPolicy:
    """Build ingress policy from dataset validation config."""

    config = dict(config or {})
    max_drop_ratio = config.get("max_drop_ratio")
    min_valid_samples = config.get("min_valid_samples")
    return SampleIngressPolicy(
        strict_sample_id=bool(config.get("strict_sample_id", False)),
        max_drop_ratio=(
            float(max_drop_ratio) if max_drop_ratio is not None else None
        ),
        min_valid_samples=(
            int(min_valid_samples) if min_valid_samples is not None else None
        ),
        drop_reasons_limit=max(1, int(config.get("drop_reasons_limit", 5))),
    )


def resolve_runtime_sample_id(
    sample: Mapping[str, Any],
    *,
    task_id: Optional[str] = None,
    logical_idx: Optional[int] = None,
) -> str:
    """Resolve the best available sample id for runtime consumers."""

    prepared_id = _extract_prepared_sample_id(sample)
    if prepared_id is not None:
        return prepared_id

    explicit_id = _extract_explicit_sample_id(sample)
    if explicit_id is not None:
        return explicit_id

    source_index = sample.get("_gage_source_index")
    if source_index is not None:
        return f"{_identity_prefix(task_id=task_id or sample.get('_gage_task_id'), dataset_id=sample.get('_gage_dataset_id'))}:{_coerce_source_index(source_index, default=0)}"

    if logical_idx is not None:
        return f"{_identity_prefix(task_id=task_id or sample.get('_gage_task_id'), dataset_id=sample.get('_gage_dataset_id'))}:{int(logical_idx)}"

    return "sample"


def _extract_prepared_sample_id(sample: Mapping[str, Any]) -> Optional[str]:
    if sample.get("_gage_sample_id_source"):
        current = _normalize_string(sample.get("sample_id")) or _normalize_string(
            sample.get("id")
        )
        if current is not None:
            return current
    return None


def _extract_explicit_sample_id(sample: Mapping[str, Any]) -> Optional[str]:
    for key in ("id", "sample_id", "uid", "_id"):
        value = _normalize_string(sample.get(key))
        if value is not None:
            return value
    return None


def _normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    if normalized.lower() in {"none", "null"}:
        return None
    return normalized


def _identity_prefix(task_id: Any, dataset_id: Any) -> str:
    if task_id not in (None, ""):
        return str(task_id)
    if dataset_id not in (None, ""):
        return str(dataset_id)
    return "sample"


def _coerce_source_index(value: Any, *, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)
