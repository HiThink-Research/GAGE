from __future__ import annotations

from typing import Any

from gage_eval.agent_runtime.contracts.failure import FailureEnvelope, FailureEnvelopeError


class FailureMapper:
    """Maps raw exceptions into stable runtime failure envelopes."""

    def map_exception(
        self,
        exc: BaseException,
        *,
        failure_domain: str,
        failure_stage: str,
        component_kind: str,
        component_id: str,
        owner: str,
        failure_code: str,
        first_bad_step: str,
        suspect_files: tuple[str, ...],
        retryable: bool = False,
        reproduction_hint: str | None = None,
        details: dict[str, Any] | None = None,
    ) -> FailureEnvelope:
        """Normalize one exception without mutating the original error."""

        if isinstance(exc, FailureEnvelopeError):
            return exc.failure
        return FailureEnvelope(
            failure_domain=failure_domain,  # type: ignore[arg-type]
            failure_stage=failure_stage,  # type: ignore[arg-type]
            failure_code=failure_code,
            component_kind=component_kind,  # type: ignore[arg-type]
            component_id=component_id,
            owner=owner,
            retryable=retryable,
            summary=str(exc) or exc.__class__.__name__,
            first_bad_step=first_bad_step,
            suspect_files=tuple(suspect_files[:3]),
            reproduction_hint=reproduction_hint,
            normalized_signals={"exception_type": exc.__class__.__name__},
            details=details or {},
        )

    @staticmethod
    def raise_failure(failure: FailureEnvelope) -> None:
        """Raise an already-normalized failure for internal propagation."""

        raise FailureEnvelopeError(failure)
