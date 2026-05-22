from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ExternalHarnessJob:
    harness_id: str
    job_id: str | None = None
    status: str | None = None
    raw_ref_ids: list[str] = field(default_factory=list)
    aggregate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "harness_id": self.harness_id,
            "job_id": self.job_id,
            "status": self.status,
            "raw_ref_ids": sorted(self.raw_ref_ids),
            "aggregate": dict(self.aggregate),
        }


@dataclass
class ExternalHarnessTrial:
    trial_id: str
    status: str | None = None
    raw_ref_ids: list[str] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    failure: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "trial_id": self.trial_id,
            "status": self.status,
            "raw_ref_ids": sorted(self.raw_ref_ids),
            "metrics": dict(self.metrics),
        }
        if self.failure is not None:
            payload["failure"] = dict(self.failure)
        return payload


class ExternalHarnessAdapter(ABC):
    harness_id: str
    adapter_id: str

    @abstractmethod
    def detect(self, evidence: Any) -> bool:
        raise NotImplementedError

    @abstractmethod
    def normalize_job(self, evidence: Any) -> ExternalHarnessJob:
        raise NotImplementedError

    @abstractmethod
    def normalize_trials(self, evidence: Any) -> list[ExternalHarnessTrial]:
        raise NotImplementedError

    @abstractmethod
    def project_metrics(self, evidence: Any) -> list[dict[str, Any]]:
        raise NotImplementedError
