from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SampleEvidenceManifest:
    """Stable sample-level evidence reference manifest."""

    sample_key: str
    task_id: str | None = None
    sample_id: str | None = None
    root_journal_ref_id: str | None = None
    namespace_detail_ref_ids: list[str] = field(default_factory=list)
    artifact_ref_ids: list[str] = field(default_factory=list)
    trial_ref_ids: list[str] = field(default_factory=list)
    aggregate_ref_ids: list[str] = field(default_factory=list)
    redaction_status: str = "unknown"
    diagnostics: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_key": self.sample_key,
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "root_journal_ref_id": self.root_journal_ref_id,
            "namespace_detail_ref_ids": sorted(self.namespace_detail_ref_ids),
            "artifact_ref_ids": sorted(self.artifact_ref_ids),
            "trial_ref_ids": sorted(self.trial_ref_ids),
            "aggregate_ref_ids": sorted(self.aggregate_ref_ids),
            "redaction_status": self.redaction_status,
            "diagnostics": list(self.diagnostics),
        }
