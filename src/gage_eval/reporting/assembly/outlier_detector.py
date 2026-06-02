from __future__ import annotations

from typing import Any

from gage_eval.reporting.contracts import OutlierEntry, OutlierGroup


class OutlierDetector:
    def __init__(self, top_k: int = 10, metric_ids: tuple[str, ...] = ("latency_s", "total_tokens")) -> None:
        self.top_k = top_k
        self.metric_ids = metric_ids

    def detect(self, records: list[dict[str, Any]]) -> list[OutlierGroup]:
        groups: list[OutlierGroup] = []
        for metric_id in self.metric_ids:
            entries = [
                OutlierEntry(
                    sample_id=str(record.get("sample_id") or record.get("id")),
                    trial_id=record.get("trial_id"),
                    value=record.get(metric_id),
                    evidence_ref_ids=list(record.get("evidence_ref_ids") or []),
                )
                for record in records
                if isinstance(record.get(metric_id), (int, float))
            ]
            if not entries:
                continue
            entries.sort(key=lambda item: (-(item.value or 0), item.sample_id or ""))
            top = entries[: self.top_k]
            for rank, item in enumerate(top, start=1):
                item.p_rank = rank / len(entries)
            groups.append(OutlierGroup(metric_id=metric_id, scope="run", ranking="descending", top_k=top))
        return groups
