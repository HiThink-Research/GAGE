from __future__ import annotations

from dataclasses import dataclass

from gage_eval.reporting.contracts import AttentionCase, FailureCluster


@dataclass
class FailureClusterResult:
    reason_code_counts: dict[str, dict[str, int]]
    failure_clusters: list[FailureCluster]


class FailureClusterer:
    def cluster(self, attention_cases: list[AttentionCase]) -> FailureClusterResult:
        counts = {"runtime": {}, "system": {}}
        grouped: dict[tuple[str, ...], list[AttentionCase]] = {}
        for case in attention_cases:
            runtime_codes = [code for code in case.reason_codes if not code.startswith("system.")]
            system_codes = [code for code in case.reason_codes if code.startswith("system.")]
            for code in runtime_codes:
                counts["runtime"][code] = counts["runtime"].get(code, 0) + 1
            for code in system_codes:
                counts["system"][code] = counts["system"].get(code, 0) + 1
            key = tuple(runtime_codes or system_codes)
            grouped.setdefault(key, []).append(case)
        clusters = [
            FailureCluster(
                cluster_id="cluster:" + "+".join(key),
                cluster_key=list(key),
                count=len(cases),
                severity=cases[0].severity,
                sample_ids=sorted({case.sample_id or case.case_id or "" for case in cases}),
                representative_ref_ids=list(cases[0].evidence_ref_ids),
                label=", ".join(key),
            )
            for key, cases in grouped.items()
            if key
        ]
        clusters.sort(key=lambda item: (-(item.count or 0), item.cluster_key))
        return FailureClusterResult(reason_code_counts=counts, failure_clusters=clusters)
