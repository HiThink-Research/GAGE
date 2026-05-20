from __future__ import annotations

from typing import Any


class ExternalHarnessScenarioProfile:
    profile_name = "external_harness"

    def build(self, index: Any) -> dict[str, Any]:
        rollup: dict[str, int] = {}
        harnesses: dict[str, dict[str, Any]] = {}
        trial_count = 0
        for sample in getattr(index, "samples", []) or []:
            harness_id = _harness_id(sample)
            if harness_id:
                harnesses.setdefault(harness_id, {"harness_id": harness_id, "sample_count": 0, "trial_count": 0})
                harnesses[harness_id]["sample_count"] += 1
            for trial in sample.get("trial_results", []) or []:
                trial_count += 1
                if harness_id:
                    harnesses[harness_id]["trial_count"] += 1
                status = str(trial.get("status") or "unknown")
                rollup[status] = rollup.get(status, 0) + 1
        return {
            "profile_version": "gage.scenario.external_harness.v1",
            "harnesses": [harnesses[key] for key in sorted(harnesses)],
            "trial_count": trial_count,
            "trial_rollup": dict(sorted(rollup.items())),
        }


def _harness_id(sample: dict[str, Any]) -> str | None:
    sample_payload = sample.get("sample") if isinstance(sample.get("sample"), dict) else {}
    task_type = str(sample_payload.get("task_type") or sample.get("task_type") or "")
    if task_type.startswith("external_harness."):
        return task_type.split(".", 1)[1]
    metadata = sample_payload.get("metadata") if isinstance(sample_payload.get("metadata"), dict) else {}
    harness = metadata.get("_harness") if isinstance(metadata.get("_harness"), dict) else {}
    kit_id = harness.get("kit_id")
    return str(kit_id) if kit_id else None
