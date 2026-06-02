from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from gage_eval.reporting.assembly.scenario_profiles import ProfileRefResolver


class AgentScenarioProfile:
    profile_name = "agent"

    def build(self, index: Any, *, ref_resolver: ProfileRefResolver | None = None) -> dict[str, Any]:
        resolver = ref_resolver or ProfileRefResolver.from_index(index)
        trial_count = 0
        failed_trial_count = 0
        representative_ref_ids: list[str] = []
        for sample in getattr(index, "samples", []) or []:
            record = _load_sample_record(index, sample)
            trials = list(record.get("trial_results") or _sample_trial_results(sample))
            trial_count += len(trials)
            failed_trial_count += sum(1 for trial in trials if trial.get("status") == "failed")
            sample_ref_ids = _artifact_ref_ids(sample, "sample_record.json", resolver)
            if not sample_ref_ids:
                sample_ref_ids = _nested_artifact_ref_ids(sample, "trial_result.json", resolver)
            representative_ref_ids.extend(sample_ref_ids)
        return {
            "profile_version": "gage.scenario.agent.v1",
            "trial_count": trial_count,
            "failed_trial_count": failed_trial_count,
            "representative_ref_ids": sorted(set(representative_ref_ids)),
        }


def _load_sample_record(index: Any, sample: dict[str, Any]) -> dict[str, Any]:
    for artifact in sample.get("artifact_refs", []) or []:
        if artifact.get("name") == "sample_record.json" or str(artifact.get("path", "")).endswith("sample_record.json"):
            path = Path(getattr(index, "run_dir", "")) / str(artifact.get("path"))
            if path.exists():
                return json.loads(path.read_text(encoding="utf-8"))
    return {}


def _artifact_ref_ids(sample: dict[str, Any], suffix: str, resolver: ProfileRefResolver) -> list[str]:
    ref_ids: list[str] = []
    for artifact in sample.get("artifact_refs", []) or []:
        path = artifact.get("path") if isinstance(artifact, dict) else None
        if str(path or "").endswith(suffix):
            ref_id = resolver.resolve(path, profile="agent", field="representative_ref_ids")
            if ref_id:
                ref_ids.append(ref_id)
    return ref_ids


def _sample_trial_results(sample: dict[str, Any]) -> list[dict[str, Any]]:
    trial_results = sample.get("trial_results")
    if isinstance(trial_results, list) and trial_results:
        return [trial for trial in trial_results if isinstance(trial, dict)]
    nested = _nested(sample, "model_output", "agent_eval", "trial_results")
    if isinstance(nested, list):
        return [trial for trial in nested if isinstance(trial, dict)]
    return []


def _nested_artifact_ref_ids(sample: dict[str, Any], suffix: str, resolver: ProfileRefResolver) -> list[str]:
    ref_ids: list[str] = []
    for trial in _sample_trial_results(sample):
        refs = trial.get("artifact_refs")
        if isinstance(refs, list):
            for artifact in refs:
                path = artifact.get("path") if isinstance(artifact, dict) else None
                if str(path or "").endswith(suffix):
                    ref_id = resolver.resolve(path, profile="agent", field="representative_ref_ids")
                    if ref_id:
                        ref_ids.append(ref_id)
        trace_ref = trial.get("trace_ref")
        if isinstance(trace_ref, dict) and str(trace_ref.get("path", "")).endswith(suffix):
            ref_id = resolver.resolve(trace_ref.get("path"), profile="agent", field="representative_ref_ids")
            if ref_id:
                ref_ids.append(ref_id)
    return ref_ids


def _nested(value: dict[str, Any], *keys: str) -> Any:
    current: Any = value
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current
