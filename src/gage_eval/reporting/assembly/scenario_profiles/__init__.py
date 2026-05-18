from __future__ import annotations

import inspect
from typing import Any


class ScenarioProfileBuilder:
    def __init__(self, profiles: list[Any] | None = None) -> None:
        if profiles is None:
            from gage_eval.reporting.assembly.scenario_profiles.agent import AgentScenarioProfile
            from gage_eval.reporting.assembly.scenario_profiles.external_harness import ExternalHarnessScenarioProfile
            from gage_eval.reporting.assembly.scenario_profiles.game import GameScenarioProfile

            profiles = [AgentScenarioProfile(), ExternalHarnessScenarioProfile(), GameScenarioProfile()]
        self.profiles = profiles

    def build(self, index: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        outputs: dict[str, Any] = {}
        diagnostics = {"warnings": [], "errors": [], "profile_ref_resolution_miss_count": 0}
        ref_resolver = ProfileRefResolver.from_index(index, diagnostics=diagnostics)
        for profile in self.profiles:
            name = str(getattr(profile, "profile_name", profile.__class__.__name__))
            try:
                value = _build_profile(profile, index, ref_resolver)
            except Exception as exc:  # pragma: no cover - profile-owned
                diagnostics["warnings"].append(
                    {
                        "code": "report_pack.scenario_profile_failed",
                        "profile": name,
                        "message": str(exc),
                    }
                )
                continue
            if value:
                outputs[name] = value
        diagnostics["profile_ref_resolution_miss_count"] = ref_resolver.miss_count
        return outputs, diagnostics


class ProfileRefResolver:
    """Resolves artifact source paths to canonical EvidenceRef ids."""

    def __init__(
        self,
        path_to_ref_id: dict[str, str],
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        self._path_to_ref_id = path_to_ref_id
        self._diagnostics = diagnostics
        self.miss_count = 0

    @classmethod
    def from_index(
        cls,
        index: Any,
        *,
        diagnostics: dict[str, Any] | None = None,
    ) -> "ProfileRefResolver":
        refs = getattr(index, "evidence_refs", {}) or {}
        values = refs.values() if isinstance(refs, dict) else refs
        path_to_ref_id: dict[str, str] = {}
        for ref in values or []:
            path = _get_field(ref, "path")
            ref_id = _get_field(ref, "ref_id")
            if path and ref_id:
                path_to_ref_id[str(path)] = str(ref_id)
        return cls(path_to_ref_id, diagnostics=diagnostics)

    def resolve(self, path: Any, *, profile: str, field: str) -> str | None:
        source_path = str(path or "").strip()
        if not source_path:
            return None
        ref_id = self._path_to_ref_id.get(source_path)
        if ref_id:
            return ref_id
        self.miss_count += 1
        if self._diagnostics is not None:
            self._diagnostics.setdefault("warnings", []).append(
                {
                    "code": "report_pack.profile_ref_resolution_miss",
                    "profile": profile,
                    "field": field,
                    "path": source_path,
                }
            )
            self._diagnostics["profile_ref_resolution_miss_count"] = self.miss_count
        return None


def _build_profile(profile: Any, index: Any, ref_resolver: ProfileRefResolver) -> dict[str, Any]:
    parameters = inspect.signature(profile.build).parameters
    if "ref_resolver" in parameters:
        return profile.build(index, ref_resolver=ref_resolver)
    return profile.build(index)


def _get_field(value: Any, field_name: str) -> Any:
    if isinstance(value, dict):
        return value.get(field_name)
    return getattr(value, field_name, None)
