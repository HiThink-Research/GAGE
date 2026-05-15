from __future__ import annotations

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
        diagnostics = {"warnings": [], "errors": []}
        for profile in self.profiles:
            name = str(getattr(profile, "profile_name", profile.__class__.__name__))
            try:
                value = profile.build(index)
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
        return outputs, diagnostics

