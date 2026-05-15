from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.scenario_profiles import ScenarioProfileBuilder


class _FailingProfile:
    profile_name = "broken"

    def build(self, index):
        raise RuntimeError("boom")


@pytest.mark.fast
def test_scenario_profile_failure_is_degraded() -> None:
    profiles, diagnostics = ScenarioProfileBuilder([_FailingProfile()]).build(index=object())

    assert profiles == {}
    assert diagnostics["warnings"][0]["code"] == "report_pack.scenario_profile_failed"
