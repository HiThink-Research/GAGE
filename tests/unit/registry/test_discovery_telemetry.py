from __future__ import annotations

import pytest

from gage_eval.registry import discovery_telemetry


@pytest.mark.fast
def test_discovery_telemetry_tracks_global_and_kind_scoped_counters() -> None:
    discovery_telemetry.reset()
    discovery_telemetry.record("manifest_hit", kind="metrics")
    discovery_telemetry.record("manifest_hit", kind="metrics")
    discovery_telemetry.record("manifest_missing", kind="roles")

    snapshot = discovery_telemetry.snapshot()

    assert snapshot.counters["manifest_hit"] == 2
    assert snapshot.counters["manifest_missing"] == 1
    assert snapshot.by_kind["metrics"]["manifest_hit"] == 2
    assert snapshot.by_kind["roles"]["manifest_missing"] == 1
