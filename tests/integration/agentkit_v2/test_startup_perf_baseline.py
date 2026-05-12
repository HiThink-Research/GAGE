from __future__ import annotations

from pathlib import Path

import pytest

from ._support import REPO_ROOT


EXPECTED_COLUMNS = [
    "provider_profile",
    "marker",
    "sample_count",
    "startup_count",
    "p50_s",
    "p95_s",
    "cold_or_warm",
    "credential_required",
    "observed_at",
    "notes",
]


@pytest.mark.io
def test_agentkit_v2_smoke_records_environment_startup_baseline() -> None:
    path = REPO_ROOT.parent / "agent-eval/0428/perf-baseline-2026-04-28.md"
    table = _read_markdown_table(path)

    assert list(table[0]) == EXPECTED_COLUMNS
    tau2_rows = [row for row in table if row["provider_profile"] == "local_process/tau2-local-process"]
    docker_rows = [row for row in table if row["provider_profile"] == "docker/swebench-docker-dind"]
    e2b_rows = [row for row in table if row["provider_profile"] == "e2b/swebench-e2b-wrapper"]
    assert len(tau2_rows) == 1
    assert len(docker_rows) == 2
    assert len(e2b_rows) == 1
    assert tau2_rows[0]["cold_or_warm"] == "warm"
    assert {row["cold_or_warm"] for row in docker_rows} == {"warm", "cold"}
    assert e2b_rows[0]["credential_required"] == "true"
    assert "E2B_API_KEY" in e2b_rows[0]["notes"]


def _read_markdown_table(path: Path) -> list[dict[str, str]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("|")]
    header = [cell.strip(" `") for cell in lines[0].strip("|").split("|")]
    rows: list[dict[str, str]] = []
    for line in lines[2:]:
        values = [cell.strip(" `") for cell in line.strip("|").split("|")]
        rows.append(dict(zip(header, values)))
    return rows
