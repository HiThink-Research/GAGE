from __future__ import annotations

from pathlib import Path

import pytest


OBSERVABILITY_ENV_NAMES = {
    "GAGE_EVAL_OBSERVABILITY",
    "GAGE_EVAL_INMEMORY_TRACE",
    "GAGE_EVAL_TRACE_BUFFER_MAX_EVENTS",
    "GAGE_EVAL_ENABLE_LOG_SINK",
    "GAGE_EVAL_LOG_SINK_LEVEL",
    "GAGE_EVAL_LOG_SINK_MAX_QUEUE",
    "GAGE_EVAL_LOG_SINK_FLUSH_INTERVAL_S",
    "GAGE_EVAL_LOG_SINK_BATCH_SIZE",
    "GAGE_EVAL_LOG_SINK_ZOMBIE_ROUTE_TTL_S",
    "GAGE_EVAL_LOG_SINK_ROUTE_SWEEP_INTERVAL_S",
    "GAGE_EVAL_LOG_SINK_CLOSE_MODE",
    "GAGE_EVAL_LOG_SINK_DRAIN_TIMEOUT_S",
    "GAGE_EVAL_REPORT_HTTP_URL",
    "GAGE_EVAL_REPORT_HTTP_BATCH",
    "GAGE_EVAL_REPORT_HTTP_FAIL_PCT",
    "GAGE_EVAL_REPORT_HTTP_TIMEOUT",
    "GAGE_EVAL_REPORT_HTTP_MAX_RETRIES",
    "GAGE_EVAL_REPORT_HTTP_RETRY_BASE_MS",
    "GAGE_EVAL_REPORT_HTTP_RETRY_MAX_MS",
    "GAGE_EVAL_REPORT_HTTP_RETRY_MULTIPLIER",
}


@pytest.mark.fast
def test_observability_env_reads_are_centralized_in_config() -> None:
    observability_dir = Path("src/gage_eval/observability")
    offenders: list[str] = []
    for path in observability_dir.glob("*.py"):
        if path.name == "config.py":
            continue
        source = path.read_text(encoding="utf-8")
        for env_name in sorted(OBSERVABILITY_ENV_NAMES):
            if env_name in source:
                offenders.append(f"{path}:{env_name}")

    assert offenders == []
