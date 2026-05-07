from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from ._support import build_fake_executor, flatten_samples_projection, write_samples_jsonl


@pytest.mark.io
def test_trials_two_projection_uses_primary_trial_scalar_and_aggregate_recomputes(
    tmp_path: Path,
) -> None:
    executor, _manager, _scheduler, _verifier = build_fake_executor(
        tmp_path,
        run_id="run-projection",
        benchmark_kit_id="swebench",
        trial_policy={"trials": 2},
    )

    output = asyncio.run(
        executor.aexecute(
            sample={"id": "sample-1"},
            payload={
                "execution_context": {
                    "run_id": "run-projection",
                    "task_id": "swebench_pro_smoke",
                    "sample_id": "sample-1",
                }
            },
        )
    )

    aggregate = output["agent_eval"]["trial_aggregate"]
    sample_projection = flatten_samples_projection(aggregate["samples_jsonl_projection"])
    samples_jsonl = write_samples_jsonl(
        tmp_path / "run-projection",
        {
            "sample_id": "sample-1",
            "task_id": "swebench_pro_smoke",
            **sample_projection,
        },
    )

    trial_scores = [
        trial["verifier_result"]["score"]
        for trial in output["agent_eval"]["trial_results"]
        if trial["status"] == "completed"
    ]
    trial_passes = [
        trial["verifier_result"]["resolved"]
        for trial in output["agent_eval"]["trial_results"]
        if trial["status"] == "completed"
    ]

    assert [trial["trial_id"] for trial in output["agent_eval"]["trial_results"]] == [
        "trial_0001",
        "trial_0002",
    ]
    assert aggregate["score_mean"] == sum(trial_scores) / len(trial_scores)
    assert aggregate["pass_rate"] == sum(1 for value in trial_passes if value) / len(trial_passes)
    [record] = [json.loads(line) for line in samples_jsonl.read_text(encoding="utf-8").splitlines()]
    assert record["primary_trial_id"] == "trial_0001"
    assert record["score"] == output["agent_eval"]["trial_results"][0]["verifier_result"]["score"]
    assert record["score_source_trial_id"] == "trial_0001"
    assert record["resolved"] == output["agent_eval"]["trial_results"][0]["verifier_result"]["resolved"]
