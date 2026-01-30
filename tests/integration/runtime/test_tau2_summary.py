from __future__ import annotations

from pathlib import Path

from gage_eval.evaluation.cache import EvalCache
from gage_eval.reporting.summary_generators.tau2 import Tau2SummaryGenerator


def _make_sample(task_id: str, domain: str) -> dict:
    return {
        "id": f"{task_id}_sample",
        "metadata": {"tau2": {"task_id": task_id, "domain": domain}},
    }


def test_tau2_summary_pass_hat(tmp_path: Path) -> None:
    cache = EvalCache(base_dir=str(tmp_path), run_id="tau2-summary")
    # task1: 2/2 success; task2: 1/2 success
    samples = [
        (_make_sample("task1", "airline"), 1.0),
        (_make_sample("task1", "airline"), 1.0),
        (_make_sample("task2", "airline"), 1.0),
        (_make_sample("task2", "airline"), 0.0),
    ]
    for idx, (sample, reward) in enumerate(samples):
        cache.write_sample(
            f"sample-{idx}",
            {"sample": sample, "judge_output": {"tau2": {"reward": reward}}},
        )

    summary = Tau2SummaryGenerator().generate(cache)

    assert summary is not None
    pass_hat = summary["tau2_summary"]["pass_hat_k"]
    assert pass_hat[1] == 0.75
    assert pass_hat[2] == 0.5
