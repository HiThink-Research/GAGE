from __future__ import annotations

from gage_eval.reporting.summary_generators.tau2 import _build_tau2_summary


class _FakeCache:
    def __init__(self, records):
        self._records = list(records)

    def iter_samples(self):
        return list(self._records)


def test_tau2_summary_uses_eval_result_when_judge_output_missing() -> None:
    cache = _FakeCache(
        [
            {
                "sample": {
                    "id": "airline_task-1__trial_0",
                    "metadata": {"tau2": {"task_id": "task-1", "domain": "airline"}},
                    "eval_result": {
                        "status": "pass",
                        "score": 1.0,
                        "summary": "reward=1.0",
                        "tau2": {
                            "task_id": "task-1",
                            "domain": "airline",
                            "reward": 1.0,
                            "agent_cost": 0.2,
                            "user_cost": 0.1,
                        },
                    },
                }
            }
        ]
    )

    summary = _build_tau2_summary(cache)

    assert summary is not None
    assert summary["overall"]["total"] == 1
    assert summary["overall"]["avg_reward"] == 1.0
    assert summary["overall"]["avg_agent_cost"] == 0.2
    assert summary["by_domain"]["airline"]["avg_reward"] == 1.0
