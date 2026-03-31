from __future__ import annotations

from gage_eval.reporting.summary_generators.swebench import _build_swebench_summary


class _FakeCache:
    def __init__(self, records):
        self._records = list(records)

    def iter_samples(self):
        return list(self._records)


def test_swebench_summary_uses_eval_result_when_judge_output_missing() -> None:
    cache = _FakeCache(
        [
            {
                "sample": {
                    "id": "swebench__1",
                    "metadata": {
                        "instance_id": "swebench__1",
                        "repo": "local/swebench-smoke",
                    },
                    "eval_result": {
                        "status": "pass",
                        "score": 1.0,
                        "summary": "resolved",
                        "raw_output": {
                            "resolved": True,
                            "failure_reason": None,
                        },
                    },
                },
            }
        ]
    )

    summary = _build_swebench_summary(cache)

    assert summary is not None
    assert summary["overall"]["total"] == 1
    assert summary["overall"]["resolved"] == 1
    assert summary["overall"]["resolve_rate"] == 1.0
    assert summary["failure_reason"] == {}
