from __future__ import annotations

from gage_eval.reporting.summary_generators.appworld import _build_appworld_summary


class _FakeCache:
    def __init__(self, records):
        self._records = list(records)

    def iter_samples(self):
        return list(self._records)


def test_appworld_summary_uses_eval_result_when_judge_output_missing() -> None:
    cache = _FakeCache(
        [
            {
                "sample": {
                    "id": "calendar_001",
                    "metadata": {"appworld": {"task_id": "calendar_001", "subset": "dev"}},
                    "eval_result": {
                        "status": "pass",
                        "score": 1.0,
                        "summary": "tgc=1.0",
                        "appworld": {"tgc": 1.0, "sgc": 0.5},
                    },
                }
            }
        ]
    )

    summary = _build_appworld_summary(cache)

    assert summary is not None
    assert summary["overall"]["total"] == 1
    assert summary["overall"]["tgc_mean"] == 1.0
    assert summary["overall"]["sgc_mean"] == 0.5
    assert summary["by_subset"]["dev"]["total"] == 1

