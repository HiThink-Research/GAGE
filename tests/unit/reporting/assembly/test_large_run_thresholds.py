from __future__ import annotations

import pytest

from gage_eval.reporting.assembly.attention_detector import AttentionCaseDetector


@pytest.mark.fast
def test_attention_detector_respects_global_top_k() -> None:
    candidates = [
        {"case_id": f"case-{idx}", "sample_id": f"s{idx}", "reason_codes": ["score.low"], "summary": "low"}
        for idx in range(20)
    ]

    cases = AttentionCaseDetector(top_k=5).detect(candidates, total_samples=20)

    assert len(cases) == 5
