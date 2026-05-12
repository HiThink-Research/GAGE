from __future__ import annotations

import pytest

from gage_eval.agent_eval_kits.swebench.tools import build_swebench_instruction, build_swebench_messages


@pytest.mark.fast
def test_swebench_instruction_enforces_patch_submission_contract() -> None:
    sample = {"instruction": "Fix the failing test in the repo."}

    instruction = build_swebench_instruction(sample)
    messages = build_swebench_messages(sample)

    assert "submission.patch" in instruction
    assert "submit_patch_tool" in instruction
    assert messages[0]["role"] == "system"
    assert "submission.patch" in str(messages[0]["content"])
