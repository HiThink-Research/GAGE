from __future__ import annotations

import pytest

from gage_eval.agent_eval_kits.swebench.units import build_swebench_instruction, build_swebench_messages
from gage_eval.agent_eval_kits.terminal_bench.units import build_terminal_instruction, build_terminal_messages


@pytest.mark.fast
def test_terminal_instruction_enforces_done_contract() -> None:
    sample = {"instruction": "Create hello.txt in the workspace."}

    instruction = build_terminal_instruction(sample)
    messages = build_terminal_messages(sample)

    assert "reply with exactly `done`" in instruction
    assert messages[0]["role"] == "system"
    assert "reply with exactly `done`" in str(messages[0]["content"])


@pytest.mark.fast
def test_swebench_instruction_enforces_patch_submission_contract() -> None:
    sample = {"instruction": "Fix the failing test in the repo."}

    instruction = build_swebench_instruction(sample)
    messages = build_swebench_messages(sample)

    assert "submission.patch" in instruction
    assert "submit_patch_tool" in instruction
    assert messages[0]["role"] == "system"
    assert "submission.patch" in str(messages[0]["content"])
