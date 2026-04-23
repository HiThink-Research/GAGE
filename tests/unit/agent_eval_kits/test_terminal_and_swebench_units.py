from __future__ import annotations

import pytest

from gage_eval.agent_eval_kits.common import extract_instruction
from gage_eval.agent_eval_kits.swebench.units import build_swebench_instruction, build_swebench_messages
from gage_eval.agent_eval_kits.swebench.runtime import build_swebench_runtime_context
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


@pytest.mark.fast
def test_extract_instruction_supports_inputs_prompt() -> None:
    sample = {"inputs": {"prompt": "Fix the regression."}}

    assert extract_instruction(sample) == "Fix the regression."


@pytest.mark.fast
def test_extract_instruction_supports_message_content_blocks() -> None:
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Fix the bug."},
                    {"type": "text", "text": "Extra context."},
                ],
            }
        ]
    }

    assert extract_instruction(sample) == "Fix the bug.\nExtra context."


@pytest.mark.fast
def test_extract_instruction_prefers_user_over_system_messages() -> None:
    sample = {
        "messages": [
            {"role": "system", "content": "system primer"},
            {"role": "assistant", "content": "assistant primer"},
            {"role": "user", "content": "Fix the bug."},
        ]
    }

    assert extract_instruction(sample) == "Fix the bug."


@pytest.mark.fast
def test_swebench_runtime_context_includes_prompt_metadata() -> None:
    sample = {
        "instruction": "Fix the bug.",
        "metadata": {
            "repo": "django/django",
            "base_commit": "abc123",
            "test_command": "python -m pytest",
            "prompt_source": "problem_statement",
            "prompt_present": True,
        },
    }

    context = build_swebench_runtime_context(sample)

    assert "Fix the bug." in context["instruction"]
    assert context["prompt_present"] is True
    assert context["prompt_source"] == "problem_statement"
    assert context["repo"] == "django/django"
    assert context["base_commit"] == "abc123"
    assert context["test_command"] == "python -m pytest"
