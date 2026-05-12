from __future__ import annotations

import asyncio

from gage_eval.agent_eval_kits.swebench.judge.patch_extraction import (
    PatchResolution,
    clean_patch_content,
    resolve_patch,
)


DIFF = (
    "diff --git a/foo.txt b/foo.txt\n"
    "index 1111111..2222222 100644\n"
    "--- a/foo.txt\n"
    "+++ b/foo.txt\n"
    "@@ -1,1 +1,1 @@\n"
    "-old\n"
    "+new\n"
)


class FakeEnvironment:
    def __init__(self, files: dict[str, bytes] | None = None, git_diff: str = "") -> None:
        self.files = files or {}
        self.git_diff = git_diff
        self.commands: list[str] = []

    async def read_file(self, path: str, *, max_bytes: int = 16 * 1024 * 1024) -> bytes:
        del max_bytes
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]

    async def exec(self, command: str, **kwargs):
        del kwargs
        self.commands.append(command)
        return type(
            "ExecResult",
            (),
            {"exit_code": 0, "stdout": self.git_diff, "stderr": "", "timed_out": False},
        )()


class FakeTrace:
    def __init__(self) -> None:
        self.events: list[tuple[str, dict, str | None]] = []

    def emit(self, event: str, payload: dict, sample_id: str | None = None) -> None:
        self.events.append((event, payload, sample_id))


def test_resolve_patch_prefers_model_output_then_submission_patch_then_git_diff() -> None:
    trace = FakeTrace()
    env = FakeEnvironment(files={"/workspace/submission.patch": b"diff --git a/sub b/sub\n"})

    model_result = asyncio.run(
        resolve_patch(
            model_output={"answer": DIFF},
            sample={"id": "sample-1"},
            environment=env,
            trace=trace,
        )
    )
    assert model_result == PatchResolution(source="model_output", patch=DIFF, failure_code=None)
    assert trace.events[-1][0] == "patch.source.resolved"
    assert trace.events[-1][1]["source"] == "model_output"

    submission_result = asyncio.run(
        resolve_patch(
            model_output={},
            sample={"id": "sample-1"},
            environment=env,
            trace=trace,
        )
    )
    assert submission_result.source == "workspace_submission_patch"
    assert submission_result.patch == "diff --git a/sub b/sub\n"
    assert trace.events[-1][0] == "swebench_patch_fallback"
    assert trace.events[-1][1]["source"] == "workspace_submission_patch"
    assert any(
        event == "patch.source.resolved" and payload["source"] == "workspace_submission_patch"
        for event, payload, _sample_id in trace.events
    )

    git_env = FakeEnvironment(git_diff="diff --git a/git b/git\n")
    git_result = asyncio.run(
        resolve_patch(model_output={}, sample={"id": "sample-1"}, environment=git_env, trace=trace)
    )
    assert git_result.source == "git_diff_fallback"
    assert git_result.patch == "diff --git a/git b/git\n"
    assert any("git diff" in command for command in git_env.commands)
    assert trace.events[-1][0] == "swebench_patch_fallback"
    assert trace.events[-1][1]["source"] == "git_diff_fallback"


def test_resolve_patch_extracts_submit_patch_tool_stdout_from_trace() -> None:
    result = asyncio.run(
        resolve_patch(
            model_output={
                "agent_trace": [
                    {"name": "run_shell", "output": {"stdout": "no patch"}},
                    {"name": "submit_patch_tool", "output": {"output": {"stdout": DIFF}}},
                ]
            },
            sample={"id": "sample-1"},
            environment=FakeEnvironment(),
            trace=None,
        )
    )

    assert result.source == "submit_patch_stdout"
    assert result.patch == DIFF


def test_resolve_patch_accepts_carried_patch_content_from_scheduler_environment() -> None:
    result = asyncio.run(
        resolve_patch(
            model_output={"patch_content": DIFF},
            sample={"id": "sample-1"},
            environment=FakeEnvironment(),
            trace=None,
        )
    )

    assert result == PatchResolution(source="model_output", patch=DIFF, failure_code=None)


def test_missing_patch_returns_stable_failure_code() -> None:
    result = asyncio.run(
        resolve_patch(
            model_output={},
            sample={"id": "sample-1"},
            environment=FakeEnvironment(),
            trace=None,
        )
    )

    assert result.patch == ""
    assert result.failure_code == "artifact_capture.patch_missing"


def test_clean_patch_content_strips_code_block_apply_patch_binary_hunks_and_tail() -> None:
    raw = (
        "```diff\n"
        "*** Begin Patch\n"
        f"{DIFF}"
        "diff --git a/image.bin b/image.bin\n"
        "GIT binary patch\n"
        "literal 4\n"
        "abcd\n"
        "*** End Patch\n"
        "Please apply this patch.\n"
        "```\n"
    )

    cleaned = clean_patch_content(raw)

    assert cleaned == DIFF
    assert "*** Begin Patch" not in cleaned
    assert "GIT binary patch" not in cleaned
    assert "Please apply" not in cleaned


def test_clean_patch_content_rejects_natural_language_without_diff_markers() -> None:
    assert clean_patch_content("Now I understand the structure and will update the file.") == ""
    assert clean_patch_content("The fix is to add a try-catch around this call.") == ""
    assert clean_patch_content(f"```\n{DIFF}```").strip().startswith("diff --git")
