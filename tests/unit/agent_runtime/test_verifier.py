from __future__ import annotations

from gage_eval.agent_runtime.verifier.base import VerifierInput, VerifierResult
from gage_eval.agent_runtime.verifier.judge_adapter import JudgeVerifierAdapter


class _FakeJudge:
    def __init__(self, response: dict[str, object]) -> None:
        self.response = response
        self.payloads: list[dict[str, object]] = []

    def invoke(self, payload):
        self.payloads.append(dict(payload))
        return dict(self.response)


def test_verifier_input_construction() -> None:
    verifier_input = VerifierInput(
        benchmark_kit_id="swebench",
        sample_id="sample-1",
        payload={"sample": {"id": "sample-1"}},
        artifact_paths={"patch_file": "/tmp/submission.patch"},
        metadata={"repo": "example/repo"},
    )

    assert verifier_input.benchmark_kit_id == "swebench"
    assert verifier_input.sample_id == "sample-1"
    assert verifier_input.artifact_paths["patch_file"] == "/tmp/submission.patch"


def test_verifier_result_construction() -> None:
    verifier_result = VerifierResult(status="pass", score=1.0, summary="ok")

    assert verifier_result.status == "pass"
    assert verifier_result.score == 1.0
    assert verifier_result.summary == "ok"


def test_judge_verifier_adapter_is_importable() -> None:
    judge = _FakeJudge({"resolved": True, "tests": []})
    adapter = JudgeVerifierAdapter(judge=judge)
    result = adapter.verify(
        VerifierInput(
            benchmark_kit_id="swebench",
            sample_id="sample-1",
            payload={"sample": {"id": "sample-1", "metadata": {}}},
        )
    )

    assert result.status == "pass"
    assert result.score == 1.0
    assert judge.payloads


def test_patch_presence_verifier_accepts_expected_terms() -> None:
    adapter = JudgeVerifierAdapter(mode="patch_presence")

    result = adapter.verify(
        VerifierInput(
            benchmark_kit_id="swebench",
            sample_id="sample-1",
            payload={
                "sample": {
                    "id": "sample-1",
                    "metadata": {"expected_patch_contains": ["return 42"]},
                },
                "scheduler_result": {
                    "raw_output": {"patch_content": "diff --git a/answer.py b/answer.py\n+    return 42\n"}
                },
            },
        )
    )

    assert result.status == "pass"
    assert result.raw_output["resolved"] is True


def test_patch_presence_verifier_rejects_missing_terms() -> None:
    adapter = JudgeVerifierAdapter(mode="patch_presence")

    result = adapter.verify(
        VerifierInput(
            benchmark_kit_id="swebench",
            sample_id="sample-1",
            payload={
                "sample": {
                    "id": "sample-1",
                    "metadata": {"expected_patch_contains": ["return 42"]},
                },
                "scheduler_result": {
                    "raw_output": {"patch_content": "diff --git a/answer.py b/answer.py\n+    return 41\n"}
                },
            },
        )
    )

    assert result.status == "fail"
    assert result.raw_output["failure_reason"] == "patch_missing_expected_terms"
