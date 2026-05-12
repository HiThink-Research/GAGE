import json
from pathlib import Path


FIXTURE_DIR = (
    Path(__file__).resolve().parents[2]
    / "_support"
    / "external_harness_kits"
    / "harbor_tb2_1case"
)


def load_fixture(name: str) -> dict:
    with (FIXTURE_DIR / name).open(encoding="utf-8") as handle:
        return json.load(handle)


def test_harbor_tb2_1case_fixtures_are_readable_and_traceable_to_spike() -> None:
    job_result = load_fixture("job_result.json")
    trial_result = load_fixture("trial_result.json")
    result_tree = load_fixture("result_tree.json")

    assert result_tree["source"] == {
        "benchmark": "Terminal-Bench 2.0",
        "spike_job_id": "gage_tb2_1case_lmstudio_20260510_161501",
        "harbor_version": "0.6.6",
    }
    assert result_tree["job_result"] == job_result
    assert result_tree["trials"][0]["result"] == trial_result


def test_real_trial_fixture_preserves_harbor_result_fields_needed_by_parser() -> None:
    trial_result = load_fixture("trial_result.json")
    result_tree = load_fixture("result_tree.json")
    trial_tree = result_tree["trials"][0]

    assert trial_result["task_name"] == "gpt2-codegolf"
    assert trial_result["trial_name"] == "gpt2-codegolf__sLkuvPz"
    assert isinstance(trial_result["agent_result"], dict)
    assert isinstance(trial_result["verifier_result"]["rewards"], dict)
    assert trial_tree["agent"]["trajectory"]["schema_version"]
    assert trial_tree["agent"]["trajectory"]["steps"]

    tool_calls = [
        tool_call
        for step in trial_tree["agent"]["trajectory"]["steps"]
        for tool_call in step.get("tool_calls", [])
    ]
    assert tool_calls
    assert {"function_name", "arguments"}.issubset(tool_calls[0])


def test_failed_trial_fixture_covers_exception_and_missing_result_paths() -> None:
    failed_trial = load_fixture("trial_with_exception_info.json")

    assert failed_trial["trial_name"] == "gpt2-codegolf__synthetic_failure"
    assert failed_trial["exception_info"]["type"] == "RuntimeError"
    assert failed_trial["agent_result"]["exit_code"] != 0
    assert failed_trial["verifier_result"] is None
    assert failed_trial["launcher_result"] is None
    assert failed_trial["job_result"] is None
