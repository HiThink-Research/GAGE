import json

from gage_eval.role.judge import appworld_evaluate as appworld_module
from gage_eval.role.judge.appworld_evaluate import AppWorldEvaluate, CommandResult


def test_appworld_evaluate_redacts_test_subset(monkeypatch) -> None:
    sample = {
        "id": "task-1",
        "metadata": {"appworld": {"task_id": "task-1", "subset": "test_normal"}},
        "predict_result": [
            {
                "agent_trace": [
                    {
                        "step_index": 1,
                        "role": "tool",
                        "name": "app__tool",
                        "status": "ok",
                        "output_payload": {"db_dump": "secret"},
                    }
                ]
            }
        ],
    }

    def fake_run(**_kwargs):
        payload = {"tgc": 0.5, "tests": {"passes": ["a"], "fails": ["b"]}}
        return CommandResult(stdout=json.dumps(payload), stderr="", returncode=0)

    monkeypatch.setattr(appworld_module, "_run_container_command", fake_run)
    evaluator = AppWorldEvaluate()
    payload = {"sample": sample, "runtime_handle": {"container_name": "appworld"}, "params": {}}

    result = evaluator.invoke(payload)
    appworld = result["appworld"]

    assert appworld["tgc"] == 0.5
    assert "tests" not in appworld

    trace_step = sample["predict_result"][0]["agent_trace"][0]
    assert trace_step.get("redacted") is True
    assert "output_payload" not in trace_step


def test_appworld_evaluate_missing_container() -> None:
    evaluator = AppWorldEvaluate()
    payload = {"sample": {"metadata": {"appworld": {"task_id": "task-2"}}}, "params": {}}

    result = evaluator.invoke(payload)

    assert result["appworld"]["status"] == "error"
    assert result["appworld"]["failure_reason"] == "missing_container"


def test_appworld_evaluate_parses_aggregate_payload(monkeypatch) -> None:
    sample = {"metadata": {"appworld": {"task_id": "task-3"}}}
    eval_payload = {
        "aggregate": {"task_goal_completion": 1.0, "scenario_goal_completion": 0.5},
        "individual": {
            "task-3": {
                "difficulty": 2,
                "passes": [{"label": "ok"}],
                "failures": [{"label": "bad"}],
            }
        },
    }

    def fake_run(**_kwargs):
        return CommandResult(stdout=json.dumps(eval_payload), stderr="", returncode=0)

    monkeypatch.setattr(appworld_module, "_run_container_command", fake_run)
    evaluator = AppWorldEvaluate()
    payload = {"sample": sample, "runtime_handle": {"container_name": "appworld"}, "params": {}}

    result = evaluator.invoke(payload)["appworld"]

    assert result["tgc"] == 1.0
    assert result["sgc"] == 0.5
    assert result["tests"]["passes"] == eval_payload["individual"]["task-3"]["passes"]
    assert result["tests"]["fails"] == eval_payload["individual"]["task-3"]["failures"]
    assert result["difficulty"] == 2
