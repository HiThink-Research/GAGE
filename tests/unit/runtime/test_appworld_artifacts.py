from types import SimpleNamespace

from gage_eval.role.judge.appworld_evaluate import _resolve_export_dir, _resolve_output_dir


def test_appworld_output_dir_defaults() -> None:
    output_dir = _resolve_output_dir(
        appworld_root="/run",
        experiment_name="exp",
        task_id="task-1",
        output_dir_template=None,
    )

    assert output_dir == "/run/experiments/outputs/exp/tasks/task-1"


def test_appworld_export_dir_defaults(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    payload = {"trace": SimpleNamespace(run_id="run-123")}

    export_dir = _resolve_export_dir(
        base_dir=None,
        payload=payload,
        task_id="task-1",
        experiment_name="exp",
        default_enabled=True,
    )

    assert export_dir == str(tmp_path / "run-123" / "appworld_artifacts")
