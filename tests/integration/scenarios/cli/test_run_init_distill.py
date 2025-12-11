import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

import run as gage_run
from gage_eval.tools.distill import DistillError


def _write_config(tmp_path: Path, tasks=1) -> Path:
    payload = {
        "datasets": [{"dataset_id": "ds1", "loader": "jsonl", "params": {"path": "dummy.jsonl"}}],
        "role_adapters": [{"adapter_id": "dut", "role_type": "dut_model"}],
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
    }
    if tasks > 1:
        payload["tasks"] = [
            {"task_id": f"t{idx}", "dataset_id": "ds1", "steps": [{"step": "inference", "adapter_id": "dut"}]}
            for idx in range(tasks)
        ]
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_run_mode_minimal_success(monkeypatch, tmp_path):
    cfg = _write_config(tmp_path)
    flags = {}

    class DummyRuntime:
        def run(self):
            flags["run_called"] = True

    monkeypatch.setenv("GAGE_EVAL_SAVE_DIR", str(tmp_path))
    monkeypatch.setattr(gage_run, "_ensure_spawn_start_method", lambda: None)
    monkeypatch.setattr(gage_run, "_preflight_checks", lambda: None)
    monkeypatch.setattr(gage_run, "_install_signal_handlers", lambda: None)
    monkeypatch.setattr(gage_run, "build_default_registry", lambda: "registry")
    monkeypatch.setattr(
        gage_run,
        "build_runtime",
        lambda *args, **kwargs: DummyRuntime(),
    )
    monkeypatch.setattr(
        gage_run,
        "ObservabilityTrace",
        lambda *args, **kwargs: type("Trace", (), {"run_id": "test-run"})(),
    )

    sys.argv = ["run.py", "--config", str(cfg), "--gpus", "0", "--cpus", "1"]
    gage_run.main()

    assert flags.get("run_called") is True


def test_distill_multi_task_rejected(monkeypatch, tmp_path, capsys):
    cfg = _write_config(tmp_path, tasks=2)
    monkeypatch.setattr(gage_run, "analyze_tasks_for_distill", lambda *a, **k: (_ for _ in ()).throw(DistillError("multi-task config detected")))

    sys.argv = ["run.py", "--config", str(cfg), "--distill"]
    with pytest.raises(SystemExit) as excinfo:
        gage_run.main()
    captured = capsys.readouterr()
    assert excinfo.value.code == 1
    assert "multi-task config" in captured.err.lower() or "distill" in captured.err.lower()
