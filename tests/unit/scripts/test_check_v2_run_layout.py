from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_validate_run():
    script_path = Path(__file__).resolve().parents[3] / "scripts/check_v2_run_layout.py"
    spec = importlib.util.spec_from_file_location("_check_v2_run_layout_under_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.validate_run


validate_run = _load_validate_run()


def test_check_v2_run_layout_rejects_sample_root_trial_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "run-1"
    sample_dir = run_dir / "artifacts/task-1/sample-1"
    infra_dir = sample_dir / "infra"
    trial_dir = sample_dir / "trials/trial_0001"
    infra_dir.mkdir(parents=True)
    (trial_dir / "infra").mkdir(parents=True)
    (trial_dir / "verifier").mkdir(parents=True)
    (infra_dir / "effective_config.json").write_text("{}", encoding="utf-8")
    (infra_dir / "trial_aggregate.json").write_text("{}", encoding="utf-8")
    (infra_dir / "sample_record.json").write_text(
        json.dumps({"trial_results": [{"trial_id": "trial_0001"}]}),
        encoding="utf-8",
    )
    (trial_dir / "infra/trace.jsonl").write_text("", encoding="utf-8")
    (trial_dir / "infra/trial_result.json").write_text("{}", encoding="utf-8")
    (trial_dir / "verifier/verifier_result.json").write_text("{}", encoding="utf-8")
    (sample_dir / "tau2_state.json").write_text("{}", encoding="utf-8")

    findings = validate_run(run_dir)

    assert any("unexpected sample-root artifact" in finding for finding in findings)
