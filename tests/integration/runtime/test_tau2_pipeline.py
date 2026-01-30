from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

from gage_eval.assets.datasets.loaders.tau2_hf_loader import Tau2TasksLoader
from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.role.context.tau2_bootstrap import Tau2BootstrapContext
from gage_eval.role.judge.tau2_eval import Tau2Evaluate
from gage_eval.sandbox.manager import SandboxManager
from gage_eval.sandbox.provider import SandboxProvider, SandboxScope
from tests.tau2_stub import install_tau2_stub


def _write_tau2_data(root: Path, *, domain: str = "airline") -> None:
    domain_dir = root / "tau2" / "domains" / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        {
            "id": f"{domain}-1",
            "user_scenario": {"instructions": f"Task {domain}"},
            "evaluation_criteria": {"reward_basis": ["DB"]},
        },
    ]
    (domain_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    (domain_dir / "split_tasks.json").write_text(json.dumps({"base": [f"{domain}-1"]}), encoding="utf-8")


def test_tau2_pipeline_end_to_end(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    _write_tau2_data(tmp_path, domain="airline")
    _write_tau2_data(tmp_path, domain="retail")

    manager = SandboxManager()
    judge = Tau2Evaluate()
    bootstrap = Tau2BootstrapContext()
    for domain in ("airline", "retail"):
        spec = DatasetSpec(
            dataset_id=f"tau2_{domain}",
            loader="tau2_tasks",
            params={
                "domain": domain,
                "task_split": "base",
                "data_dir": str(tmp_path),
                "num_trials": 2,
                "preprocess": "tau2_preprocessor",
            },
        )
        loader = Tau2TasksLoader(spec)
        source = loader.load(None)
        records = list(source.records)
        assert len(records) == 2
        trials = {record.metadata["tau2"]["trial"] for record in records if hasattr(record, "metadata")}
        assert trials == {0, 1}
        record = records[0]
        sample = asdict(record) if is_dataclass(record) else record

        provider = SandboxProvider(
            manager,
            {"runtime": "tau2", "runtime_configs": {"data_dir": str(tmp_path)}},
            SandboxScope(sample_id=f"tau2-{domain}"),
        )
        bootstrap.provide({"sample": sample, "sandbox_provider": provider})
        runtime = provider.get_handle().sandbox
        runtime.exec_tool("respond", {"message": "please stop"})
        output = judge.invoke({"sample": sample, "sandbox_provider": provider})
        assert output["tau2"]["reward"] == 1.0
