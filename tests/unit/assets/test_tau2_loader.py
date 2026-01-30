from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pytest

from gage_eval.assets.datasets.loaders.tau2_hf_loader import Tau2TasksLoader
from gage_eval.config.pipeline_config import DatasetSpec
from tests.tau2_stub import install_tau2_stub


def _write_tau2_data(root: Path, *, domain: str = "airline") -> Path:
    domain_dir = root / "tau2" / "domains" / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    tasks = [
        {"id": "1", "user_scenario": {"instructions": "Task 1"}, "evaluation_criteria": {"reward_basis": ["DB"]}},
        {"id": "2", "user_scenario": {"instructions": "Task 2"}, "evaluation_criteria": {"reward_basis": ["DB"]}},
    ]
    (domain_dir / "tasks.json").write_text(json.dumps(tasks), encoding="utf-8")
    split = {"base": ["1", "2"]}
    (domain_dir / "split_tasks.json").write_text(json.dumps(split), encoding="utf-8")
    return domain_dir


def test_tau2_loader_builds_trials(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    _write_tau2_data(tmp_path, domain="airline")
    spec = DatasetSpec(
        dataset_id="tau2_airline",
        loader="tau2_tasks",
        params={
            "domain": "airline",
            "task_split": "base",
            "num_trials": 2,
            "seed": 7,
            "data_dir": str(tmp_path),
            "preprocess": "tau2_preprocessor",
        },
    )
    loader = Tau2TasksLoader(spec)
    source = loader.load(None)
    records = list(source.records)

    assert len(records) == 4
    sample = records[0]
    if is_dataclass(sample):
        sample_dict = asdict(sample)
    else:
        sample_dict = sample
    assert "metadata" in sample_dict
    tau2_meta = sample_dict["metadata"]["tau2"]
    assert tau2_meta["task_split"] == "base"
    assert tau2_meta["trial"] in {0, 1}
    assert tau2_meta["seed"] in {7, 8}


def test_tau2_loader_missing_data_dir(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"
    spec = DatasetSpec(
        dataset_id="tau2_missing",
        loader="tau2_tasks",
        params={"domain": "airline", "data_dir": str(missing_dir)},
    )
    loader = Tau2TasksLoader(spec)
    with pytest.raises(FileNotFoundError) as exc:
        loader.load(None)
    assert "TAU2_DATA_DIR" in str(exc.value)
