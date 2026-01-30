from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path

from gage_eval.assets.datasets.preprocessors.tau2_preprocessor import Tau2Preprocessor
from tests.tau2_stub import install_tau2_stub


def test_tau2_preprocessor_builds_sample(tmp_path: Path, monkeypatch) -> None:
    install_tau2_stub(monkeypatch, data_dir=tmp_path)
    preprocessor = Tau2Preprocessor()
    record = {
        "id": "task-1",
        "user_scenario": {"instructions": "Help with booking"},
        "evaluation_criteria": {"reward_basis": ["DB"]},
        "_tau2_domain": "airline",
        "_tau2_task_set": "airline",
        "_tau2_split": "base",
        "trial": 0,
        "seed": 300,
    }
    sample = preprocessor.transform(record)

    assert sample is not None
    sample_dict = asdict(sample) if is_dataclass(sample) else sample
    assert sample_dict["metadata"]["tau2"]["task_id"] == "task-1"
    assert sample_dict["metadata"]["tau2"]["domain"] == "airline"
    assert sample_dict["metadata"]["tau2"]["trial"] == 0
    assert sample_dict["raw_assets"]["tau2"]["task"]["id"] == "task-1"
