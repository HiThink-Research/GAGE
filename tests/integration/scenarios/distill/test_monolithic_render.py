import os
from pathlib import Path

import yaml

from gage_eval.tools.distill import DistillError, distill_to_template


def _base_payload(task_count: int) -> dict:
    tasks = []
    for idx in range(task_count):
        tasks.append(
            {
                "task_id": f"task_{idx}",
                "dataset_id": "dummy_dataset",
                "steps": [{"step": "inference", "adapter_id": "dut"}],
            }
        )
    return {
        "metadata": {"name": "monolithic_fixture"},
        "custom": {"steps": [{"step": "inference", "adapter_id": "dut"}]},
        "datasets": [
            {"dataset_id": "dummy_dataset", "loader": "jsonl", "params": {"path": "dummy.jsonl"}},
        ],
        "role_adapters": [
            {"adapter_id": "dut", "role_type": "dut_model"},
        ],
        "tasks": tasks,
    }


def test_force_merge_writes_template(tmp_path: Path):
    payload = _base_payload(2)
    out_dir = tmp_path / "suite_out"
    target = distill_to_template(
        payload,
        builtin_name="suite_monolithic",
        version="V9",
        output_root=out_dir,
        force_merge=True,
    )
    assert target.exists()
    data = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert data.get("kind") == "BuiltinTemplate"
    assert data.get("metadata", {}).get("monolithic") is True
    assert data.get("metadata", {}).get("version") == "V9"
    assert data.get("metadata", {}).get("digest", "").startswith("sha256:")
    definition = data.get("definition") or {}
    assert len(definition.get("tasks") or []) == 2


def test_distill_rejects_overwrite(tmp_path: Path):
    payload = _base_payload(1)
    distill_to_template(payload, builtin_name="suite_atomic", version="V1", output_root=tmp_path, force_merge=False)
    try:
        distill_to_template(payload, builtin_name="suite_atomic", version="V1", output_root=tmp_path, force_merge=False)
    except DistillError:
        pass
    else:
        raise AssertionError("expected DistillError when writing to existing template file")


def test_auto_increment_version(tmp_path: Path):
    payload = _base_payload(1)
    # First distill without version -> V1
    first = distill_to_template(payload, builtin_name="suite_auto", version=None, output_root=tmp_path, force_merge=False)
    assert first.name == "v1.yaml"
    data1 = yaml.safe_load(first.read_text(encoding="utf-8"))
    assert data1["metadata"]["version"] == "V1"
    # Second distill without version -> V2
    second = distill_to_template(payload, builtin_name="suite_auto", version=None, output_root=tmp_path, force_merge=False)
    assert second.name == "v2.yaml"
    data2 = yaml.safe_load(second.read_text(encoding="utf-8"))
    assert data2["metadata"]["version"] == "V2"


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        test_force_merge_writes_template(tmp)
        test_distill_rejects_overwrite(tmp)
        print("monolithic render tests passed")
