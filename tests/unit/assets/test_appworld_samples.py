import json
from pathlib import Path

import pytest


@pytest.mark.io
def test_appworld_demo_samples_shape(test_data_dir: Path) -> None:
    sample_path = test_data_dir / "samples" / "appworld_demo.jsonl"
    assert sample_path.exists()

    records = []
    with sample_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))

    assert records
    for record in records:
        assert record.get("task_type") == "agent"
        assert isinstance(record.get("messages"), list)
        assert record.get("messages")
        assert record.get("tool_choice") == "auto"
        assert record.get("sandbox", {}).get("sandbox_id") == "appworld_local"
        appworld = record.get("metadata", {}).get("appworld", {})
        assert appworld.get("task_id")
        allowed_apps = appworld.get("allowed_apps")
        assert isinstance(allowed_apps, list)
        assert allowed_apps
