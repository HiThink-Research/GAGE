from __future__ import annotations

from gage_eval.support.inspector import infer_schema


def test_infer_schema_basic(sample_records) -> None:
    schema = infer_schema(sample_records)
    assert "fields" in schema
    assert "modalities" in schema
    assert schema["record_count"] == len(sample_records)
    assert "messages" in schema["fields"]
    assert "text" in schema["modalities"]

