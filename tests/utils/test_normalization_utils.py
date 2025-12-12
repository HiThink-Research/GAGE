from gage_eval.assets.datasets.utils.normalization import normalize_sample
import json
import hashlib


def test_normalize_sample_fills_core_fields():
    sample = {
        "prompt": "hello",
        "answer": "yes",
    }
    normalized = normalize_sample(sample, dataset_id="ds1", dataset_metadata={"path": "/tmp/data.jsonl"})

    assert normalized["id"]
    assert normalized["_dataset_id"] == "ds1"
    assert normalized["_dataset_metadata"]["path"] == "/tmp/data.jsonl"
    assert normalized["messages"][0]["content"][0]["text"] == "hello"
    assert normalized["choices"][0]["message"]["content"][0]["text"] == "yes"
    assert normalized["predict_result"] == []
    assert normalized["eval_result"] == {}


def test_normalize_sample_id_hash_stable():
    sample = {
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    normalized = normalize_sample(dict(sample), dataset_id="ds1")
    assert normalized["id"] == "3b6021cdcd4d63218eac241b38088bfc3dcef113"
