from gage_eval.assets.datasets.manager import DataManager, DataSource


def test_datamanager_transport_without_mutation():
    dm = DataManager()
    records = [
        {
            "id": "s1",
            "_dataset_id": "ds",
            "_dataset_metadata": {"path": "/tmp/data.jsonl"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "choices": [],
            "inputs": {"prompt": "hi"},
        }
    ]
    source = DataSource(dataset_id="ds", records=records, metadata={"path": "/tmp/data.jsonl"})
    dm.register_source(source)

    out = list(dm.iter_samples("ds"))
    assert out[0]["id"] == "s1"
    assert out[0]["messages"][0]["content"][0]["text"] == "hi"
    assert out[0]["inputs"]["prompt"] == "hi"
