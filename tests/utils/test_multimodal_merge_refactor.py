from gage_eval.assets.datasets.utils.multimodal import merge_multimodal_inputs


def test_merge_multimodal_inputs_dedup_and_sync():
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "question"},
                    {"type": "image_url", "image_url": {"url": "a.png", "detail": "hi-res"}},
                    {"type": "image_url", "image_url": {"url": "a.png"}},
                ],
            }
        ],
        "inputs": {"multi_modal_data": {"image": ["b.png", "a.png"]}},
        "_dataset_metadata": {"path": "/data/mock.jsonl"},
    }

    merge_multimodal_inputs(sample)

    mm = sample["inputs"]["multi_modal_data"]
    assert mm.get("image") == ["a.png"]  # 仅保留消息引用且去重
    media_meta = sample.get("_media_meta") or {}
    assert media_meta.get("images") and media_meta["images"][0]["url"] == "a.png"


def test_merge_multimodal_inputs_embed_local(monkeypatch, tmp_path):
    img_path = tmp_path / "local.png"
    img_path.write_bytes(b"\x89PNG\r\n")
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": str(img_path)}}],
            }
        ],
        "_dataset_metadata": {"path": str(tmp_path / "data.jsonl"), "embed_local_media": True},
    }
    monkeypatch.setenv("GAGE_EVAL_EMBED_LOCAL_MEDIA", "1")

    merge_multimodal_inputs(sample)

    mm = sample["inputs"]["multi_modal_data"]
    assert mm["image"] and str(mm["image"][0]).startswith("data:")
    media_meta = sample.get("_media_meta") or {}
    assert media_meta.get("images") and media_meta["images"][0]["url"] == str(img_path)
