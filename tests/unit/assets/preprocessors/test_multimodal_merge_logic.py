from gage_eval.assets.datasets.utils.multimodal import merge_multimodal_inputs


def test_merge_multimodal_inputs_prunes_unreferenced_media():
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "see image"},
                    {"type": "image_url", "image_url": {"url": "a.jpg"}},
                ],
            }
        ],
        "inputs": {
            "prompt": "hello",
            "multi_modal_data": {
                "image": ["a.jpg", "b.jpg"],  # b.jpg is unreferenced and should be pruned
            },
        },
        "_media_meta": {"images": [{"url": "a.jpg"}, {"url": "b.jpg"}]},
    }

    merge_multimodal_inputs(sample)

    mm = sample["inputs"]["multi_modal_data"]
    assert mm["image"] == ["a.jpg"]
    # _media_meta should be pruned consistently as well.
    assert sample["_media_meta"]["images"] == [{"url": "a.jpg"}]
