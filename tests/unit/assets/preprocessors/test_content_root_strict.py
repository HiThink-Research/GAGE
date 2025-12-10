import pytest

from gage_eval.assets.datasets.utils.multimodal import embed_local_image_as_data_url


def test_content_root_strict_success(media_assets):
    sample = {
        "metadata": {"content_root": str(media_assets)},
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "dummy.jpg"}}]}],
    }
    result = embed_local_image_as_data_url(sample, strict=True)
    assert result is not None
    assert "data:" in result["image_url"]["url"]


def test_content_root_strict_missing_raises(media_assets):
    sample = {
        "metadata": {"content_root": str(media_assets)},
        "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "missing.jpg"}}]}],
    }
    with pytest.raises(FileNotFoundError):
        embed_local_image_as_data_url(sample, strict=True)
