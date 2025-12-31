import base64
from io import BytesIO

import pytest
from PIL import Image

from gage_eval.assets.datasets.preprocessors.mathvista_preprocessor import (
    MathVistaPreprocessor,
    MathVistaStructOnlyPreprocessor,
)


@pytest.mark.fast
def test_mathvista_preprocess_with_pil_image_encodes_data_url():
    """PIL image + multi-choice: emits data URL, option_map/correct_choice, and render flags."""

    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    pre = MathVistaPreprocessor()
    sample = {
        "question": "What is shown?",
        "choices": ["Red", "Green", "Blue"],
        "answer": "Green",
        "decoded_image": img,
    }
    out = pre.to_sample(sample, pre_encode_images=True)

    assert out["messages"] and out["messages"][0]["content"], "messages should include content"
    # The image fragment should be encoded as a data URL.
    image_frags = [frag for frag in out["messages"][0]["content"] if frag.get("type") == "image_url"]
    assert image_frags, "expected an image_url fragment"
    assert image_frags[0]["image_url"]["url"].startswith("data:image/"), "expected a data URL"
    # Multiple-choice mapping and correct choice.
    meta = out["metadata"]
    assert meta["option_map"]["A"] == "Red"
    assert meta["correct_choice"] == "B"
    # Render flags.
    assert out["chat_template_mode"] == "preprocess"
    assert out["cache_suffix"] == "-converted"
    # multi_modal_data should contain the image URL (data URL).
    assert "multi_modal_data" in out["inputs"]
    assert image_frags[0]["image_url"]["url"] in (out["inputs"]["multi_modal_data"].get("image") or [])


@pytest.mark.fast
def test_mathvista_preprocess_path_no_preencode(tmp_path):
    """Local path + pre_encode_images=False: keep the path for backend-side loading."""

    img_path = tmp_path / "foo.png"
    Image.new("RGB", (2, 2), color=(0, 255, 0)).save(img_path)
    pre = MathVistaPreprocessor()
    sample = {
        "question": "Pick color",
        "choices": ["Red", "Green"],
        "answer": 1,
        "image": str(img_path.name),  # relative path
    }
    out = pre.to_sample(sample, content_root=str(tmp_path), pre_encode_images=False)

    # image_url should resolve to an absolute local path (not a data URL).
    img_frag = [frag for frag in out["messages"][0]["content"] if frag.get("type") == "image_url"][0]
    assert img_frag["image_url"]["url"] == str(img_path), "expected an absolute resolved path"
    assert out["metadata"]["image_url"] == str(img_path)
    # multi_modal_data should include the resolved path.
    assert str(img_path) in (out["inputs"]["multi_modal_data"].get("image") or [])
    # Render flags exist.
    assert out["cache_suffix"] == "-converted"


@pytest.mark.fast
def test_mathvista_struct_only_strips_render_flags(tmp_path):
    """Struct-only keeps structured fields and removes prompt/messages/render flags."""

    img = Image.new("RGB", (1, 1), color=(0, 0, 255))
    pre = MathVistaStructOnlyPreprocessor()
    sample = {
        "question": "Q?",
        "choices": ["Yes", "No"],
        "answer": "Yes",
        "decoded_image": img,
    }
    out = pre.to_sample(sample)

    assert out.get("prompt") is None
    assert out.get("messages") == []
    # Render flags should be removed.
    for key in ("chat_template_mode", "template_source", "rendered_by", "cache_suffix"):
        assert key not in out
    # choices/metadata/inputs are still present.
    assert out["choices"]
    assert out["metadata"]["option_map"]["A"] == "Yes"
    assert out["inputs"] == {} or isinstance(out["inputs"], dict)


@pytest.mark.fast
def test_mathvista_preprocess_fills_answer_from_label_and_infers_type():
    pre = MathVistaPreprocessor()
    sample = {
        "question": "How many apples?",
        "label": "3",
        "decoded_image": Image.new("RGB", (1, 1)),
    }
    out = pre.to_sample(sample)
    assert out["answer"] == "3"
    assert out.get("answer_type") == "integer"
