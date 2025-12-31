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
    """PIL 图像 + 多选题：应生成 data URL、option_map/correct_choice、渲染标记。"""

    img = Image.new("RGB", (2, 2), color=(255, 0, 0))
    pre = MathVistaPreprocessor()
    sample = {
        "question": "What is shown?",
        "choices": ["Red", "Green", "Blue"],
        "answer": "Green",
        "decoded_image": img,
    }
    out = pre.to_sample(sample, pre_encode_images=True)

    assert out["messages"] and out["messages"][0]["content"], "messages 应包含内容"
    # 图片片段应该是 data URL
    image_frags = [frag for frag in out["messages"][0]["content"] if frag.get("type") == "image_url"]
    assert image_frags, "应包含 image_url 片段"
    assert image_frags[0]["image_url"]["url"].startswith("data:image/"), "应编码为 data URL"
    # 多选映射与答案
    meta = out["metadata"]
    assert meta["option_map"]["A"] == "Red"
    assert meta["correct_choice"] == "B"
    # 渲染标记
    assert out["chat_template_mode"] == "preprocess"
    assert out["cache_suffix"] == "-converted"
    # multi_modal_data 应包含图片 URL（data URL）
    assert "multi_modal_data" in out["inputs"]
    assert image_frags[0]["image_url"]["url"] in (out["inputs"]["multi_modal_data"].get("image") or [])


@pytest.mark.fast
def test_mathvista_preprocess_path_no_preencode(tmp_path):
    """本地路径 + pre_encode_images=False：保留路径，后端可自行加载。"""

    img_path = tmp_path / "foo.png"
    Image.new("RGB", (2, 2), color=(0, 255, 0)).save(img_path)
    pre = MathVistaPreprocessor()
    sample = {
        "question": "Pick color",
        "choices": ["Red", "Green"],
        "answer": 1,
        "image": str(img_path.name),  # 相对路径
    }
    out = pre.to_sample(sample, content_root=str(tmp_path), pre_encode_images=False)

    # image_url 应为解析后的本地路径（非 data URL）
    img_frag = [frag for frag in out["messages"][0]["content"] if frag.get("type") == "image_url"][0]
    assert img_frag["image_url"]["url"] == str(img_path), "应解析为绝对路径"
    assert out["metadata"]["image_url"] == str(img_path)
    # multi_modal_data 应收录路径
    assert str(img_path) in (out["inputs"]["multi_modal_data"].get("image") or [])
    # 渲染标记存在
    assert out["cache_suffix"] == "-converted"


@pytest.mark.fast
def test_mathvista_struct_only_strips_render_flags(tmp_path):
    """Struct-only 仅保留结构化字段，移除 prompt/messages/渲染标记。"""

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
    # 渲染标记应被移除
    for key in ("chat_template_mode", "template_source", "rendered_by", "cache_suffix"):
        assert key not in out
    # choices/metadata/inputs 仍保留
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
