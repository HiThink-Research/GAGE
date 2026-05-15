from __future__ import annotations

from gage_eval.assets.datasets.preprocessors.mmmu_preprocessor import MMMUMultimodalPreprocessor


def test_mmmu_preprocessor_attaches_image_fields_to_user_message() -> None:
    sample = MMMUMultimodalPreprocessor(on_error="raise").transform(
        {
            "id": "validation_Agriculture_16",
            "messages": [
                {"role": "system", "content": "System prompt"},
                {"role": "user", "content": "<image 1> What is shown?"},
            ],
            "images": [
                {
                    "path": (
                        "https://huggingface.co/datasets/modelscope/"
                        "MMMU-Reasoning-Distill-Validation/resolve/main/images/"
                        "validation_Agriculture_16_1.png"
                    )
                }
            ],
        },
        dataset_id="mmmu_val_huggingface",
    )

    user_content = sample.messages[1].content

    assert user_content[0].type == "image_url"
    assert user_content[0].image_url["url"].endswith("validation_Agriculture_16_1.png")
    assert sample.inputs["multi_modal_data"]["image"] == [user_content[0].image_url["url"]]


def test_mmmu_preprocessor_uses_image_number_fields_when_images_list_is_absent() -> None:
    sample = MMMUMultimodalPreprocessor(on_error="raise").transform(
        {
            "id": "sample",
            "messages": [{"role": "user", "content": "<image 1> What is shown?"}],
            "image_1": {"path": "https://example.test/image.png"},
        },
        dataset_id="mmmu",
    )

    assert sample.messages[0].content[0].image_url["url"] == "https://example.test/image.png"
    assert sample.inputs["multi_modal_data"]["image"] == ["https://example.test/image.png"]


def test_mmmu_preprocessor_does_not_duplicate_existing_image_fragments() -> None:
    sample = MMMUMultimodalPreprocessor(on_error="raise").transform(
        {
            "id": "sample",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": "https://example.test/existing.png"}},
                        {"type": "text", "text": "What is shown?"},
                    ],
                }
            ],
            "image_1": {"path": "https://example.test/extra.png"},
        },
        dataset_id="mmmu",
    )

    content = sample.messages[0].content
    assert [item.type for item in content].count("image_url") == 1
    assert content[0].image_url["url"] == "https://example.test/existing.png"


def test_mmmu_preprocessor_leaves_messages_unchanged_without_images() -> None:
    sample = MMMUMultimodalPreprocessor(on_error="raise").transform(
        {
            "id": "sample",
            "messages": [{"role": "user", "content": "Only text"}],
        },
        dataset_id="mmmu",
    )

    assert sample.messages[0].content[0].type == "text"
    assert sample.messages[0].content[0].text == "Only text"
    assert "multi_modal_data" not in sample.inputs


def test_mmmu_preprocessor_skips_unsupported_local_image_paths() -> None:
    sample = MMMUMultimodalPreprocessor(on_error="raise").transform(
        {
            "id": "sample",
            "messages": [{"role": "user", "content": "<image 1> What is shown?"}],
            "image_1": {"path": "images/local.png"},
        },
        dataset_id="mmmu",
    )

    assert sample.messages[0].content[0].type == "text"
    assert "multi_modal_data" not in sample.inputs


def test_mmmu_preprocessor_skips_image_injection_without_user_message() -> None:
    sample = MMMUMultimodalPreprocessor(on_error="raise").transform(
        {
            "id": "sample",
            "messages": [{"role": "system", "content": "System only"}],
            "image_1": {"path": "https://example.test/image.png"},
        },
        dataset_id="mmmu",
    )

    assert sample.messages[0].role == "system"
    assert sample.messages[0].content[0].type == "text"
    assert "multi_modal_data" not in sample.inputs
