import base64

import pytest

from gage_eval.role.model.backends.litellm.message_normalizer import MultimodalMessageNormalizer
from gage_eval.role.model.backends.litellm.service_profile import VLLMServiceProfile
from gage_eval.role.model.config.litellm import LiteLLMMultimodalPolicyConfig


pytestmark = pytest.mark.fast


def _normalizer(**policy_kwargs):
    return MultimodalMessageNormalizer(
        LiteLLMMultimodalPolicyConfig(**policy_kwargs),
    )


def test_multimodal_normalizer_preserves_audio_url_block_by_default():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "transcribe"},
                {"type": "audio_url", "audio_url": {"url": "file:///tmp/a.wav"}},
            ],
        }
    ]

    normalized = _normalizer().normalize(messages)

    assert normalized[0]["content"][1] == {
        "type": "audio_url",
        "audio_url": {"url": "file:///tmp/a.wav"},
    }


def test_preserve_blocks_false_applies_fallback_to_supported_multimodal_blocks():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "describe"},
                {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
            ],
        }
    ]

    normalized = _normalizer(preserve_blocks=False, fallback="text").normalize(messages)

    assert normalized[0]["content"] == [
        {"type": "text", "text": "describe"},
        {"type": "text", "text": "[unsupported multimodal block: image_url]"},
    ]


def test_preserve_blocks_false_with_fail_rejects_supported_multimodal_blocks():
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
        }
    ]

    with pytest.raises(ValueError, match="error_type=unsupported_capability"):
        _normalizer(preserve_blocks=False, fallback="fail").normalize(messages)


def test_audio_url_mapping_to_input_audio_uses_data_url_mime_format():
    payload = base64.b64encode(b"audio").decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{payload}"}},
            ],
        }
    ]

    normalized = _normalizer(audio_url_mapping="input_audio", default_audio_format="mp3").normalize(messages)

    assert normalized[0]["content"] == [
        {"type": "input_audio", "input_audio": {"data": payload, "format": "wav"}}
    ]


def test_audio_url_mapping_to_input_audio_uses_default_audio_format_when_missing():
    payload = base64.b64encode(b"audio").decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:audio;base64,{payload}"}},
            ],
        }
    ]

    normalized = _normalizer(audio_url_mapping="input_audio", default_audio_format="wav").normalize(messages)

    assert normalized[0]["content"][0]["input_audio"] == {"data": payload, "format": "wav"}


def test_audio_url_mapping_to_input_audio_uses_default_format_for_mime_less_data_url():
    payload = base64.b64encode(b"audio").decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "audio_url", "audio_url": {"url": f"data:;base64,{payload}"}},
            ],
        }
    ]

    normalized = _normalizer(audio_url_mapping="input_audio", default_audio_format="wav").normalize(messages)

    assert normalized[0]["content"][0]["input_audio"] == {"data": payload, "format": "wav"}


def test_video_file_file_url_and_document_blocks_are_preserved_or_converted():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "video_url", "video_url": "file:///tmp/clip.mp4"},
                {"type": "file", "file": {"file_id": "file-123", "filename": "a.pdf"}},
                {"type": "file_url", "file_url": {"url": "https://example.com/b.pdf", "filename": "b.pdf"}},
                {"type": "file_url", "file_url": "https://example.com/c.pdf"},
                {"type": "document", "document": {"url": "https://example.com/doc.pdf"}},
            ],
        }
    ]

    normalized = _normalizer().normalize(messages)

    assert normalized[0]["content"][0] == {"type": "video_url", "video_url": {"url": "file:///tmp/clip.mp4"}}
    assert normalized[0]["content"][1] == {
        "type": "file",
        "file": {"file_id": "file-123", "filename": "a.pdf"},
    }
    assert normalized[0]["content"][2] == {
        "type": "file",
        "file": {"file_id": "https://example.com/b.pdf", "filename": "b.pdf"},
    }
    assert normalized[0]["content"][3] == {
        "type": "file",
        "file": {"file_id": "https://example.com/c.pdf"},
    }
    assert normalized[0]["content"][4] == {
        "type": "document",
        "document": {"url": "https://example.com/doc.pdf"},
    }


def test_document_source_block_is_preserved_as_official_source_shape():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "document", "source": {"type": "file", "file_id": "file-abc"}},
            ],
        }
    ]

    normalized = _normalizer().normalize(messages)

    assert normalized[0]["content"] == [
        {"type": "document", "source": {"type": "file", "file_id": "file-abc"}},
    ]


def test_unknown_block_fails_loudly_when_fallback_is_fail():
    messages = [{"role": "user", "content": [{"type": "unsupported", "value": "x"}]}]

    with pytest.raises(ValueError, match="unsupported_modality"):
        _normalizer().normalize(messages)


def test_provider_specific_unknown_block_requires_explicit_preserve_fallback():
    messages = [{"role": "user", "content": [{"type": "unsupported", "value": "x"}]}]

    normalized = _normalizer().normalize(messages, fallback="preserve")

    assert normalized[0]["content"] == [{"type": "unsupported", "value": "x"}]


def test_invalid_data_url_raises_invalid_media_payload_without_payload_echo():
    messages = [
        {
            "role": "user",
            "content": [{"type": "audio_url", "audio_url": {"url": "data:audio/wav;base64,@@not-base64@@"}}],
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        _normalizer(audio_url_mapping="input_audio").normalize(messages)

    message = str(exc_info.value)
    assert "invalid_media_payload" in message
    assert "@@not-base64@@" not in message


def test_oversized_data_url_raises_media_payload_too_large_without_payload_echo():
    payload = base64.b64encode(b"abcd").decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [{"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{payload}"}}],
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        _normalizer(audio_url_mapping="input_audio", max_media_bytes=3).normalize(messages)

    message = str(exc_info.value)
    assert "media_payload_too_large" in message
    assert payload not in message
    assert "size=4" in message
    assert "limit=3" in message


def test_oversized_payload_fails_before_scanning_later_invalid_characters():
    payload = ("A" * 64) + "@@invalid-tail@@"
    messages = [
        {
            "role": "user",
            "content": [{"type": "audio_url", "audio_url": {"url": f"data:audio/wav;base64,{payload}"}}],
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        _normalizer(audio_url_mapping="input_audio", max_media_bytes=3).normalize(messages)

    message = str(exc_info.value)
    assert "media_payload_too_large" in message
    assert payload not in message
    assert "@@invalid-tail@@" not in message


def test_oversized_data_url_does_not_echo_long_mime_header():
    long_mime = "audio/" + ("x" * 256)
    header = f"data:{long_mime};base64"
    payload = base64.b64encode(b"abcd").decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [{"type": "audio_url", "audio_url": {"url": f"{header},{payload}"}}],
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        _normalizer(audio_url_mapping="input_audio", max_media_bytes=3).normalize(messages)

    message = str(exc_info.value)
    assert "media_payload_too_large" in message
    assert long_mime not in message
    assert header not in message
    assert len(message) < 180


def test_document_source_raw_base64_data_is_validated():
    messages = [
        {
            "role": "user",
            "content": [{"type": "document", "source": {"type": "base64", "data": "@@bad-base64@@"}}],
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        _normalizer().normalize(messages)

    message = str(exc_info.value)
    assert "invalid_media_payload" in message
    assert "@@bad-base64@@" not in message


def test_document_source_raw_base64_size_limit_is_enforced_without_payload_echo():
    payload = base64.b64encode(b"abcd").decode("ascii")
    messages = [
        {
            "role": "user",
            "content": [{"type": "document", "source": {"type": "base64", "data": payload}}],
        }
    ]

    with pytest.raises(ValueError) as exc_info:
        _normalizer(max_media_bytes=3).normalize(messages)

    message = str(exc_info.value)
    assert "media_payload_too_large" in message
    assert payload not in message
    assert "limit=3" in message


def test_language_model_only_profile_rejects_multimodal_request():
    profile = VLLMServiceProfile(
        mode="direct",
        model="hosted_vllm/text-only",
        topology={"language_model_only": True},
        language_model_only=True,
    )
    messages = [
        {
            "role": "user",
            "content": [{"type": "image_url", "image_url": {"url": "https://example.com/image.png"}}],
        }
    ]

    with pytest.raises(ValueError, match="language_model_only"):
        _normalizer().normalize(messages, profile=profile)
