import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.utils.multimodal import merge_multimodal_inputs


def test_merge_multimodal_audio_video_file():
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "a1.wav"}},
                    {"type": "audio_url", "audio_url": {"url": "a1.wav"}},  # duplicate
                    {"type": "video_url", "video_url": {"url": "v1.mp4"}},
                    {"type": "file_url", "file_url": {"url": "f1.pdf"}},
                ],
            }
        ],
        "inputs": {"multi_modal_data": {"audio": ["legacy.wav"], "video": [], "file": []}},
    }

    merge_multimodal_inputs(sample)

    mm = sample["inputs"]["multi_modal_data"]
    # 仅保留消息中引用的媒体，legacy 未被引用会被清理
    assert mm["audio"] == ["a1.wav"]
    assert mm["video"] == ["v1.mp4"]
    assert mm["file"] == ["f1.pdf"]


def test_merge_multimodal_sync_keeps_referenced_only():
    sample = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "audio_url", "audio_url": {"url": "keep.wav"}},
                    {"type": "video_url", "video_url": {"url": "keep.mp4"}},
                ],
            }
        ],
        "inputs": {"multi_modal_data": {"audio": ["keep.wav", "drop.wav"], "video": ["keep.mp4", "drop.mp4"]}},
    }

    merge_multimodal_inputs(sample)

    mm = sample["inputs"]["multi_modal_data"]
    assert mm["audio"] == ["keep.wav"]
    assert mm["video"] == ["keep.mp4"]
