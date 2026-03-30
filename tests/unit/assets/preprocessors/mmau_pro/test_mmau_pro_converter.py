import os  # 文件路径与存在性检查
import sys  # 修改 sys.path 以便导入 src 包
import tempfile  # 创建临时目录与文件
from pathlib import Path  # 路径拼接
import unittest  # 单元测试框架
from dataclasses import is_dataclass  # 检查返回是否为 dataclass


ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from gage_eval.assets.datasets.preprocessors.mmau_pro.mmau_pro_converter import (  # noqa: E402
    MMAUProConverter,
)


class MMAUProConverterTests(unittest.TestCase):
    def test_to_sample_with_local_base64_audio(self) -> None:
        """验证预处理后 Sample 结构与音频 base64 data URL 是否符合预期。"""
        pre = MMAUProConverter()

        with tempfile.TemporaryDirectory() as tmpdir:
            # 构造类似 MMAU-Pro 的相对路径与本地假音频文件
            rel_audio = "data/test_audio.wav"
            audio_dir = Path(tmpdir) / "data"
            audio_dir.mkdir(parents=True, exist_ok=True)
            audio_path = audio_dir / "test_audio.wav"

            # 写入少量字节，避免真实依赖外部数据
            with open(audio_path, "wb") as f:
                f.write(b"\x00\x01\x02")

            record = {
                "id": "sample-1",
                "audio_path": [rel_audio],
                "question": "What is being prepared in the audio?",
                "answer": "Boba tea",
                "choices": ["Boba tea", "Milk", "Coffee"],
                "category": "sound",
                "length_type": "medium",
                "perceptual_skills": ["Acoustic Source Characterization"],
                "reasoning_skills": ["Procedural Reasoning"],
            }

            sample = pre.to_sample(record, audio_path_root=tmpdir)

            # 基本结构检查
            self.assertIsNotNone(sample)
            self.assertTrue(is_dataclass(sample))
            self.assertEqual(sample.id, "sample-1")
            self.assertEqual(sample.label, "Boba tea")
            self.assertEqual(sample.references, ["Boba tea"])
            self.assertEqual(sample.options, ["Boba tea", "Milk", "Coffee"])

            # 消息与内容结构检查：user 消息 + audio_url + text
            self.assertEqual(len(sample.messages), 1)
            user_msg = sample.messages[0]
            self.assertEqual(user_msg.role, "user")
            self.assertGreaterEqual(len(user_msg.content), 2)

            audio_content = user_msg.content[0]
            text_content = user_msg.content[1]

            self.assertEqual(audio_content.type, "audio_url")
            self.assertIn("data:audio/wav;base64,", audio_content.audio_url["url"])
            self.assertEqual(text_content.type, "text")
            self.assertIn(record["question"], text_content.text)

            # 元数据中应包含基础信息与路径
            self.assertEqual(sample.metadata["category"], "sound")
            self.assertEqual(sample.metadata["length_type"], "medium")
            self.assertIn(rel_audio, sample.metadata["audio_paths"])
            self.assertEqual(sample.metadata["picked_audio"], rel_audio)
            self.assertEqual(sample.metadata["audio_path_root"], tmpdir)


if __name__ == "__main__":
    unittest.main()

