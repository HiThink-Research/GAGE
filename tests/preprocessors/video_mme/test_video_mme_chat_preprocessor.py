"""Tests for Video-MME chat preprocessor."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from dataclasses import is_dataclass

from gage_eval.assets.datasets.preprocessors.video_mme import VideoMMEChatPreprocessor
from gage_eval.assets.datasets.sample import Sample


class VideoMMEChatPreprocessorTests(unittest.TestCase):
    """Tests for VideoMMEChatPreprocessor."""

    def setUp(self) -> None:
        self.preprocessor = VideoMMEChatPreprocessor()
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_dir = Path(self.temp_dir.name)
        # Create a dummy local video file so filtering passes.
        self.dummy_video = self.video_dir / "dummy.mp4"
        self.dummy_video.write_bytes(b"fake")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _with_local_video(self, record: dict, video_id: str | None = None) -> dict:
        """Inject a valid local_video_path into the record."""
        record = dict(record)
        if video_id is None:
            video_id = record.get("video_id") or record.get("videoID") or "dummy"
        local_path = self.video_dir / f"{video_id}.mp4"
        if not local_path.exists():
            local_path.write_bytes(b"fake")
        record["local_video_path"] = str(local_path)
        return record

    def test_to_sample_basic(self) -> None:
        record = self._with_local_video({
            "video_id": "001",
            "videoID": "v001",
            "question_id": "001-1",
            "duration": "short",
            "domain": "Knowledge",
            "sub_category": "Humanity & History",
            "task_type": "Counting Problem",
            "url": "http://example.com/fake.mp4",
            "question": "How many dogs?",
            "options": ["A. 1", "B. 2", "C. 3", "D. 4"],
            "answer": "B",
        })
        result = self.preprocessor.to_sample(record, pre_encode_video=False)
        self.assertIsNotNone(result)
        self.assertTrue(is_dataclass(result))
        self.assertIsInstance(result, Sample)
        self.assertEqual(result.id, "001_v001_001-1")
        self.assertEqual(result.label, "B")
        self.assertEqual(result.references, ["B"])
        self.assertEqual(len(result.options), 4)
        self.assertEqual(result.options[0], "1")
        self.assertEqual(result.options[1], "2")
        # Messages
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].role, "user")
        self.assertEqual(result.messages[0].content[0].type, "text")
        self.assertIn("How many dogs?", result.messages[0].content[0].text)
        self.assertEqual(result.messages[0].content[1].type, "video_url")
        self.assertEqual(
            result.messages[0].content[1].video_url["url"],
            record["local_video_path"],
        )

    def test_to_sample_without_video_id(self) -> None:
        record = self._with_local_video({
            "videoID": "v002",
            "question_id": "002-1",
            "question": "What color?",
            "options": ["A. Red", "B. Blue"],
            "answer": "A",
        }, video_id="v002")
        result = self.preprocessor.to_sample(record, pre_encode_video=False)
        self.assertEqual(result.id, "v002_002-1")

    def test_to_sample_system_prompt(self) -> None:
        record = self._with_local_video({
            "video_id": "003",
            "question_id": "003-1",
            "question": "What is the genre?",
            "options": [],
            "answer": "C",
        })
        result = self.preprocessor.to_sample(
            record,
            system_prompt="You are a helpful assistant.",
            pre_encode_video=False,
        )
        self.assertEqual(len(result.messages), 2)
        self.assertEqual(result.messages[0].role, "system")
        self.assertEqual(
            result.messages[0].content[0].text, "You are a helpful assistant."
        )

    def test_to_sample_options_without_prefix(self) -> None:
        record = self._with_local_video({
            "video_id": "004",
            "question_id": "004-1",
            "question": "Which one?",
            "options": ["Apple", "Banana", "Cherry", "Date"],
            "answer": "C",
        })
        result = self.preprocessor.to_sample(record, pre_encode_video=False)
        self.assertEqual(result.options, ["Apple", "Banana", "Cherry", "Date"])
        prompt = result.messages[0].content[0].text
        self.assertIn("(A) Apple", prompt)
        self.assertIn("(B) Banana", prompt)

    def test_to_sample_with_subtitles(self) -> None:
        record = self._with_local_video({
            "video_id": "005",
            "question_id": "005-1",
            "question": "What is said?",
            "options": ["A. Hello", "B. Hi"],
            "answer": "A",
        })
        result = self.preprocessor.to_sample(
            record,
            include_subtitles=True,
            subtitles="Hello world",
            pre_encode_video=False,
        )
        prompt = result.messages[0].content[0].text
        self.assertIn("This video's subtitles are listed below:", prompt)
        self.assertIn("Hello world", prompt)

    def test_metadata_fields(self) -> None:
        record = self._with_local_video({
            "video_id": "006",
            "question_id": "006-1",
            "duration": "long",
            "domain": "Sports Competition",
            "sub_category": "Basketball",
            "task_type": "Action Recognition",
            "question": "Who scores?",
            "options": ["A. Player 1", "B. Player 2"],
            "answer": "A",
        })
        result = self.preprocessor.to_sample(record, pre_encode_video=False)
        self.assertIsNotNone(result.metadata)
        self.assertEqual(result.metadata["duration"], "long")
        self.assertEqual(result.metadata["domain"], "Sports Competition")
        self.assertEqual(result.metadata["sub_category"], "Basketball")
        self.assertEqual(result.metadata["task_type"], "Action Recognition")

    def test_to_sample_prefers_local_video_path(self) -> None:
        record = self._with_local_video({
            "video_id": "007",
            "question_id": "007-1",
            "question": "What happens?",
            "options": ["A. Jump", "B. Run"],
            "answer": "A",
            "url": "http://example.com/remote.mp4",
        }, video_id="007")
        result = self.preprocessor.to_sample(record, pre_encode_video=False)
        self.assertEqual(len(result.messages), 1)
        self.assertEqual(result.messages[0].content[1].type, "video_url")
        self.assertEqual(
            result.messages[0].content[1].video_url["url"],
            record["local_video_path"],
        )

    def test_to_sample_filters_missing_local_video(self) -> None:
        record = {
            "video_id": "missing",
            "question_id": "missing-1",
            "question": "What?",
            "options": ["A. X", "B. Y"],
            "answer": "A",
        }
        result = self.preprocessor.to_sample(record, pre_encode_video=False)
        self.assertIsNone(result)

    def test_to_sample_filters_nonexistent_local_video(self) -> None:
        record = {
            "video_id": "009",
            "question_id": "009-1",
            "question": "What?",
            "options": ["A. X", "B. Y"],
            "answer": "A",
            "local_video_path": "/does/not/exist.mp4",
        }
        result = self.preprocessor.to_sample(record, pre_encode_video=False)
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
