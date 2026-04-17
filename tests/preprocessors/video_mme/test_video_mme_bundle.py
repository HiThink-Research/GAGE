"""Tests for Video-MME bundle."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.bundles.video_mme.bundle import VideoMMEBundle


class VideoMMEBundleTests(unittest.TestCase):
    """Tests for VideoMMEBundle."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.video_dir = Path(self.temp_dir.name)
        # Create fake mp4 files
        (self.video_dir / "abc123.mp4").write_bytes(b"fake")
        (self.video_dir / "def456.mp4").write_bytes(b"fake")

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_load_scans_directory(self) -> None:
        bundle = VideoMMEBundle(video_dir=str(self.video_dir))
        bundle.load()
        self.assertEqual(len(bundle.available_video_ids), 2)
        self.assertIn("abc123", bundle.available_video_ids)
        self.assertIn("def456", bundle.available_video_ids)

    def test_provide_returns_sample_with_local_path(self) -> None:
        bundle = VideoMMEBundle(video_dir=str(self.video_dir))
        bundle.load()
        sample = {"video_id": "abc123", "question": "What?"}
        result = bundle.provide(sample)
        self.assertIsNotNone(result)
        self.assertEqual(
            Path(result["local_video_path"]).resolve(),
            (self.video_dir / "abc123.mp4").resolve(),
        )

    def test_provide_does_not_set_path_for_missing_video(self) -> None:
        bundle = VideoMMEBundle(video_dir=str(self.video_dir))
        bundle.load()
        sample = {"video_id": "missing", "question": "What?"}
        result = bundle.provide(sample)
        self.assertIsNotNone(result)
        self.assertNotIn("local_video_path", result)

    def test_provide_uses_videoID_when_video_id_missing(self) -> None:
        bundle = VideoMMEBundle(video_dir=str(self.video_dir))
        bundle.load()
        sample = {"videoID": "def456", "question": "What?"}
        result = bundle.provide(sample)
        self.assertIsNotNone(result)
        self.assertEqual(
            Path(result["local_video_path"]).resolve(),
            (self.video_dir / "def456.mp4").resolve(),
        )

    def test_provide_does_not_set_path_when_no_id(self) -> None:
        bundle = VideoMMEBundle(video_dir=str(self.video_dir))
        bundle.load()
        sample = {"question": "What?"}
        result = bundle.provide(sample)
        self.assertIsNotNone(result)
        self.assertNotIn("local_video_path", result)


if __name__ == "__main__":
    unittest.main()
