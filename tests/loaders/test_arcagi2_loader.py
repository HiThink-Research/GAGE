"""Tests for ARC-AGI-2 dataset loader."""

import sys
from pathlib import Path
import unittest
import tempfile
import json

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.loaders.arcagi2.arcagi2_loader import ARCAGI2DatasetLoader
from gage_eval.config.pipeline_config import DatasetSpec


class ARCAGI2LoaderTests(unittest.TestCase):
    """Tests for ARC-AGI-2 dataset loader."""

    def setUp(self):
        """Create a temporary directory with test JSON files."""
        self.test_dir = tempfile.mkdtemp()

        # Create sample ARC-AGI-2 JSON files
        sample_data_1 = {
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
            ],
            "test": [
                {"input": [[1, 0], [0, 1]], "output": [[0, 1], [1, 0]]},
            ],
            "id": "test_problem_001",
        }
        sample_data_2 = {
            "train": [
                {"input": [[1]], "output": [[0]]},
            ],
            "test": [
                {"input": [[0]], "output": [[1]]},
            ],
            "id": "test_problem_002",
        }

        file1 = Path(self.test_dir) / "test_problem_001.json"
        file2 = Path(self.test_dir) / "test_problem_002.json"

        with file1.open("w") as f:
            json.dump(sample_data_1, f)
        with file2.open("w") as f:
            json.dump(sample_data_2, f)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_load_basic(self):
        """Test basic loading of ARC-AGI-2 data."""
        spec = DatasetSpec(
            dataset_id="test_arcagi2_basic",
            loader="arcagi2",
            hub="inline",
            params={
                "path": self.test_dir,
                "preprocess": "arcagi2",
            },
        )

        loader = ARCAGI2DatasetLoader(spec=spec)
        data_source = loader.load(hub_handle=None)

        # Verify data source properties
        self.assertEqual(data_source.dataset_id, "test_arcagi2_basic")
        self.assertIsNotNone(data_source.records)
        self.assertEqual(len(data_source.records), 2)
        self.assertFalse(data_source.streaming)

        # Verify metadata
        self.assertEqual(data_source.metadata["loader"], "arcagi2")
        self.assertEqual(data_source.metadata["path"], self.test_dir)
        self.assertEqual(data_source.metadata["total_files"], 2)

        # Verify sample IDs
        sample_ids = [record.id for record in data_source.records]
        self.assertIn("test_problem_001", sample_ids)
        self.assertIn("test_problem_002", sample_ids)

    def test_load_with_limit(self):
        """Test loading with limit parameter."""
        spec = DatasetSpec(
            dataset_id="test_arcagi2_limit",
            loader="arcagi2",
            hub="inline",
            params={
                "path": self.test_dir,
                "preprocess": "arcagi2",
                "limit": 1,
            },
        )

        loader = ARCAGI2DatasetLoader(spec=spec)
        data_source = loader.load(hub_handle=None)

        # Should only load 1 file
        self.assertEqual(len(data_source.records), 1)

    def test_load_missing_directory(self):
        """Test loading from non-existent directory."""
        spec = DatasetSpec(
            dataset_id="test_arcagi2_missing",
            loader="arcagi2",
            hub="inline",
            params={
                "path": "/non/existent/directory",
            },
        )

        loader = ARCAGI2DatasetLoader(spec=spec)
        with self.assertRaises(FileNotFoundError):
            loader.load(hub_handle=None)

    def test_load_non_directory_path(self):
        """Test loading from a file path instead of directory."""
        spec = DatasetSpec(
            dataset_id="test_arcagi2_file",
            loader="arcagi2",
            hub="inline",
            params={
                "path": __file__,  # This is a file, not a directory
            },
        )

        loader = ARCAGI2DatasetLoader(spec=spec)
        with self.assertRaises(ValueError):
            loader.load(hub_handle=None)

    def test_load_malformed_json(self):
        """Test loading directory with malformed JSON files."""
        # Create a malformed JSON file
        malformed_file = Path(self.test_dir) / "malformed.json"
        with malformed_file.open("w") as f:
            f.write("{invalid json content")

        spec = DatasetSpec(
            dataset_id="test_arcagi2_malformed",
            loader="arcagi2",
            hub="inline",
            params={
                "path": self.test_dir,
                "preprocess": "arcagi2",
            },
        )

        loader = ARCAGI2DatasetLoader(spec=spec)
        # Should still load the valid files, skip malformed ones
        data_source = loader.load(hub_handle=None)
        # Should have loaded 2 valid files (skip malformed)
        self.assertEqual(len(data_source.records), 2)


if __name__ == "__main__":
    unittest.main()