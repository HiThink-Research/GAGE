"""Integration tests for ARC-AGI-2 dataset loading and preprocessing."""

import sys
from pathlib import Path
import unittest
import tempfile
import json

ROOT = Path(__file__).resolve().parents[3] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.config.pipeline_config import DatasetSpec
from gage_eval.assets.datasets.loaders.arcagi2.arcagi2_loader import ARCAGI2DatasetLoader
from gage_eval.registry import registry


class ARCAGI2IntegrationTests(unittest.TestCase):
    """Integration tests for ARC-AGI-2 dataset end-to-end flow."""

    def setUp(self):
        """Create a temporary directory with test JSON files."""
        self.test_dir = tempfile.mkdtemp()

        # Create sample ARC-AGI-2 JSON files
        sample_data_1 = {
            "train": [
                {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
                {"input": [[2, 3], [3, 2]], "output": [[3, 2], [2, 3]]},
            ],
            "test": [
                {"input": [[4, 5], [5, 4]], "output": [[5, 4], [4, 5]]},
            ],
            "id": "problem_001",
        }
        sample_data_2 = {
            "train": [
                {"input": [[1]], "output": [[0]]},
            ],
            "test": [
                {"input": [[0]], "output": [[1]]},
            ],
            "id": "problem_002",
        }

        file1 = Path(self.test_dir) / "problem_001.json"
        file2 = Path(self.test_dir) / "problem_002.json"

        with file1.open("w") as f:
            json.dump(sample_data_1, f)
        with file2.open("w") as f:
            json.dump(sample_data_2, f)

    def tearDown(self):
        """Clean up temporary directory."""
        import shutil
        shutil.rmtree(self.test_dir)

    def test_end_to_end_loading_with_preprocessing(self):
        """Test end-to-end loading and preprocessing of ARC-AGI-2 data."""
        spec = DatasetSpec(
            dataset_id="test_arcagi2_integration",
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
        self.assertEqual(data_source.dataset_id, "test_arcagi2_integration")
        self.assertIsNotNone(data_source.records)
        self.assertEqual(len(data_source.records), 2)
        self.assertFalse(data_source.streaming)

        # Verify first sample
        sample_001 = data_source.records[0]
        self.assertEqual(sample_001.id, "problem_001")
        self.assertIsNotNone(sample_001.messages)
        self.assertEqual(len(sample_001.messages), 1)  # user message
        self.assertEqual(sample_001.messages[0].role, "user")

        # Verify training examples in prompt
        prompt_text = sample_001.messages[0].content[0].text
        self.assertIn("--Training Examples--", prompt_text)
        self.assertIn("--End of Training Examples--", prompt_text)
        self.assertIn("--Test Input--", prompt_text)
        self.assertIn("--End of Test Input--", prompt_text)
        self.assertIn("Example 1:", prompt_text)
        self.assertIn("Example 2:", prompt_text)

        # Verify metadata
        self.assertIsNotNone(sample_001.metadata)
        self.assertEqual(sample_001.metadata["problem_id"], "problem_001")
        self.assertEqual(sample_001.metadata["train_example_count"], 2)

        # Verify references (expected output)
        self.assertIsNotNone(sample_001.references)
        self.assertEqual(len(sample_001.references), 1)
        expected_output = [[5, 4], [4, 5]]
        self.assertEqual(sample_001.references[0], expected_output)

    def test_end_to_end_loading_multiple_files(self):
        """Test loading multiple ARC-AGI-2 problem files."""
        spec = DatasetSpec(
            dataset_id="test_arcagi2_multi",
            loader="arcagi2",
            hub="inline",
            params={
                "path": self.test_dir,
                "preprocess": "arcagi2",
            },
        )

        loader = ARCAGI2DatasetLoader(spec=spec)
        data_source = loader.load(hub_handle=None)

        # Should have loaded both files
        self.assertEqual(len(data_source.records), 2)

        # Verify problem IDs are from filenames
        problem_ids = {r.id for r in data_source.records}
        self.assertIn("problem_001", problem_ids)
        self.assertIn("problem_002", problem_ids)

    def test_end_to_end_with_limit(self):
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

    def test_end_to_end_with_system_prompt(self):
        """Test preprocessing with custom system prompt."""
        spec = DatasetSpec(
            dataset_id="test_arcagi2_system",
            loader="arcagi2",
            hub="inline",
            params={
                "path": self.test_dir,
                "preprocess": "arcagi2",
                "preprocess_kwargs": {
                    "system_prompt": "You are a pattern recognition expert.",
                },
            },
        )

        loader = ARCAGI2DatasetLoader(spec=spec)
        data_source = loader.load(hub_handle=None)

        sample = data_source.records[0]
        # Should have system + user messages
        self.assertEqual(len(sample.messages), 2)
        self.assertEqual(sample.messages[0].role, "system")
        self.assertIn("pattern recognition expert", sample.messages[0].content[0].text)
        self.assertEqual(sample.messages[1].role, "user")

    def test_registry_has_arcagi2_components(self):
        """Test that ARC-AGI-2 components are registered in the registry."""
        # Check loader registration
        loader_class = registry.get("dataset_loaders", "arcagi2")
        self.assertIsNotNone(loader_class)
        self.assertEqual(loader_class.__name__, "ARCAGI2DatasetLoader")

        # Check preprocessor registration
        preprocessor_class = registry.get("dataset_preprocessors", "arcagi2")
        self.assertIsNotNone(preprocessor_class)
        self.assertEqual(preprocessor_class.__name__, "ARCAGI2PreprocessorProvider")


if __name__ == "__main__":
    unittest.main()