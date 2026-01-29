import sys
from pathlib import Path
import unittest
from dataclasses import is_dataclass

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.arcagi2.arcagi2_preprocessor import (
    ARCAGI2Preprocessor,
)
from gage_eval.assets.datasets.sample import (
    Sample,
)


class ARCAGI2PreprocessorTests(unittest.TestCase):
    def test_to_sample_basic(self):
        """Basic ARC-AGI-2: emits messages, references, metadata."""
        pre = ARCAGI2Preprocessor()
        sample = {
            "train": [
                {
                    "input": [[0, 1], [1, 0]],
                    "output": [[1, 0], [0, 1]],
                }
            ],
            "test": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]],
                }
            ],
            "id": "test_problem_001",
        }
        ret = pre.to_sample(sample)

        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))

        # Messages should contain the problem with train examples and test input.
        self.assertIsNotNone(ret.messages)
        self.assertEqual(len(ret.messages), 1)
        self.assertEqual(ret.messages[0].role, "user")
        self.assertIn("Training Examples", ret.messages[0].content[0].text)
        self.assertIn("Test Input", ret.messages[0].content[0].text)
        self.assertIn("--Training Examples--", ret.messages[0].content[0].text)
        self.assertIn("--End of Training Examples--", ret.messages[0].content[0].text)
        self.assertIn("--Test Input--", ret.messages[0].content[0].text)
        self.assertIn("--End of Test Input--", ret.messages[0].content[0].text)

        # References and label should be set to test output.
        self.assertIsNotNone(ret.references)
        self.assertEqual(len(ret.references), 1)
        expected_output = [[0, 1], [1, 0]]
        self.assertEqual(ret.references[0], expected_output)

        # Metadata should contain problem_id, train examples, etc.
        self.assertIsNotNone(ret.metadata)
        self.assertEqual(ret.metadata["problem_id"], "test_problem_001")
        self.assertEqual(ret.metadata["train_example_count"], 1)
        self.assertEqual(len(ret.metadata["train_examples"]), 1)
        self.assertIn("test_input", ret.metadata)

        # ID and schema version should be set.
        self.assertIsNotNone(ret.id)
        self.assertIsNotNone(ret.schema_version)

    def test_to_sample_with_system_prompt(self):
        """Test with system prompt."""
        pre = ARCAGI2Preprocessor()
        sample = {
            "train": [],
            "test": [
                {
                    "input": [[1, 0], [0, 1]],
                    "output": [[0, 1], [1, 0]],
                }
            ],
            "id": "test_002",
        }

        ret = pre.to_sample(sample, system_prompt="You are a pattern recognition expert.")

        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertEqual(len(ret.messages), 2)  # system + user
        self.assertEqual(ret.messages[0].role, "system")
        self.assertEqual(ret.messages[1].role, "user")
        self.assertIn("pattern recognition expert", ret.messages[0].content[0].text)

    def test_to_sample_multiple_train_examples(self):
        """Test with multiple training examples."""
        pre = ARCAGI2Preprocessor()
        sample = {
            "train": [
                {"input": [[0]], "output": [[1]]},
                {"input": [[1]], "output": [[0]]},
                {"input": [[2]], "output": [[3]]},
            ],
            "test": [
                {
                    "input": [[3]],
                    "output": [[2]],
                }
            ],
            "id": "test_003",
        }
        ret = pre.to_sample(sample)

        self.assertIsNotNone(ret)
        self.assertEqual(ret.metadata["train_example_count"], 3)

        # Check that prompt includes all training examples
        prompt_text = ret.messages[0].content[0].text
        self.assertIn("Example 1:", prompt_text)
        self.assertIn("Example 2:", prompt_text)
        self.assertIn("Example 3:", prompt_text)

    def test_to_sample_no_train_examples(self):
        """Test with no training examples."""
        pre = ARCAGI2Preprocessor()
        sample = {
            "train": [],
            "test": [
                {
                    "input": [[1, 2], [3, 4]],
                    "output": [[5, 6], [7, 8]],
                }
            ],
            "id": "test_004",
        }
        ret = pre.to_sample(sample)

        self.assertIsNotNone(ret)
        self.assertEqual(ret.metadata["train_example_count"], 0)

    def test_to_sample_without_test_output(self):
        """Test case where test output is not provided (challenge mode)."""
        pre = ARCAGI2Preprocessor()
        sample = {
            "train": [
                {"input": [[0]], "output": [[1]]},
            ],
            "test": [
                {
                    "input": [[1]],
                    # No output provided - challenge mode
                }
            ],
            "id": "test_005",
        }
        ret = pre.to_sample(sample)

        self.assertIsNotNone(ret)
        # References should be empty when no test output
        self.assertEqual(len(ret.references), 0)

    def test_to_sample_without_problem_id(self):
        """Test case where problem_id is not provided."""
        pre = ARCAGI2Preprocessor()
        sample = {
            "train": [],
            "test": [
                {
                    "input": [[1]],
                    "output": [[0]],
                }
            ],
            # No id provided
        }
        ret = pre.to_sample(sample)

        self.assertIsNotNone(ret)
        # ID should be generated as hash
        self.assertIsNotNone(ret.id)
        # Metadata should still have problem_id (hash-generated)
        self.assertIn("problem_id", ret.metadata)

    def test_to_sample_large_grid(self):
        """Test with larger grid dimensions."""
        pre = ARCAGI2Preprocessor()
        large_grid = [[i % 10 for i in range(30)] for _ in range(30)]
        sample = {
            "train": [],
            "test": [
                {
                    "input": large_grid,
                    "output": large_grid,
                }
            ],
            "id": "test_large_grid",
        }
        ret = pre.to_sample(sample)

        self.assertIsNotNone(ret)
        # Verify grid dimensions are preserved in metadata
        self.assertEqual(len(ret.metadata["test_input"]), 30)
        self.assertEqual(len(ret.metadata["test_input"][0]), 30)

    def test_to_sample_with_instruction(self):
        """Test with custom instruction."""
        pre = ARCAGI2Preprocessor()
        sample = {
            "train": [],
            "test": [
                {
                    "input": [[1]],
                    "output": [[0]],
                }
            ],
            "id": "test_006",
        }

        ret = pre.to_sample(sample, instruction="Output only the JSON grid, nothing else.")

        self.assertIsNotNone(ret)
        prompt_text = ret.messages[0].content[0].text
        self.assertIn("Output only the JSON grid", prompt_text)


if __name__ == "__main__":
    unittest.main()