import sys
from pathlib import Path
import unittest
from dataclasses import is_dataclass

ROOT = Path(__file__).resolve().parents[2] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.assets.datasets.preprocessors.simpleqa_verified.simpleqa_verified_preprocessor import (
    SimpleQAVerifiedPreprocessor,
)
from gage_eval.assets.datasets.sample import (
    Sample,
)


class SimpleQAVerifiedPreprocessorTests(unittest.TestCase):
    def test_to_sample(self):
        """Text QA: emits messages, references, metadata."""
        pre = SimpleQAVerifiedPreprocessor()
        sample = {
            "problem": "To whom did Mehbooba Mufti Sayed contest the 2019 Lok Sabha elections and lose?",
            "answer": "Hasnain Masoodi",
            "topic": "Politics",
            "answer_type": "Person",
            "multi_step": False,
            "requires_reasoning": False,
            "urls": "https://example.com/url1,https://example.com/url2",
            "original_index": 9,
        }
        ret = pre.to_sample(sample)

        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        
        # Messages should contain the problem.
        self.assertIsNotNone(ret.messages)
        self.assertEqual(len(ret.messages), 1)
        self.assertEqual(ret.messages[0].role, "user")
        self.assertIn(sample["problem"], ret.messages[0].content[0].text)

        # References and label should be set.
        self.assertIsNotNone(ret.references)
        self.assertEqual(len(ret.references), 1)
        self.assertEqual(ret.references[0], sample["answer"])
        self.assertEqual(ret.label, sample["answer"])

        # Metadata should contain topic, answer_type, multi_step, requires_reasoning, urls, original_index.
        self.assertIsNotNone(ret.metadata)
        self.assertEqual(ret.metadata["topic"], "Politics")
        self.assertEqual(ret.metadata["answer_type"], "Person")
        self.assertEqual(ret.metadata["multi_step"], False)
        self.assertEqual(ret.metadata["requires_reasoning"], False)
        self.assertEqual(ret.metadata["urls"], sample["urls"])
        self.assertEqual(ret.metadata["original_index"], 9)

        # ID and schema version should be set.
        self.assertIsNotNone(ret.id)
        self.assertIsNotNone(ret.schema_version)

    def test_to_sample_with_system_prompt(self):
        """Test with system prompt."""
        pre = SimpleQAVerifiedPreprocessor()
        sample = {
            "problem": "What is the capital of France?",
            "answer": "Paris",
            "topic": "Geography",
            "answer_type": "Place",
            "multi_step": False,
            "requires_reasoning": False,
        }

        ret = pre.to_sample(sample, system_prompt="You are a helpful assistant.")
        
        self.assertIsNotNone(ret)
        self.assertTrue(is_dataclass(ret))
        self.assertEqual(len(ret.messages), 2)  # system + user
        self.assertEqual(ret.messages[0].role, "system")
        self.assertEqual(ret.messages[1].role, "user")
        # System prompt should not be duplicated in the user message content.
        self.assertNotIn("You are a helpful assistant.", ret.messages[1].content[0].text)

    def test_simpleqa_verified_preprocess_handles_missing_optional_fields(self):
        """Missing optional fields are handled gracefully."""
        pre = SimpleQAVerifiedPreprocessor()
        sample = {
            "problem": "What is 2+2?",
            "answer": "4",
        }
        ret = pre.to_sample(sample)

        # Should handle missing fields gracefully.
        self.assertEqual(ret.metadata["topic"], "")
        self.assertEqual(ret.metadata["answer_type"], "")
        self.assertEqual(ret.metadata["multi_step"], False)
        self.assertEqual(ret.metadata["requires_reasoning"], False)
        self.assertEqual(ret.metadata["urls"], "")
        self.assertIsNone(ret.metadata["original_index"])
        self.assertEqual(len(ret.references), 1)
        self.assertEqual(ret.references[0], "4")

    def test_simpleqa_verified_preprocess_handles_empty_answer(self):
        """Empty answer results in empty references list."""
        pre = SimpleQAVerifiedPreprocessor()
        sample = {
            "problem": "What is 2+2?",
            "answer": "",
        }
        ret = pre.to_sample(sample)

        # Empty answer should result in empty references list.
        self.assertEqual(len(ret.references), 0)
        self.assertEqual(ret.label, "")  # label is set to answer, which is empty string

    def test_to_sample_basic(self):
        """Test basic sample creation with all fields."""
        pre = SimpleQAVerifiedPreprocessor()
        record = {
            "problem": "In which year did Melbourne's Monash Gallery of Art rebrand?",
            "answer": "2023",
            "topic": "Art",
            "answer_type": "Date",
            "multi_step": False,
            "requires_reasoning": False,
            "urls": "https://example.com/url1",
            "original_index": 13,
        }
        sample = pre.to_sample(record)
        
        # Check that it's a Sample dataclass
        self.assertIsNotNone(sample)
        self.assertTrue(is_dataclass(sample))
        
        # Check ID and schema version
        self.assertIsNotNone(sample.id)
        self.assertIsNotNone(sample.schema_version)
        
        # Check messages
        self.assertIsNotNone(sample.messages)
        self.assertEqual(len(sample.messages), 1)  # Only user message (no system prompt)
        self.assertEqual(sample.messages[0].role, "user")
        self.assertIn(record["problem"], sample.messages[0].content[0].text)
        
        # Check references and label
        self.assertIsNotNone(sample.references)
        self.assertEqual(len(sample.references), 1)
        self.assertEqual(sample.references[0], record["answer"])
        self.assertEqual(sample.label, record["answer"])
        
        # Check metadata
        self.assertIsNotNone(sample.metadata)
        self.assertEqual(sample.metadata["topic"], "Art")
        self.assertEqual(sample.metadata["answer_type"], "Date")
        self.assertEqual(sample.metadata["multi_step"], False)
        self.assertEqual(sample.metadata["requires_reasoning"], False)
        self.assertEqual(sample.metadata["urls"], "https://example.com/url1")
        self.assertEqual(sample.metadata["original_index"], 13)


if __name__ == "__main__":
    unittest.main()
