import unittest
from gage_eval.assets.datasets.preprocessors.math500_preprocessor import Math500Preprocessor

class TestMath500Preprocessor(unittest.TestCase):
    def setUp(self):
        self.processor = Math500Preprocessor()

    def test_to_sample_basic(self):
        record = {
            "problem": "Calculate 1+1",
            "answer": "2",
            "subject": "Math",
            "level": 1,
            "unique_id": "test/1",
            "solution": "1+1=2"
        }
        sample = self.processor.to_sample(record)
        
        self.assertEqual(sample["_dataset_id"], self.processor.dataset_id)
        self.assertEqual(sample["question"], "Calculate 1+1")
        self.assertEqual(sample["answers"], ["2"])
        self.assertEqual(sample["label"], "2")
        self.assertEqual(len(sample["messages"]), 2)
        self.assertEqual(sample["messages"][0]["role"], "system")
        self.assertIn("Please solve", sample["messages"][0]["content"][0]["text"])
        self.assertEqual(sample["messages"][1]["role"], "user")
        self.assertEqual(sample["messages"][1]["content"][0]["text"], "Calculate 1+1")
        
        # Metadata check
        self.assertEqual(sample["metadata"]["subject"], "Math")
        self.assertEqual(sample["metadata"]["level"], 1)
        self.assertEqual(sample["metadata"]["unique_id"], "test/1")
        self.assertEqual(sample["metadata"]["solution"], "1+1=2")

    def test_missing_fields(self):
        record = {"problem": "Q"}
        with self.assertRaises(ValueError):
            self.processor.to_sample(record) # Missing answer

        record = {"answer": "A"}
        with self.assertRaises(ValueError):
            self.processor.to_sample(record) # Missing problem

    def test_custom_fields(self):
        processor = Math500Preprocessor(question_field="q", answers_field="a", content_field="q")
        record = {"q": "Q", "a": "A"}
        sample = processor.to_sample(record)
        self.assertEqual(sample["question"], "Q")
        self.assertEqual(sample["label"], "A")

if __name__ == "__main__":
    unittest.main()
