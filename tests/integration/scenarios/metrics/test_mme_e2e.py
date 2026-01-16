"""End-to-end integration tests for MME preprocessor and metric."""

import sys
from pathlib import Path
import unittest
from PIL import Image
import numpy as np

ROOT = Path(__file__).resolve().parents[3] / "src"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from gage_eval.config.pipeline_config import PipelineConfig
from gage_eval.metrics import MetricRegistry, MetricContext
from gage_eval.observability.trace import ObservabilityTrace
from gage_eval.assets.datasets.preprocessors.mme_preprocessor import MMEPreprocessor


def generate_random_image(width, height):
    """Generate a random PIL Image for testing."""
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data, 'RGB')


class MMEE2ETests(unittest.TestCase):
    """End-to-end tests for MME preprocessor and metric."""

    def test_mme_preprocessor_and_metric_pipeline(self):
        """Test MME preprocessor + metric pipeline."""
        # STEP 1: Setup preprocessor
        preprocessor = MMEPreprocessor()
        
        # STEP 2: Preprocess samples
        records = [
            {
                "question_id": "test_001",
                "question": "Is a python code shown in the picture?",
                "answer": "Yes",
                "category": "code_reasoning",
                "decoded_image": generate_random_image(32, 32),
            },
            {
                "question_id": "test_002",
                "question": "Is a c++ code shown in the picture?",
                "answer": "No",
                "category": "code_reasoning",
                "decoded_image": generate_random_image(32, 32),
            },
        ]
        
        samples = []
        for record in records:
            sample = preprocessor.transform(record)
            samples.append(sample)
        
        self.assertEqual(len(samples), 2)
        
        # STEP 3: Setup metric
        config_dict = {
            "metrics": [
                {"metric_id": "mme_acc", "implementation": "mme_accuracy"},
            ]
        }
        pipeline_config = PipelineConfig.from_dict(
            config_dict | {
                "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
                "role_adapters": [{"adapter_id": "r1", "role_type": "dut_model"}],
                "custom": {"steps": [{"step": "auto_eval"}]}
            }
        )
        registry = MetricRegistry()
        metric_instance = registry.build_metric(pipeline_config.metrics[0])
        
        # STEP 4: Evaluate samples with model outputs
        trace = ObservabilityTrace()
        results = []
        
        # Sample 1: Correct prediction
        sample_dict_1 = {
            "id": samples[0].id,
            "label": samples[0].label,
            "references": samples[0].references,
            "metadata": samples[0].metadata,
            "predict_result": [{
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Yes"}]
                }
            }],
        }
        ctx_1 = MetricContext(
            sample_id=str(sample_dict_1["id"]),
            sample=sample_dict_1,
            model_output={"answer": "Yes"},
            judge_output={},
            args=metric_instance.spec.params,
            trace=trace,
        )
        res_1 = metric_instance.evaluate(ctx_1)
        results.append(res_1)
        
        # Sample 2: Incorrect prediction
        sample_dict_2 = {
            "id": samples[1].id,
            "label": samples[1].label,
            "references": samples[1].references,
            "metadata": samples[1].metadata,
            "predict_result": [{
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Yes"}]
                }
            }],
        }
        ctx_2 = MetricContext(
            sample_id=str(sample_dict_2["id"]),
            sample=sample_dict_2,
            model_output={"answer": "Yes"},  # Wrong: should be "No"
            judge_output={},
            args=metric_instance.spec.params,
            trace=trace,
        )
        res_2 = metric_instance.evaluate(ctx_2)
        results.append(res_2)
        
        # STEP 5: Verify results
        self.assertEqual(res_1.values["acc"], 1.0)
        self.assertEqual(res_2.values["acc"], 0.0)
        
        # STEP 6: Aggregate and verify
        aggregated = metric_instance.finalize()
        self.assertIn("metric_id", aggregated)
        self.assertEqual(aggregated["metric_id"], "mme_acc")
        # Mean accuracy: (1.0 + 0.0) / 2 = 0.5
        self.assertAlmostEqual(aggregated["values"]["acc"], 0.5, places=6)

    def test_mme_metric_with_various_formats(self):
        """Test MME metric handles various prediction formats."""
        registry = MetricRegistry()
        config_dict = {
            "metrics": [
                {"metric_id": "mme_acc", "implementation": "mme_accuracy"},
            ]
        }
        pipeline_config = PipelineConfig.from_dict(
            config_dict | {
                "datasets": [{"dataset_id": "d1", "loader": "dummy"}],
                "role_adapters": [{"adapter_id": "r1", "role_type": "dut_model"}],
                "custom": {"steps": [{"step": "auto_eval"}]}
            }
        )
        metric_instance = registry.build_metric(pipeline_config.metrics[0])
        
        test_cases = [
            ("Yes", "Yes", 1.0),
            ("yes", "Yes", 1.0),
            ("No", "No", 1.0),
            ("no", "No", 1.0),
            ("Yes", "No", 0.0),
            ("No", "Yes", 0.0),
        ]
        
        trace = ObservabilityTrace()
        for prediction, reference, expected_score in test_cases:
            sample_dict = {
                "id": "test",
                "label": reference,
                "references": [reference],
                "predict_result": [{
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": prediction}]
                    }
                }],
            }
            ctx = MetricContext(
                sample_id="test",
                sample=sample_dict,
                model_output={"answer": prediction},
                judge_output={},
                args=metric_instance.spec.params,
                trace=trace,
            )
            result = metric_instance.evaluate(ctx)
            self.assertEqual(
                result.values["acc"],
                expected_score,
                f"Failed for prediction='{prediction}', reference='{reference}'"
            )


if __name__ == "__main__":
    unittest.main()
