from gage_eval.metrics.base import BaseMetric
from gage_eval.registry import registry

import os, json

@registry.asset(
    "metrics",
    "omnidocbenchallmetric",
    desc="All Metric for OmniDocBench",
    tags=("vision", "prompt", "docparsing"),
    default_aggregation="mean",
)
class OmniDocBenchMetric(BaseMetric):
    """
    OmniDocBenchMetric的评测类
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self, context): 
        # OmniDocBench metrics rely on complex non-Python tools. Since tables and formulas are not present in every sample, the evaluation logic is dummied here.
        # Instead we use the aggregator to call the bench kit.
        result = DummyTextResult(context)
        return result

class DummyTextResult:
    def __init__(self, context):
        self.context = context

    def to_dict(self):
        """
        Converts the results into a dictionary to match the output requirements of the evaluation framework.
        """
        root = self.context.sample['metadata']['content_root']
        names = self.context.sample['metadata']['image_name'][0]
        img_path = os.path.join(root, names)
        return {
            "sample_id": self.context.sample_id,
            "prediction": self.context.model_output['answer'],
            "img_path": img_path,
            "metrics": None
        }

__all__ = ["OmniDocBenchMetric"]