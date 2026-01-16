"""Metric aggregator implementations."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

from loguru import logger
from gage_eval.config.pipeline_config import MetricSpec
from gage_eval.metrics.base import AggregatedMetric, MetricResult


class MetricAggregator:
    """Base class for all aggregators."""

    def __init__(self, spec: MetricSpec) -> None:
        self.spec = spec

    def add(self, result: MetricResult) -> None:  # pragma: no cover
        raise NotImplementedError

    def finalize(self) -> AggregatedMetric:  # pragma: no cover
        raise NotImplementedError


class MeanAggregator(MetricAggregator):
    """Computes the mean for each metric value key."""

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._sums: Dict[str, float] = defaultdict(float)
        self._counts: Dict[str, int] = defaultdict(int)
        self._total_results = 0

    def add(self, result: MetricResult) -> None:
        for key, value in result.values.items():
            self._sums[key] += float(value)
            self._counts[key] += 1
        self._total_results += 1

    def finalize(self) -> AggregatedMetric:
        values = {
            key: (self._sums[key] / self._counts[key]) if self._counts[key] else 0.0
            for key in self._sums
        }
        logger.debug(
            "MeanAggregator finalized metric={} samples={}",
            self.spec.metric_id,
            self._total_results,
        )
        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "mean",
            values=values,
            count=self._total_results,
        )


class WeightedMeanAggregator(MetricAggregator):
    """Computes a weighted mean for each value key.

    The weight is read from `MetricResult.metadata["weight"]` (defaults to 1.0).
    """

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._weighted_sums: Dict[str, float] = defaultdict(float)
        self._weight_totals: Dict[str, float] = defaultdict(float)
        self._total_samples = 0

    def add(self, result: MetricResult) -> None:
        weight = float(result.metadata.get("weight", 1.0))
        for key, value in result.values.items():
            self._weighted_sums[key] += float(value) * weight
            self._weight_totals[key] += weight
        self._total_samples += 1

    def finalize(self) -> AggregatedMetric:
        values = {
            key: (self._weighted_sums[key] / self._weight_totals[key]) if self._weight_totals[key] else 0.0
            for key in self._weighted_sums
        }
        logger.debug(
            "WeightedMeanAggregator finalized metric={} samples={}",
            self.spec.metric_id,
            self._total_samples,
        )
        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "weighted_mean",
            values=values,
            count=self._total_samples,
        )

class IdentityAggregator(MetricAggregator):
    """Returns all per-sample values without aggregation.

    This is mostly useful for debugging or custom aggregations at higher layers.
    """

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._results: List[MetricResult] = []

    def add(self, result: MetricResult) -> None:
        self._results.append(result)

    def finalize(self) -> AggregatedMetric:
        values = {str(idx): res.values for idx, res in enumerate(self._results)}
        logger.debug("IdentityAggregator captured {} samples for metric={}", len(self._results), self.spec.metric_id)
        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "identity",
            values=values,
            count=len(self._results),
            metadata={"samples": [res.to_dict() for res in self._results]},
        )

class OmniDocLazyCalcAggregator(MetricAggregator):
    """Returns all OmniDocBench values by calling its toolkit.

    This is mostly useful for direct calc the coupled metrics.
    """

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._results: List[MetricResult] = []

    def add(self, result: MetricResult) -> None:
        self._results.append(result)

    def finalize(self) -> AggregatedMetric:
        import os,shutil
        omnidoc_home = os.getenv("OMNIDOCBENCH_HOME")
        if not omnidoc_home:
            raise EnvironmentError(
                "Environment variable 'OMNIDOCBENCH_HOME' is not set. "
                "Please set it to the root directory of the OmniDocBench"
                "using 'export OMNIDOCBENCH_HOME=/path/to/OmniDocBench-main' in your terminal."
            )
        write_folder = os.path.join(omnidoc_home, 'prediction', 'gage_run')
        if os.path.exists(write_folder):
            shutil.rmtree(write_folder)
        os.makedirs(write_folder, exist_ok=True) # exist_ok for FS latency

        values = {str(idx): res.to_dict() for idx, res in enumerate(self._results)}
        logger.debug("OmniDocLazyCalcAggregator captured {} samples for metric={}", len(self._results), self.spec.metric_id)
        for idx,value in values.items():
            basename=os.path.basename(value['img_path'])
            write_path=os.path.join(write_folder, f'{basename}.md')
            with open(write_path,'w',encoding='utf8')as f:
                f.write(value['prediction'])
        logger.debug("Now, runing the OmniDocBench Toolkit for full metrics. It may take about 30min+ as CDM renders latex formulas")
        bench_stdout=self.run_pdf_validation(omnidoc_home, write_folder)
        logger.debug("OmniDocBench Toolkit Finished.")
        values = self.get_metric_per_page(values, omnidoc_home, write_folder)
        overall_dic=self.get_bench_overall(omnidoc_home, write_folder)

        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "omnidoclazycalc",
            values=values,
            count=len(self._results),
            metadata={"overall": overall_dic, "samples": [res.to_dict() for res in self._results]},
        )

    def get_metric_per_page(self, values, omnidoc_home, write_folder):
        import os,json
        metric_names=[
                    'quick_match_text_block_per_page_edit',
                    'quick_match_display_formula_per_page_edit',
                    'quick_match_reading_order_per_page_edit',
                    'quick_match_table_per_page_edit',
                    ]
        pr_folder_name=os.path.basename(write_folder)

        metric_per_page_dic={}
        for metric_name in metric_names:
            json_path=os.path.join(omnidoc_home, "result", f"{pr_folder_name}_{metric_name}.json")
            with open(json_path,'r',encoding='utf8')as f:
                metric_data=json.load(f)
            metric_per_page_dic[metric_name]=metric_data
        
        for idx, value in values.items():
            img_basename=os.path.basename(value['img_path'])
            this_sample_metrics={}
            for metric_name in metric_names:
                this_sample_metrics[metric_name]=metric_per_page_dic[metric_name].get(img_basename)
            values[idx]['metrics']=this_sample_metrics
        return values

    def get_bench_overall(self, omnidoc_home, write_folder) -> dict:
        import os,csv,subprocess
        import pandas as pd
        pr_folder_name=os.path.basename(write_folder)
        gen_res_table_py=os.path.join(omnidoc_home, "tools/generate_result_tables.py")
        if not os.path.exists(gen_res_table_py):
            raise FileNotFoundError(f"Could not find {gen_res_table_py}. Please modify generate_result_tables.ipynb to generate_result_tables.py within df.to_csv('./overall.csv') ")
        command = ["python", "tools/generate_result_tables.py", pr_folder_name]
        result = subprocess.run(command, cwd=omnidoc_home,capture_output=True,text=True,)
        df = pd.read_csv(os.path.join(omnidoc_home, "overall.csv"), index_col=0)
        overall_dic=df.to_dict(orient='index')
        return overall_dic

    def run_pdf_validation(self, omnidoc_home, write_folder) -> str:
        import os,yaml,subprocess
        gt_json = os.path.join(omnidoc_home, "OmniDocBench.json")
        if not os.path.exists(gt_json):
            raise EnvironmentError(
                "The GT json OmniDocBench.json is missing, please download the dataset(not the toolkit)."
                f" And loacate OmniDocBench.json under OMNIDOCBENCH_HOME={omnidoc_home}"
            )
        config_dict = {
            "end2end_eval": {
                "metrics": {
                    "text_block": {
                        "metric": ["Edit_dist"]
                    },
                    "display_formula": {
                        "metric": ["Edit_dist", "CDM"]
                    },
                    "table": {
                        "metric": ["TEDS", "Edit_dist"]
                    },
                    "reading_order": {
                        "metric": ["Edit_dist"]
                    }
                },
                "dataset": {
                    "dataset_name": "end2end_dataset",
                    "ground_truth": {
                        "data_path": os.path.join(omnidoc_home, "OmniDocBench.json")
                    },
                    "prediction": {
                        "data_path": write_folder
                    },
                    "match_method": "quick_match"
                }
            }
        }


        temp_config_path = os.path.join(omnidoc_home, "configs", "end2end_gage.yaml")
        with open(temp_config_path, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f, allow_unicode=True)

        command = [
            "python", 
            "pdf_validation.py", 
            "--config", temp_config_path
        ]
    
        try:
            result = subprocess.run(
                command,
                cwd=omnidoc_home,
                capture_output=True,
                text=True,
                # check=True
            )
            return result.stdout

        except subprocess.CalledProcessError as e:
            print(f"ERROR code：{e.returncode}")
            print(f"{e.stdout}")
            print(f"{e.stderr}")
            raise


class CategoricalCountAggregator(MetricAggregator):
    """Counts occurrences of a categorical field in per-sample metadata."""

    def __init__(self, spec: MetricSpec) -> None:
        super().__init__(spec)
        self._counts: Dict[str, int] = defaultdict(int)
        self._total = 0
        self._category_field = str(spec.params.get("category_field", "failure_reason"))
        self._include_none = bool(spec.params.get("include_none", False))
        self._none_label = str(spec.params.get("none_label", "unknown"))

    def add(self, result: MetricResult) -> None:
        category = result.metadata.get(self._category_field)
        if category is None:
            if not self._include_none:
                return
            category = self._none_label
        key = str(category)
        self._counts[key] += 1
        self._total += 1

    def finalize(self) -> AggregatedMetric:
        logger.debug(
            "CategoricalCountAggregator finalized metric={} samples={}",
            self.spec.metric_id,
            self._total,
        )
        return AggregatedMetric(
            metric_id=self.spec.metric_id,
            aggregation=self.spec.aggregation or "categorical_count",
            values=dict(self._counts),
            count=self._total,
        )
