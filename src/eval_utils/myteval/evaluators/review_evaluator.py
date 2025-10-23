from collections import defaultdict

from src.eval_utils.myteval.utils.template import parse_string
from src.eval_utils.myteval.schema import ResponseDataSample
import ast
import numpy as np
from sentence_transformers import SentenceTransformer, util
from src.eval_utils.myteval.utils.format_load import format_load


class MyReviewEvaluator:
    """Review Capability Evaluation

    Args:
        data_list(list[dict]): the list of predicted data needed to be evaluated.

    """

    def __init__(
        self,
        data_list: list,
        # bert_score_model: str = "all-mpnet-base-v2",
        **kwargs,
    ) -> None:
        self.data_list = data_list
        # self.bert_score_model = bert_score_model
        # self.sentence_model = SentenceTransformer(self.bert_score_model)

    def _load_dataset(self):
        self.dataset = []

        for datum in self.data_list:
            data_sample = self._process_response(datum)

            self.dataset.append(
                dict(
                    origin_prompt=datum["messages"][:-1],
                    response_data_sample=data_sample,
                )
            )
        self.num_samples = len(self.dataset)

    def _process_response(
        self,
        datum: dict,
    ) -> ResponseDataSample:
        """Process the response to needed format.

        Args:
            datum(dict): inputs.

        Returns:
            dict: Processed response data sample.
        """

        template = datum["template"]
        # Generated response.
        if "predict_result" in datum:
            pred_data = datum["predict_result"]
        elif "prediction" in datum:
            pred_data = datum["prediction"]

        # Response of ground truth.
        assert (
            datum["messages"][-1]["role"] == "assistant"
        ), "messages的最后一个role不是assistant, 数据有问题"
        gt_data = datum["messages"][-1]["content"]["answer"]
        meta_data = datum["meta_data"]

        if meta_data["response_format"] == "json":
            pred_data = self.json_format_parse(pred_data)
        else:
            pred_data = pred_data[pred_data.find(":") + 1 :]
            pred_data = pred_data.strip()
            if len(pred_data) > 0 and pred_data[0] in ["A", "B", "C", "D", "E"]:
                pred_data = pred_data[0]
            else:
                pred_data = None

        return ResponseDataSample(
            template=template, pred=pred_data, gt=gt_data, meta_data=meta_data
        )

    def _evaluate(self, data_sample) -> dict:
        metrics_result = dict(
            parse_rate=0,
            review_quality=0,
        )

        pred_data = data_sample.pred
        if pred_data is not None:
            # import pdb; pdb.set_trace()
            metrics_result["review_quality"] = (
                1.0 if pred_data == data_sample.gt else 0.0
            )
            metrics_result["parse_rate"] = 1.0
        return metrics_result

    # def compute_sen_similarity(self, gt, pred):
    #     gt_embed = self.sentence_model.encode(gt, convert_to_tensor=True)
    #     pred_embed = self.sentence_model.encode(pred, convert_to_tensor=True)
    #     sen_sim = max(0, util.cos_sim(gt_embed, pred_embed).item())
    #     return sen_sim

    def json_format_parse(self, pred_data):
        try:
            data = format_load(pred_data)
        except Exception as e:
            return None
        try:
            new_data = dict()
            new_data["review"] = data["is_finished"]
            assert new_data["review"] in [True, False]
        except Exception as e:
            return None
        return new_data

    def evaluate(self):
        self._load_dataset()
        results_list = []
        for data_sample in self.dataset:
            metrics_result = self._evaluate(data_sample["response_data_sample"])
            results_list.append(metrics_result)
        return self._post_process(results_list)

    def _post_process(self, results_list):
        # list of dict to dict of list
        results_dict = defaultdict(list)
        {results_dict[key].append(sub[key]) for sub in results_list for key in sub}
        metric_list = ["parse_rate", "review_quality"]
        for metric in metric_list:
            results_dict[metric] = np.round(np.mean(results_dict[metric]), decimals=4)
        return results_dict
