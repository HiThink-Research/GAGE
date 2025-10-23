import json
import numpy as np
import re
import logging
from typing import List, Dict, Optional


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("evaluation.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def extract_tqa_answer_list(model_output):
    """
    Extract the answer list from the model output to compute accuracy
    """
    model_output = model_output.replace("\n", " ")
    ret = re.match(".*({[\"']answer[\"']\:.*}).*", model_output)
    if ret is not None:
        answer_str = ret.group(1)
        try:
            answer_str = re.sub("[\"']+", '"', answer_str)
            answer_item = eval(answer_str)
            predicted_answer = answer_item["answer"]
            if type(predicted_answer) != list and type(predicted_answer) == str:
                predicted_answer = [predicted_answer]
            elif type(predicted_answer) != list and type(predicted_answer) in [
                float,
                int,
            ]:
                predicted_answer = [str(predicted_answer)]
            else:
                pass
        # The answer is considered to be wrong if we can not extract answer list from the json str
        except:
            predicted_answer = []
        return predicted_answer
    else:
        return []


def process_item(item):
    correct_flag = 0
    try:
        model_output = item["predict_result"]
        # parse the predicted answer list
        predicted_answer_list = extract_tqa_answer_list(model_output)
        gold_answer_list = item["answer_list"]
        # Sometimes the order of multiple answer text is not necessarily same as the gold answer,
        # so we convert the answer list to a set for comparison
        if set(gold_answer_list) == set(predicted_answer_list):
            correct_flag = 1
    except Exception:
        correct_flag = 0
    item["correct_flag"] = correct_flag
    return item


def load_jsonl(file_path):
    """加载 jsonl 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def dump_jsonl(data, file_path):
    """保存结果到JSONL文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def evaluation(eval_file):
    """评估函数"""
    data = load_jsonl(eval_file)
    results = []
    for item in data:
        result = process_item(item)
        results.append(result)

    result_jsonl = eval_file.replace(".jsonl", "_with_results.jsonl")
    dump_jsonl(results, result_jsonl)

    correct_flags = [x["correct_flag"] for x in results]
    correct_num = np.sum(correct_flags)
    accuracy = correct_num / len(data)

    return {"acc": accuracy}

