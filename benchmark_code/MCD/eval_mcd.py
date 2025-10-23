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


def process_item(item):
    correct_cell_num = 0
    pred_cell_num = 0
    gold_cell_num = 0

    model_output = item["predict_result"].replace('\n',' ')
    gold_answer_list = item["answer_list"]
    gold_cell_num += len(gold_answer_list)
    merged_cell_region_list = []
    if gold_answer_list == ['None']: # There are no merged cells in the table
        # Different models may use different sentences to express 'there is no merged cell'.
        # In such cases, you need to include more expressions for a more accurate evaluation.
        if "does not contain any merged cells" in model_output.lower() or "no merged cell" in model_output.lower():
            correct_cell_num += 1
            pred_cell_num += 1
        else:
            pred_cell_num += 1
    else:  # There are merged cells in the table
        # parse the ground truth coordinates of merged cells
        for answer_str in gold_answer_list:
            gold_answer_item = eval(answer_str)
            top_row_id, left_col_id = gold_answer_item['top-left']
            bottom_row_id, right_col_id = gold_answer_item['bottom-right']
            gold_merged_region_repr = f"{top_row_id}_{left_col_id}_{bottom_row_id}_{right_col_id}"
            merged_cell_region_list.append(gold_merged_region_repr)
        # parse the predicted coordinates of merged cells
        pred_answer_str_list = re.findall('{[\"\']top-left[\"\']\:.*?,\s?[\"\']bottom-right[\"\']\:.*?}', model_output)
        print("*"*100)
        print(merged_cell_region_list)
        print(pred_answer_str_list)
        for answer_str in pred_answer_str_list:
            try:
                pred_answer_item = eval(answer_str)
                top_row_id, left_col_id = pred_answer_item['top-left']
                bottom_row_id, right_col_id = pred_answer_item['bottom-right']
                pred_merged_region_repr = f"{top_row_id}_{left_col_id}_{bottom_row_id}_{right_col_id}"
            except Exception as e:
                print(f"解析错误: {e}")
                continue
            
            if pred_merged_region_repr in merged_cell_region_list:
                correct_cell_num += 1
        pred_cell_num += len(pred_answer_str_list)
        
    item["correct_cell_num"] = correct_cell_num
    item["pred_cell_num"] = pred_cell_num
    item["gold_cell_num"] = gold_cell_num
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
    """Evaluation for merged cell detection (MCD) benchmark."""
    data = load_jsonl(eval_file)
    results = []
    pred_cell_num = 0 # number of predicted merged cells 
    gold_cell_num = 0 # number of gold merged cells
    correct_cell_num = 0 # number of predicted merged cells which are correct
    for item in data:
        result = process_item(item)
        results.append(result)

    result_jsonl = eval_file.replace(".jsonl", "_with_results.jsonl")
    dump_jsonl(results, result_jsonl)
    correct_cell_num =  np.sum(x["correct_cell_num"] for x in results)
    pred_cell_num = np.sum(x["pred_cell_num"] for x in results)
    gold_cell_num = np.sum(x["gold_cell_num"] for x in results)
    P = correct_cell_num / pred_cell_num
    R = correct_cell_num / gold_cell_num
    print("Precision:",P)
    print("Recall:",R)
    if P+R == 0:
        F1 = 0
    else:
        F1 = 2*P*R/(P+R)
    print("F1 score:",F1)

    return {"f1_score": F1}

