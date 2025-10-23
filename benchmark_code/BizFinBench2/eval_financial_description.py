import json
import re
from rouge import Rouge
import numpy as np

def evaluation(input_path, **kwargs):
    total_TP = 0
    total_FP = 0
    total_FN = 0
    rouge_scores = []
    rouge = Rouge()
    
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            
            # Extract predict_result
            predict_result_str = data.get("predict_result", "")
            # Extract JSON from predict_result using regex
            json_match = re.search(r'\[\s*\{.*?\}\s*\]', predict_result_str, re.DOTALL)
            if json_match:
                predict_json_str = json_match.group()
            else:
                # If no JSON found, try to extract from the end after </think>
                parts = predict_result_str.split('</think>')
                if len(parts) > 1:
                    json_match = re.search(r'\[\s*\{.*?\}\s*\]', parts[-1], re.DOTALL)
                    if json_match:
                        predict_json_str = json_match.group()
                    else:
                        predict_json_str = None
                else:
                    predict_json_str = None
            
            if predict_json_str is None:
                # If still not found, skip this sample or treat as empty list
                predict_list = []
            else:
                try:
                    predict_list = json.loads(predict_json_str)
                except json.JSONDecodeError:
                    predict_list = []
            
            # Extract ground truth from choices
            choices = data.get("choices", [])
            if choices and len(choices) > 0:
                content = choices[0].get("message", {}).get("content", [])
                if content and len(content) > 0:
                    ground_truth_str = content[0].get("text", "")
                    try:
                        ground_truth_list = json.loads(ground_truth_str)
                    except json.JSONDecodeError:
                        ground_truth_list = []
                else:
                    ground_truth_list = []
            else:
                ground_truth_list = []
            
            # Create dictionaries for easy lookup
            predict_dict = {item["answer"]: item["cot"] for item in predict_list if "answer" in item and "cot" in item}
            truth_dict = {item["answer"]: item["cot"] for item in ground_truth_list if "answer" in item and "cot" in item}
            
            # Calculate TP, FP, FN for this sample
            predict_ids = set(predict_dict.keys())
            truth_ids = set(truth_dict.keys())
            
            TP = len(predict_ids & truth_ids)
            FP = len(predict_ids - truth_ids)
            FN = len(truth_ids - predict_ids)
            
            total_TP += TP
            total_FP += FP
            total_FN += FN
            
            # Calculate ROUGE scores for matching answers
            for answer_id in predict_ids & truth_ids:
                try:
                    scores = rouge.get_scores(predict_dict[answer_id], truth_dict[answer_id])
                    rouge_scores.append(scores[0]['rouge-l']['f'])
                except:
                    # If ROUGE calculation fails, skip this pair
                    pass
    
    # Calculate precision, recall, and F1
    if total_TP + total_FP == 0:
        precision = 0
    else:
        precision = total_TP / (total_TP + total_FP)
    
    if total_TP + total_FN == 0:
        recall = 0
    else:
        recall = total_TP / (total_TP + total_FN)
    
    if precision + recall == 0:
        f1 = 0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    # Calculate average ROUGE score
    if rouge_scores:
        avg_rouge = np.mean(rouge_scores)
    else:
        avg_rouge = 0
    
    return {"acc": f1, "cot_quality": avg_rouge}