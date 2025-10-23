import re
import json
import json_repair

import sys, os
# 获取 `utils` 目录的绝对路径，并添加到 sys.path
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from parser.grader import check_is_correct
from parser.parser import extract_answer

def evaluate_prediction(model_pred: str, gold_answer: str) -> bool:
    """
    Evaluate a model's prediction against a gold answer.
    
    Args:
        model_pred (str): The model's prediction with answer in \boxed{}.
        gold_answer (str): The correct answer to compare against.
    
    Returns:
        bool: True if the prediction matches the gold answer, False otherwise.
    """
    # Extract the answer from model prediction
    # extracted_answer = extract_answer(model_pred)
    try:
        extracted_answer = json_repair.loads(model_pred.strip())
        if 'answer' not in extracted_answer:
            return '', False
        return extracted_answer, extracted_answer['answer'].lower() == json_repair.loads(gold_answer)['answer'].lower()
    except Exception as e:
        return '', False
    
def evaluation(input_path):
    with open(input_path,'r',encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    count = 0
    out = []
    passed = 0
    for d in data:
        count += 1
        text_item = d["choices"][0]["message"]["content"][0]["text"]
        d['answer'] = text_item
        d['extracted_answer'], res = evaluate_prediction(d['predict_result'],d['answer'])
        if res:
            d['eval_result'] = {"result":"True"}
            passed += 1
        else:
            d['eval_result'] = {"result":"False"}
        out.append(d)
    precision_dict = {}
    precision_dict["acc"] = passed/count
    with open(input_path,'w',encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o,ensure_ascii=False)+'\n')
    return precision_dict


if __name__ == "__main__":
    pass