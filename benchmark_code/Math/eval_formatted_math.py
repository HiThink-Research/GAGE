import re
import json

# import sys, os
# # 获取 `utils` 目录的绝对路径，并添加到 sys.path
# sys.path.append(os.path.abspath(os.path.dirname(__file__)))

# from parser.grader import check_is_correct
# from parser.parser import extract_answer

import sys, os
PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "PolyhedronEvaluator") 
if PATH not in sys.path:
    sys.path.append(PATH)

from extract_answer import extract_answer
from eval_numeral import numeral_equal

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
    extracted_answer = extract_answer(model_pred, last=True)
    
    # Check if the extracted answer matches the gold answer
    # return extracted_answer, check_is_correct(extracted_answer, gold_answer)
    return extracted_answer, numeral_equal(extracted_answer, gold_answer)
    
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
            passed += 1
            d['eval_result'] = {"result":"True"}
        else:
            d['eval_result'] = {"result":"False"}
        out.append(d)
    
    with open(input_path,'w',encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o,ensure_ascii=False)+'\n')
    precision_dict = {}
    precision_dict["acc"] = passed/count
    return precision_dict



if __name__ == "__main__":
    print(evaluate_prediction("The answer is \\boxed{答案是二十五}", "25"))