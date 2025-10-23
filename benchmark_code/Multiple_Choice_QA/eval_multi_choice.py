import json
import re

import sys, os
PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "PolyhedronEvaluator") 
if PATH not in sys.path:
    sys.path.append(PATH)

from extract_answer import extract_answer
from eval_multiple_choice import option_equal

def evaluation(input_path, **kwargs):
    """
    评估生成式选择题的选项是否正确
    
    Args:
        input_path: 待评估的文件
        
    Returns:
        score: acc(准确率)
    """
    corrects = []
    with open(input_path,'r',encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    count = 0
    out = []
    right = 0
    no_answer = 0
    for d in data:
        pred = d['predict_result']
        target = d["choices"][0]["message"]["content"][0]['text']
        true_choice = extract_answer(pred, last=True)
        d['extract_answer'] = true_choice
        d['answer'] = target
        
        # if target == true_choice:
        if option_equal(true_choice, target):
            d['predict_label'] = True
            right += 1
        else:
            d['predict_label'] = False
        if not true_choice:
            no_answer += 1

    with open(input_path,'w',encoding='utf-8') as f:
        for idx, o in enumerate(data):
            if 'id' not in o:
                o['id'] = idx
            o['eval_result'] = {"result": o.get('predict_label', False)}
            f.write(json.dumps(o,ensure_ascii=False)+'\n')
    print(right, no_answer)
    return {'acc': right / len(data)}


if __name__ == "__main__":
    print(extract_answer("/boxed{F. xxx}"))
    print(option_equal(extract_answer("/boxed{F. xxx}")[-1], "F"))
    print(option_equal(extract_answer("\\boxed{\\text{A}}")[-1], "a"))