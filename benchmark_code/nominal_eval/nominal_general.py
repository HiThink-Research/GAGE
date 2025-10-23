import re
import json

import sys, os
PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "PolyhedronEvaluator") 
if PATH not in sys.path:
    sys.path.append(PATH)

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
        if isinstance(d['predict_result'], str) and isinstance(d['answer'], str):
            if d['predict_result'].lower() == d['answer'].lower():
                passed += 1
                d['eval_result'] = {"result":"True"}
            else:
                d['eval_result'] = {"result":"False"}
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
    pass
