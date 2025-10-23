import json
import re

def evaluation(input_path,**kwargs):
    # import pdb; pdb.set_trace()
    corrects = []
    with open(input_path,'r',encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    count = 0
    out = []
    for d in data:
        try:
            d["predict_result"] = d["predict_result"].split("Q")[0]
        except:
            d["predict_result"] = ''
        text_item = json.loads(d["choices"][0]["message"]["content"][0]["text"])
        d['answer'] = text_item["answer"]
        # extract all numbers from predict_result,check is the last number matches the answer
        pred = [s for s in re.findall(r'-?\d+\.?\d*', d["predict_result"].replace(",", ""))]
        pred = pred[-1] if pred else ""
        pred = pred[:-1] if pred != "" and pred[-1] == "." else pred
        if pred == d['answer']:
            count += 1
            d['eval_result'] = {"result":"True"}
        else:
            d['eval_result'] = {"result":"False"}
        out.append(d)
    
    with open(input_path,'w',encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o,ensure_ascii=False)+'\n')

    return {"acc": count/len(data) if len(data) != 0 else 0}
