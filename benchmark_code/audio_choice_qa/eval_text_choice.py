import json
import re

def extract_answer(text):
    # 正则表达式匹配\boxed{X}格式，其中X是任意字符
    pattern = r'\\boxed\{([A-Za-z])\}'
    
    # 查找所有匹配项
    matches = re.findall(pattern, text)
    
    # 将匹配到的字母转换为大写
    uppercase_matches = [match.upper() for match in matches]
    
    return uppercase_matches

def evaluation(input_path, **kwargs):
    """
    评估音频选择题和问答题
    
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
        pred = d['predict_result'].lower()
        target = d["choices"][0]["message"]["content"][0]['text']
        if target in 'ABCDEF' and len(target) == 1:
            d['true_choice'] = target
        if "true_choice" in d:
            true_choice, true_idx = "", 1000
            true_choice = extract_answer(d['predict_result'])
            if true_choice:
                true_idx = -1
            else:
                true_choice = ""
            match = re.search('answer is ([A-D])', d['predict_result'])
            if not true_choice and match:
                true_choice = match.group()[-1]
                true_idx = -1
            for t in 'ABCDEFG':
                if t not in d['predict_result']:
                    continue
                idx = d['predict_result'].index(t)
                if idx != 0:
                    if d['predict_result'][idx-1].lower() in {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}:
                        continue
                if idx + 1 < len(d['predict_result']):
                    if d['predict_result'][idx+1].lower() in {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z'}:
                        continue
                if idx >= true_idx:
                    continue
                true_choice = t
                true_idx = idx
            if d['true_choice'] == true_choice:
                d['predict_label'] = True
                right += 1
            else:
                d['predict_label'] = False
            if len(true_choice) == 0:
                no_answer += 1
                # print(target, '$$', d['predict_result'])
            continue
        label = False
        if target.lower() in pred:
            right += 1
            label = True
            d['predict_label'] = label
            continue
        try:
            target = eval(target.lower())
            assert type(target) == list
            if any(i.lower() in pred for i in target):
                right += 1
                label = True
        except:
            pass
        d['predict_label'] = label

    with open(input_path,'w',encoding='utf-8') as f:
        for idx, o in enumerate(data):
            if 'id' not in o:
                o['id'] = idx
            o['eval_result'] = {"result": o.get('predict_label', False)}
            f.write(json.dumps(o,ensure_ascii=False)+'\n')
    print(right, no_answer)
    return {'acc': right / len(data)}

if __name__ == "__main__":
    score = evaluation("paas_mmau.jsonl")
    print(score)

