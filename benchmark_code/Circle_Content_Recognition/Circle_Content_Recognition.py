import json


def Ans_Extraction(output):
    if '是' in output and '否' not in output:
        return '是'
    if '是' not in output and '否' in output:
        return '否'
    return '未知'


def evaluation(input_path, **kwargs):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    out = []
    
    total = 0
    correct = 0
    tp = 0
    fp = 0
    fn = 0

    for d in data:
        pred = Ans_Extraction(d['predict_result'])
        true_label = d["choices"][0]["message"]["content"][0]['text']
        total += 1

        if pred == true_label:
            correct += 1
            d['eval_result'] = {"result": "True"}
        else:
            d['eval_result'] = {"result": "False"}

        # Assume "是" is the positive label.
        if pred == "是":
            if true_label == "是":
                tp += 1
            else:
                fp += 1
        elif pred == "否":
            if true_label == "是":
                fn += 1
        out.append(d)
    
    with open(input_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')

    acc = correct / total if total > 0 else 0
    f1 = (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0

    return {"acc": acc, "f1_score": f1}


if __name__ == "__main__":
    evaluation("QuanxuanJudge.jsonl")
