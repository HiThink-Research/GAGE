import json


def evaluation(input_path, **kwargs):
    """
    评估生成式选择题的选项是否正确

    Args:
        input_path: 待评估的文件

    Returns:
        score: top_k(召回率)
    """
    with open(input_path,'r',encoding='utf-8') as f:
        data = [json.loads(l) for l in f]

    recalls = []
    for d in data:
        pred = d['predict_result']
        docs = sorted(d['docs'], key=lambda x: x["score"], reverse=True)
        target = set(d["uid"] for d in docs[:3])
        recall_3 = len(target.intersection(pred)) / len(target)
        recalls.append(recall_3)
        d['top_3_docs'] = list(target)
        d['top_3_recall'] = recall_3
        d['eval_result'] = {'result': recall_3}

    avg = sum(recalls) / len(recalls)
    print("recall_3:%.4f" % avg)

    with open(input_path, 'w', encoding='utf-8') as f:
        for idx, o in enumerate(data):
            if 'id' not in o:
                o['id'] = idx
            f.write(json.dumps(o, ensure_ascii=False) + '\n')

    return {'top_k': avg}


if __name__ == "__main__":
    print(evaluation("embedding_candi_retrieval_llm.jsonl"))
