import json
import re
import sacrebleu

def evaluation(input_path, **kwargs):
    """
    评估系统答案和参考答案间的bleu值。
    
    Args:
        input_path: 待评估的文件
        language：语言，zh（中文）、13a（英文）、ja（粤语）
        
    Returns:
        score: wer or cer
    """
    corrects = []
    with open(input_path,'r',encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    count = 0
    out = []
    refs, hyps = [], []
    for d in data:
        hyps.append(d['predict_result'])
        refs.append(d['messages'][-1]['content'][0]['text'])
    bleu = sacrebleu.corpus_bleu(hyps,[refs], tokenize=kwargs['language']).score
    with open(input_path,'w',encoding='utf-8') as f:
        for idx, o in enumerate(data):
            if 'id' not in o:
                o['id'] = idx
            # o['eval_result'] = {"result": distance_list[idx]}
            f.write(json.dumps(o,ensure_ascii=False)+'\n')
    return {'bleu': bleu}

if __name__ == "__main__":
    pass
