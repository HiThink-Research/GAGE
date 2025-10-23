import json
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
import ast
from typing import Optional
import re


def istype(s, type):
    if isinstance(s, type):
        return True
    try:
        return isinstance(eval(s), type)
    except Exception as _:
        return False
    

# https://github.com/google-research/pix2struct/blob/main/pix2struct/metrics.py#L81
def relaxed_correctness(target: str,
                        prediction: str,
                        max_relative_change: float = 0.05) -> bool:
    """Calculates relaxed correctness.

    The correctness tolerates certain error ratio defined by max_relative_change.
    See https://arxiv.org/pdf/2203.10244.pdf, end of section 5.1:
    “Following Methani et al. (2020), we use a relaxed accuracy measure for the
    numeric answers to allow a minor inaccuracy that may result from the automatic
    data extraction process. We consider an answer to be correct if it is within
    5% of the gold answer. For non-numeric answers, we still need an exact match
    to consider an answer to be correct.”

    Args:
      target: Target string.
      prediction: Predicted string.
      max_relative_change: Maximum relative change.

    Returns:
      Whether the prediction was correct given the specified tolerance.
    """

    def _to_float(text: str) -> Optional[float]:
        try:
            if text.endswith('%'):
                # Convert percentages to floats.
                return float(text.rstrip('%')) / 100.0
            else:
                return float(text)
        except ValueError:
            return None
    prediction = str(prediction)
    target = str(target)
    prediction_float = _to_float(prediction)
    target_float = _to_float(target)
    if prediction_float is not None and target_float:
        relative_change = abs(prediction_float - target_float) / abs(target_float)
        return relative_change <= max_relative_change
    else:
        return prediction.lower() == target.lower()



def process_line(line):
    ret = {}
    answer = line["choices"][0]["message"]['content'][0]['text']
    pred_answer = line['predict_result']
    prec_answer = re.sub(r'<think>.*?</think>', '', pred_answer, flags=re.DOTALL)
    if istype(answer, list):
        answers = eval(answer)
    else:
        answers = [answer]
    ret['gt'] = answers
    ret['pred'] = pred_answer.strip()
    ret['match'] = [relaxed_correctness(ret['pred'], x) for x in ret['gt']]
    ret['line'] = line

    return ret

def load_jsonl(file_path):
    """加载 jsonl 文件"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def dump_jsonl(data, file_path):
    """保存结果到JSONL文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def dump_csv(data, file_path):
    """保存结果到CSV文件"""
    data.to_csv(file_path, index=False)

def d2df(D):
    return pd.DataFrame({x: [D[x]] for x in D})

def hit_calculate(result):
    return [np.max(x['match']) for x in result]


def evaluation(eval_file, num_processes=4):
    """评估函数"""
    data = load_jsonl(eval_file)
    lines = data
    pool = mp.Pool(num_processes)
    res = pool.map(partial(process_line), data)
    pool.close()
    pool.join()
    hit = hit_calculate(res)
    ret = dict()
    ret['Overall'] = np.mean(hit)
    
    if 'split' in data[0]:
        splits = set([item['split']for item in data])
        for sp in splits:
            sub = [r for l, r in zip(lines, res) if l['split'] == sp]
            # print(sub)
            hit = hit_calculate(sub)
            ret[sp] = np.mean(hit) 
    overall_score = ret['Overall']
    ret = d2df(ret)
    ret.round(2)
    result_csv = eval_file.replace('.jsonl', '_acc.csv')
    dump_csv(ret, result_csv)
   
    result_jsonl = eval_file.replace('.jsonl', '_with_results.jsonl')
    dump_jsonl(ret, result_jsonl)

    return {'acc': overall_score}
