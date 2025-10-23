import json
import numpy as np
import pandas as pd
from functools import partial
import multiprocessing as mp
import ast

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

def levenshtein_distance(s1, s2):
    """计算编辑距离"""
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2 + 1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

def normalize_answer(text):
    """标准化答案格式"""
    text = text.lower().strip()
    text = ' '.join(text.split())
    return text

def anls_score(groundtruth, prediction):
    """计算ANLS分数"""
    gt_answer = normalize_answer(groundtruth)
    pred_answer = normalize_answer(prediction)
    
    dist = levenshtein_distance(gt_answer, pred_answer)
    length = max(len(gt_answer), len(pred_answer))
    
    if length == 0:
        return 1.0
    
    return 1.0 - (float(dist) / float(length))

def process_line(line):
    """处理每一行数据，计算匹配分数并添加评估结果"""
    try:
        gt_answers = ast.literal_eval(line["choices"][0]["message"]['content'][0]['text'])
        pred_answer = line['predict_result']
        prec_answer = re.sub(r'<think>.*?</think>', '', pred_answer, flags=re.DOTALL)
        
        scores = [anls_score(gt, pred_answer) for gt in gt_answers]
        max_score = max(scores)
        
        line['eval_result'] = 'TRUE' if max_score >= 0.5 else 'FALSE'
        
        return {'match': max_score, 'line': line}
    except Exception as e:
        print(f"Error processing line: {e}")
        line['eval_result'] = 'FALSE'
        return {'match': 0.0, 'line': line}



def evaluation(eval_file, num_processes=4):
    """评估函数"""
    data = load_jsonl(eval_file)
    
    pool = mp.Pool(num_processes)
    res = pool.map(partial(process_line), data)
    pool.close()
    pool.join()
    
    hit_scores = [x['match'] for x in res]
    evaluated_data = [x['line'] for x in res]
    
    hit = [score if score >= 0.5 else 0.0 for score in hit_scores]
    ret = {'Overall': np.mean(hit) * 100}
    ret = pd.DataFrame.from_dict(ret, orient='index', columns=['Accuracy'])
    ret = ret.round(2)
    print(ret)
    result_jsonl = eval_file.replace('.jsonl', '_with_results.jsonl')
    dump_jsonl(evaluated_data, result_jsonl)
    
    result_csv = eval_file.replace('.jsonl', '_acc.csv')
    dump_csv(ret, result_csv)
    return {'acc': np.mean(hit)}
