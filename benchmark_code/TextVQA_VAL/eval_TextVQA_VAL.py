#coding:utf-8

import json
import numpy as np
import ast

def istype(s, type_):
    if isinstance(s, type_):
        return True
    try:
        return isinstance(eval(s), type_)
    except Exception:
        return False

def process_answer(answer):
    """标准化答案格式"""
    if isinstance(answer, list):
        answer = ' '.join(str(item) for item in answer)
    answer = str(answer).replace('\n', ' ').replace('\t', ' ').strip()
    answer = process_punctuation(answer)
    answer = _process_digit_article(answer)
    return answer

def _process_digit_article(in_text):
    out_text = []
    temp_text = in_text.lower().split()
    articles = ['a', 'an', 'the']
    manual_map = {
        'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3',
        'four': '4', 'five': '5', 'six': '6', 'seven': '7', 'eight': '8',
        'nine': '9', 'ten': '10'
    }
    for word in temp_text:
        word = manual_map.get(word, word)
        if word not in articles:
            out_text.append(word)
    return ' '.join(out_text)

def process_punctuation(in_text):
    import re
    punct = [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
    comma_strip = re.compile(r'(\d)(,)(\d)')
    period_strip = re.compile(r'(?!<=\d)(\.)(?!\d)')

    out_text = in_text
    for p in punct:
        if p in out_text:
            out_text = out_text.replace(p, ' ')
    out_text = comma_strip.sub(r'\1\3', out_text)
    out_text = period_strip.sub('', out_text)
    return out_text


def hit_calculate(results):
    return [np.mean(res['match']) for res in results]

def process_line(line):
    """计算分数，匹配到三次以上算1分，两个是2/3分，一个是1/3分"""
    gt_answers_raw = line["choices"][0]["message"]['content'][0]['text']

    if isinstance(gt_answers_raw, list):
        gt_answers = [process_answer(item) for item in ast.literal_eval(gt_answers_raw[0])]
    else:
        gt_answers = [process_answer(item) for item in ast.literal_eval(gt_answers_raw)]
    prec_answer = line.get("predict_result", "")
    prec_answer = re.sub(r'<think>.*?</think>', '', pred_answer, flags=re.DOTALL)

    pred_answer = process_answer(line.get("predict_result", ""))

    exact_matches = sum(1 for gt in gt_answers if gt == pred_answer)

    if exact_matches >= 3:
        match_score = 1.0  
    elif exact_matches == 2:
        match_score = 2/3  
    elif exact_matches == 1:
        match_score = 1/3 
    else:
        match_score = 0.0 

    line['eval_result'] = {
        'gt': gt_answers,
        'pred': pred_answer,
        'match': [match_score],
        'score': match_score * 100
    }
    return line

def evaluation(eval_file):
    """Main evaluation function."""
    with open(eval_file, 'r') as f:
        data = [json.loads(line) for line in f]

    output_file = eval_file.replace('.jsonl', '_result.jsonl')
    with open(output_file, 'w') as outf:
        results = []
        for line in data:
            processed_line = process_line(line)
            results.append(processed_line['eval_result']['score'])
            json.dump(processed_line, outf)
            outf.write('\n')
    overall_score = np.mean(results)
    print(f"Overall Score: {overall_score:.2f}%")
    return {'acc': overall_score/100}
