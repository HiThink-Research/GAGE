import json
import numpy as np
import re
import logging
from typing import List, Dict, Optional
from mathruler.grader import grade_answer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def process_response(response: str, choices: Optional[List[str]]) -> str:
    if choices is not None:
        try:
            response_index = ['A', 'B', 'C', 'D', 'E', 'F', 'G'].index(response)
            response = choices[response_index]
            return response
        except:
            pass
    return response

def process_line(line):
    correct_flag = 0
    answer = line["choices"][0]["message"]['content'][0]['text']
    response = line['predict_result']
    question = line["messages"][0]['content'][1]['text']
    answer_choices = line['answer_choices']
    if response:
        match = re.search(r'\\boxed\{(.+?)\}', response)
        response = match.group(1) if match else response
        response= response.strip()
        logger.info(f"Generated answer: {response}")
        processed_response = process_response(response, answer_choices)
        if processed_response.lower() == answer.lower() or grade_answer(processed_response, answer):
            correct_flag = 1
    else:
        response = "Failed to generate."
        logger.warning(f"Failed to generate answer for question: {question} ")
    line['correct_flag'] = correct_flag
    return line
            

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



def evaluation(eval_file, num_processes=1):
    """评估函数"""
    data = load_jsonl(eval_file)
    results = []
    for item in data:
        result = process_line(item)
        results.append(result)

    result_jsonl = eval_file.replace('.jsonl', '_with_results.jsonl')
    dump_jsonl(results, result_jsonl)
    
    correct_flags = [x['correct_flag'] for x in results]
    correct_num = np.sum(correct_flags)
    accuracy = correct_num / len(data)

    return {'acc': accuracy}

