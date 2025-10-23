import json
# from rouge import Rouge
from tqdm import tqdm
# import Levenshtein
import os
import requests, random
import re
import sys
import numpy as np


def extract_rating(llm_output):
    """
    Extracts the rating in the format [[number]] from the LLM output.

    Args:
    - llm_output (str): The response from the LLM containing the evaluation and rating.

    Returns:
    - int: The extracted rating, or None if the rating is not found.
    """
    # Define the regular expression pattern to match the rating in the format [[number]]
    pattern = r"\[\[(\d+)\]\]"

    # Search for the pattern in the LLM output
    match = re.search(pattern, llm_output)

    if match:
        # Convert the matched rating to an integer and return it
        return int(match.group(1))
    else:
        # Return None if no rating is found
        # return None
        raise NotImplementedError

def llm_score(file_path):
    judgement_result = list()
    line_info = {}
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            line = json.loads(line.strip())
            predict_result = line["predict_result"]
            try:
                score = float(predict_result)
            except:
                score = extract_rating(predict_result)
            judgement_result.append(score)
    
    print("{} 成功评估测试集条数： {}".format(file_path, len(judgement_result)))
    head_ = file_path.split('/')[-1].split('.')[0]
    out_file = '/'.join(file_path.split('/')[:-1]) + '/' + head_ + '_evalres.json'
    with open(out_file, "w") as f:
        output = "\n".join([json.dumps(x, ensure_ascii=False) for x in judgement_result])
        f.write(output)
    
    judge_file = '/'.join(file_path.split('/')[:-3]) + '/' + str(head_) +'.jsonl'

    with open(judge_file, "r") as f:
        original_lines = [json.loads(line.strip()) for line in f]
    
    assert len(original_lines) == len(judgement_result) 
    for idx, line_info in enumerate(judgement_result):
        original_lines[idx]["eval_result"] = {"result": line_info}

    with open(judge_file, "w") as f:
        output = "\n".join([json.dumps(x, ensure_ascii=False) for x in original_lines])
        f.write(output)
    
    return np.mean(judgement_result)


def evaluation(file_path):
    out = []
    llm_win_rate = llm_score(file_path)
    return {'llm_score': llm_win_rate}


if __name__ == "__main__":
    # file_path = 'air_jtb_appellation.json'
    file_path = sys.argv[1]
    result = evaluation(file_path)
    print(result)
    head_ = file_path.split('/')[-1].split('.')[0]
    out_file = '/'.join(file_path.split('/')[:-1]) + '/' + 'stastic.json'
    with open(out_file, 'a', encoding='utf8') as fw:
        json.dump({head_: result}, fw, ensure_ascii=False)
        fw.write('\n')

    out_file = '/'.join(file_path.split('/')[:-2]) + '/' + str(head_) +'.log'
    # print(out_file)
    with open(out_file, 'w', encoding='utf8') as fw:
        json.dump([{"Average": result}] ,fw, ensure_ascii=False)
        fw.write('\n')