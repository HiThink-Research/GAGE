import pandas as pd
import json
from tqdm import tqdm
import jieba
import os
import requests, random
import re
import sys
from nltk.translate.bleu_score import sentence_bleu


url = 'http://101.200.130.142:8181/aime-qwen2/v1/inference'

def qwen_72b_request(info):
    q = """<|im_start|>user
{}<|im_end|>
<|im_start|>assistant
""".format(info)
    d = {
        "prompts":[q],
        "model_name":"aime-hithinkgpt-70b",
        "trace_id":"test_991",
        "return_details":True,
        "generation_params":{"max_new_tokens":512,"do_sample":False,"temperature":1.0}
    }
    try:
        r = requests.post(url, json=d).json()
        return r['data']['responses'][0]['completions'][0]
    except Exception as e:
        return str(e)

def llm_score(file_path):
    judgement_result = list()
    line_info = {}
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            print(idx)
            line = json.loads(line.strip())
            predict_type = line["predict_type"]
            predict_result = line["predict_result"]
            selection_pattern = re.search("\[\[([\s\S])*\]\]", predict_result)
            try:
                if idx % 2 == 0 and selection_pattern:
                    line_info = {'idx': line['idx']}
                    line_info["messages"] = line['M']
                    line_info["output"] = line['groundtruth']
                    line_info["predict_result"] = line['model_predict']
                    line_info["predict_A_content"] = predict_result
                    line_info["predict_A"] = selection_pattern.groups()[0].strip()
                elif idx % 2 == 1 and selection_pattern:
                    assert line_info['idx'] == line['idx']
                    line_info["predict_B_content"] = predict_result
                    line_info["predict_B"] = selection_pattern.groups()[0].strip()
                    judgement_result.append(line_info)
            except:
                print('错误数据id:{}'.format(idx))
                continue
            
    print("{} 成功评估测试集条数： {}".format(file_path, len(judgement_result)))
    head_ = file_path.split('/')[-1].split('.')[0]
    out_file = '/'.join(file_path.split('/')[:-1]) + '/' + head_ + '_evalres.json'
    with open(out_file, "w") as f:
        output = "\n".join([json.dumps(x, ensure_ascii=False) for x in judgement_result])
        f.write(output)
    
    counter_dict = {"win": 0, "tie": 0, "loss": 0}
    for line_info in judgement_result:
        # print(line_info)
        tmp_score = 0
        predict_A = line_info.get("predict_A", "")
        predict_B = line_info.get("predict_B", "")
        if predict_A.lower() == "a":
            tmp_score += 1
        elif predict_A.lower() == "b":   
            tmp_score -= 1  
        
        if predict_B.lower() == "b":
            tmp_score += 1
        elif predict_B.lower() == "a":   
            tmp_score -= 1 
        if  tmp_score > 0:
            counter_dict["win"] += 1
        elif tmp_score == 0:
            counter_dict["tie"] += 1
        else:
            counter_dict["loss"] += 1
    total_num = counter_dict["win"] + counter_dict["tie"] + counter_dict["loss"] 
    win_num = counter_dict["win"]
    tie_num = counter_dict["tie"]
    loss_num = counter_dict["loss"]
    return win_num / float((total_num + 0.00001)), tie_num / float((total_num + 0.00001)), loss_num / float((total_num + 0.00001))


def evaluation(file_path):
    out = []
    llm_win_rate, llm_tie_rate, llm_lose_rate = llm_score(file_path)
    winMlose = llm_win_rate - llm_lose_rate
    return {'llm_win_rate': llm_win_rate, 'llm_tie_rate': llm_tie_rate, 'llm_lose_rate': llm_lose_rate, 'winMlose': winMlose, 'win_rate': llm_win_rate}



if __name__ == "__main__":
    file_path = sys.argv[1]
    result = evaluation(file_path)
    print(result)
    head_ = file_path.split('/')[-1].split('.')[0]
    out_file = '/'.join(file_path.split('/')[:-1]) + '/' + 'stastic.json'
    with open(out_file, 'a', encoding='utf8') as fw:
        json.dump({head_: result}, fw, ensure_ascii=False)
        fw.write('\n')

    out_file = '/'.join(file_path.split('/')[:-2]) + '/' + str(head_) +'.log'
    with open(out_file, 'w', encoding='utf8') as fw:
        json.dump([{"Average": result}] ,fw, ensure_ascii=False)
        fw.write('\n')