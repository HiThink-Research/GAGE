import json
import os
import pandas as pd
from collections import defaultdict
import copy


def gen_prompt(ques, ans):
    sys_prompt = "You are a helpful and precise assistant for checking the quality of the answer."
    prompt_template = "[Question]\n{question}\n\n[The Start of Assistant's Answer]\n{answer}\n\n[System]\n{criteria}\n\n"
    #criteria = "We would like to request your feedback on the performance of two AI assistants in response to the user question displayed above.\nPlease rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 to 10, where a higher score indicates better overall performance.\nPlease first output a single line containing only two values indicating the scores for Assistant 1 and 2, respectively. The two scores are separated by a space. In the subsequent line, please provide an explanation of your evaluation, avoiding any potential bias and ensuring that the order in which the responses were presented does not affect your judgment."
    criteria = "We would like to request your feedback on the performance of  AI assistants in response to the user question displayed above.\n \
    Please rate the helpfulness, relevance, accuracy, level of details of their responses. Each assistant receives an overall score on a scale of 1 \
    to 10, where a higher score indicates better overall performance.\nYou need to focaus on whether the generated answer meet the instruction of the \
    question,whether the generated answers are divers, conform to investment logic, and are helpful to users' investment decisions.\
    Please output a single line containing only a value indicating the scores for Assistant. Important!!! Just output the score value without explanation. The score is:"
    prompt = prompt_template.format(
        question=ques, answer=ans, criteria=criteria
    )
    return sys_prompt, prompt


def data_preprocess(input_path, input_file, save_dir='step1'):
    output_path = os.path.join(input_path, save_dir)
    os.makedirs(output_path, exist_ok=True)
    currrent_file_name = os.path.basename(input_file)
    out_file = os.path.join(output_path, currrent_file_name)
    
    with open(os.path.join(input_path, input_file), encoding='utf8') as f:
        data = [json.loads(i) for i in f]
    
    with open(out_file, 'w', encoding='utf8') as fw:
        for line_info in data:
            predict_result = line_info.get("predict_result", "")
            query = line_info.get("messages", [{}])[0].get("content", [{}])[0].get("text", "")
            system_prompt, instruction = gen_prompt(query, predict_result)
            
            ret_data = copy.deepcopy(line_info)
            ret_data["messages"][0]["content"][0]["text"] = system_prompt + instruction

            ret_data["model_prompt_tmpl"] = ""
            ret_data["model_prompt_placeholder"] =  []
            ret_data.pop("predict_result")

            json.dump(ret_data, fw, ensure_ascii=False)

            #json.dump({
            #    "messages": [{"role": "user", "content": [{"text": instruction, "type": "text"}]}],
            #    "choices": [{"index": 0, "message": {"role": "assistant", "content": [
            #
            #        {"text": predict_result, "type": "text"}]}}],
            #    "model_prompt_tmpl": "",
            #    "model_prompt_placeholder": []
            #}, fw, ensure_ascii=False)
            fw.write('\n')

def extract_score(pred_str):
    ss_list = pred_str.split("\n")
    for ss in ss_list:
        _ss = ss.strip()
        try:
            score = float(_ss)
            if score > 0 and score < 11:
                return score
        except Exception as e:
            pass 
    return 0.0

def evaluation(input_path, **kwargs):
    """
    计算大模型打分的平均分
    
    Args:
        input_path: 待评估的文件
        


    Returns:
        score: avg_score
    """
    corrects = []
    with open(input_path,'r',encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    total_count = 0
    pos_count = 0
    score_sum = 0.0
    pos_score_list = []
    total_score_list = []
    for d in data:
        pred = d['predict_result']
        try:
            score = extract_score(pred)
            # score = float(pred.strip())
            if score > 0:
                pos_score_list.append(score)
                pos_count += 1
        except Exception as e:
            score = 0.0
        total_score_list.append(score)
        score_sum += score
        total_count += 1
    
    total_avg_score = score_sum / max(total_count, 1.0)
    pos_avg_score = score_sum / max(pos_count, 1.0)

    print("total_avg_score:", total_avg_score)
    print("total_score_list:", total_score_list)
    print("pos_avg_score:", pos_avg_score)
    print("pos_score_list:", pos_score_list)  
    print("pos_count:", pos_count)
    print("total_count:", total_count)  
    return {'win_rate': pos_avg_score}

if __name__ == "__main__":
    pass
