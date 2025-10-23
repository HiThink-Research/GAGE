import pandas as pd
import json
# from rouge import Rouge
from tqdm import tqdm
import jieba
# import Levenshtein
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


# #rouge计算
# def rouge_score(prediction, ground_truth, lang='cn'):
#     # assert type(prediction) == list and type(ground_truth) == list
#     rouge = Rouge()
#     if type(ground_truth) == list:
#         ground_truth = ground_truth[0]
#     if lang == 'cn':
#         prediction = " ".join(list(jieba.cut(prediction, cut_all=False)))
#         ground_truth = " ".join(list(jieba.cut(ground_truth, cut_all=False)))
#     elif lang == 'en':
#         prediction = prediction
#         ground_truth = ground_truth
#     scores = rouge.get_scores(prediction, ground_truth,avg=True)
#     return scores

# def bleu_score(prediction, ground_truth, lang='cn'):
#     weights = [0.25,0.25,0.25,0.25]  # BLEU-1,2,3,4
#     if lang == 'cn':
#         if type(ground_truth) == list:
#             ground_truth = [list(jieba.cut(g, cut_all=False)) for g in ground_truth]
#         else:
#             ground_truth = [list(jieba.cut(ground_truth, cut_all=False))]
#         prediction = list(jieba.cut(prediction, cut_all=False))
#     elif lang == 'en':
#         if type(ground_truth) == list:
#             ground_truth = [g.split(' ') for g in ground_truth]
#         else:
#             ground_truth = [ground_truth.split(' ')]
#         prediction = prediction.split(' ')
#     return sentence_bleu(ground_truth, prediction,weights=weights)


def llm_score(file_path):
    judgement_result = list()
    line_info = {}
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
         
            #print(idx)
            line = json.loads(line.strip())
            
            predict_type = line["predict_type"]
            predict_result = line["predict_result"]

            #print(lir)
            pattern = r'\[\[(.*?)\]\]'
           
            con_result = re.findall(pattern, predict_result)[0]
            selection_pattern = re.search("\[\[([\s\S])*\]\]", predict_result)
            try:
                if idx % 2 == 0 and selection_pattern:
                    line_info = {'idx': line['idx']}
                    line_info["messages"] = line['M']
                    line_info["output"] = line['groundtruth']
                    line_info["predict_result"] = line['model_predict']
                    line_info["first_eval_content"] = predict_result
                    line_info["first_eval"] = con_result
                elif idx % 2 == 1 and selection_pattern:
                    assert line_info['idx'] == line['idx']
                    line_info["second_eval_content"] = predict_result
                    line_info["second_eval"] = con_result
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
    
    #counter_dict = {"win": 0, "tie": 0, "loss": 0}
    counter_dict = {"consistent": 0, "in": 0}
    judge_file = '/'.join(file_path.split('/')[:-3]) + '/' + str(head_) +'.jsonl'

    with open(judge_file, "r") as f:
        original_lines = [json.loads(line.strip()) for line in f]
    
    assert len(original_lines) == len(judgement_result) 
    for idx, line_info in enumerate(judgement_result):
        #print(line_info)
     
        predict_A = line_info.get("first_eval", "")
        predict_B = line_info.get("second_eval", "")
        #print(predict_A)
        if predict_A.lower() == "consistent" and predict_B.lower() == "consistent":
            counter_dict["consistent"] += 1
            original_lines[idx]["eval_result"] = {"result":"consistent"}
        else:
            counter_dict["in"] += 1
            original_lines[idx]["eval_result"] = {"result":"inconsistent"}

    with open(judge_file, "w") as f:
        output = "\n".join([json.dumps(x, ensure_ascii=False) for x in original_lines])
        f.write(output)
    
    total_num = counter_dict["consistent"] + counter_dict["in"]
    win_num = counter_dict["consistent"]
    loss_num = counter_dict["in"]
    #loss_num = counter_dict["loss"]
    return win_num / float((total_num + 0.00001))


# def LevenshteinDistance(prediction, ground_truth):
#     distance = Levenshtein.distance(prediction, ground_truth)
#     return distance


# def JudgeAppellation(ground_truth, appellation):
#     if appellation in ground_truth:
#         return 1
#     else:
#         return 0


def evaluation(file_path):
    out = []
    # with open(file_path, "r") as f:
    #     for idx, line in enumerate(f):
    #         if idx % 2 == 0:
    #             line_info = json.loads(line.strip())
    #             predict_result, output = line_info['model_predict'], line_info['groundtruth']
    #             rouge = rouge_score(predict_result, output)
    #             bleu = bleu_score(predict_result, output)
    #             line_info['rouge'] = rouge['rouge-l']['f']
    #             line_info['bleu'] = bleu
    #             l_distance = LevenshteinDistance(predict_result, output)
    #             line_info["l_distance"] = l_distance
    #             if "appellation" in line_info.keys():
    #                 appellation = JudgeAppellation(line_info['output'], line_info["appellation"])
    #                 line_info["judge_appellation"] = appellation
    #             out.append(line_info)
    # print("{} 测试集条数： {}".format(file_path, len(out)))
    llm_win_rate = llm_score(file_path)

    # llm_win_rate, llm_tie_rate, llm_loss_rate = 0, 0 , 0
    # rouge = [o['rouge'] for o in out]
    # rouge = sum(rouge) / len(rouge)
    # bleu = [o['bleu'] for o in out]
    # bleu = sum(bleu) / len(bleu)
    # l_distance = [o['l_distance'] for o in out]
    # l_distance = sum(l_distance) / len(l_distance)

    # if "judge_appellation" in out[0].keys():
    #     judge_appellation = [o['judge_appellation'] for o in out]
    #     judge_appellation = sum(judge_appellation) / len(judge_appellation)
    #     return {'bleu': bleu, 'rouge': rouge, 'l_distance': l_distance, 'llm_win_rate': llm_win_rate, 'llm_tie_rate': llm_tie_rate, 'llm_loss_rate': llm_loss_rate, 'judge_appellation': judge_appellation}
    #     # return {'bleu': bleu, 'rouge': rouge, 'l_distance': l_distance, 'judge_appellation': judge_appellation}
    # else:
    #     return {'bleu': bleu, 'rouge': rouge, 'l_distance': l_distance,'llm_win_rate': llm_win_rate, 'llm_tie_rate': llm_tie_rate, 'llm_loss_rate': llm_loss_rate}
    return {'precision': llm_win_rate}



if __name__ == "__main__":
    #file_path = '/mnt/data/ycs/chunk_split/llm_judge_chunk_infer_V23/judge/output/TESTSET__中文解析问句改写__4-0-0.jsonl'
    file_path = sys.argv[1]
    result = evaluation(file_path)
    #print(result)
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