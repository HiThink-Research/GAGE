from collections import defaultdict
import pandas as pd
import numpy as np
import json
# from rouge import Rouge
from tqdm import tqdm
import jieba
# import Levenshtein
import os
import requests, random
import re
import csv
import sys
from nltk.translate.bleu_score import sentence_bleu


def load(f, fmt=None):
    def load_json(pth):
        return json.load(open(pth, 'r', encoding='utf-8'))

    def load_jsonl(f):
        lines = open(f, encoding='utf-8').readlines()
        lines = [x.strip() for x in lines]
        if lines[-1] == '':
            lines = lines[:-1]
        data = [json.loads(x) for x in lines]
        return data

    def load_xlsx(f):
        return pd.read_excel(f)

    def load_csv(f):
        return pd.read_csv(f)

    def load_tsv(f):
        return pd.read_csv(f, sep='\t')

    handlers = dict(json=load_json, jsonl=load_jsonl, xlsx=load_xlsx, csv=load_csv, tsv=load_tsv)
    if fmt is not None:
        return handlers[fmt](f)

    suffix = f.split('.')[-1]
    return handlers[suffix](f)

def dump(data, f, **kwargs):
    def dump_pkl(data, pth, **kwargs):
        pickle.dump(data, open(pth, 'wb'))

    def dump_json(data, pth, **kwargs):
        json.dump(data, open(pth, 'w'), indent=4, ensure_ascii=False, cls=NumpyEncoder)

    def dump_jsonl(data, f, **kwargs):
        lines = [json.dumps(x, ensure_ascii=False, cls=NumpyEncoder) for x in data]
        with open(f, 'w', encoding='utf8') as fout:
            fout.write('\n'.join(lines))

    def dump_xlsx(data, f, **kwargs):
        data.to_excel(f, index=False, engine='xlsxwriter')

    def dump_csv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, index=False, encoding='utf-8', quoting=quoting)

    def dump_tsv(data, f, quoting=csv.QUOTE_ALL):
        data.to_csv(f, sep='\t', index=False, encoding='utf-8', quoting=quoting)

    handlers = dict(pkl=dump_pkl, json=dump_json, jsonl=dump_jsonl, xlsx=dump_xlsx, csv=dump_csv, tsv=dump_tsv)
    suffix = f.split('.')[-1]
    return handlers[suffix](data, f, **kwargs)

prompt_template = """You are an assistant skilled at evaluating the quality of creative text.
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to \
the user question displayed below. You'll need to assess the response on the following dimensions: \
Creativity, Richness, Visual Perception, Logical Coherence, Answer Accuracy and Image Relationship Understanding. \
We will provide you with a creative question and the AI model's response and a reference answer for your evaluation. \
As you begin your assessment, follow this process:
1. Evaluate the AI model's answers on different dimensions, pointing out its strengths or weaknesses \
in each dimension and assigning a score of 1 to 10 for each.
2. Finally, based on the assessments across dimensions, \
provide an overall score of 1 to 10 for the AI model's response.
3. Your scoring should be as stringent as possible and follow the scoring rules below:
In general, the higher the quality of the model's response and its strict adherence to user needs, \
the higher the score. Responses that do not meet user needs will receive lower scores.
Scoring rules:
Creativity:
Scores 1-2 when there is no innovation or uniqueness in the content.
Scores 3-4 when providing partially original content but with low creative quality.
Scores 5-6 when mostly creative but lacks significant novelty, with moderate quality.
Scores 7-8 when having novelty and high-quality content.
Scores 9-10 when highly novel and of exceptional quality compared to the reference answer.
Richness:
Scores 1-2 when lacking depth and breadth, with very limited information.
Scores 3-4 when limited in depth and breadth, with fewer explanations and examples, showing low diversity.
Scores 5-6 when limited in depth and breadth but provides basic necessary information.
Scores 7-8 when providing depth and useful additional information.
Scores 9-10 when providing exceptional depth, breadth, and high diversity compared to the reference answer.
Visual Perception:
Scores 1-2 when the description of the visual information in the image contains errors or \
is significantly inconsistent with the content of the image.
Scores 3-4 When the description of the visual information in the image reflects only a small amount \
of the image's information and contains some errors.
Scores 5-6 when the description of the visual information in the image includes the basic information \
of the image but contains minimal information.
Scores 7-8 when the description of the visual information in the image matches the image well and is rich in content, \
providing a substantial amount of information about the image.
Scores 9-10 when the description of the visual information in the image not only matches the image \
but also is more detailed and informative compared to the reference answer, providing more information about the image.
Logical Coherence:
Scores 1-2 when entirely incoherent, lacking any logic, and not matching the question or known information.
Scores 3-4 when somewhat coherent but with many logical errors or inconsistencies.
Scores 5-6 when mostly coherent, with few errors, but may struggle to maintain complete coherence in complex situations.
Scores 7-8 when excellent logical handling, very few errors.
Scores 9-10 when flawless logic, impeccable in handling complexity, \
and significantly higher logical coherence compared to the reference answer.
Answer Accuracy:
Scores 1-2 when the answer is significantly inconsistent with the question or contains obvious errors.
Scores 3-4 when the answer is partially correct but contains some errors or is incomplete.
Scores 5-6 when the answer is basically correct but lacks details or is not sufficiently detailed.
Scores 7-8 when the answer is accurate and detailed, fully corresponding to the question.
Scores 9-10 when the answer is not only accurate and detailed but also provides additional useful information, \
exceeding expectations.
Image Relationship Understanding:
Scores 1-2 when there are significant errors or confusion in distinguishing and describing different images, \
unable to correctly identify and relate the content of the images.
Scores 3-4 when the description of different images reflects only minimal distinguishing information, \
contains some errors and confusion, and fails to clearly differentiate and relate the images.
Scores 5-6 when the description of different images includes basic distinguishing information, \
is able to correctly identify and relate the images in a basic manner, \
but the information provided is minimal and lacks detail.
Scores 7-8 when the description of different images is accurate and detailed, \
clearly distinguishing and relating the images, \
with rich content that points out the main commonalities and differences between the images.
Scores 9-10 when the description of different images is not only accurate and detailed but also \
provides richer information and analysis, clearly distinguishing and relating the images, \
more comprehensively pointing out the commonalities and differences \
between the images compared to the reference answer.
Overall Score:
Scores 1-2 when irrelevant to the question, factually incorrect, or generates harmful content.
Scores 3-4 when no serious errors, mostly harmless, but of low quality and does not meet requirements.
Scores 5-6 when basically meeting requirements but performing poorly in some dimensions, with moderate quality.
Scores 7-8 when performing well in all dimensions.
Scores 9-10 when fully addressing user questions and all requirements, significantly surpassing the reference answer.
Please remember, you must evaluate and explain before scoring. After your explanation for each dimension, \
add the score for that dimension. Finally, at the end of your response, \
in the format of the dictionary (including brackets), return all your scoring results, \
ensuring your scores are integers:
{'Dimension One': Score, 'Dimension Two': Score, ..., 'Overall Score': Score}, \
for example: {'Creativity': 9, 'Richness': 6, ..., 'Overall Score': 7}.\n
[Question]
{question}

[The Start of Assistant's Answer]
{answer_a}
[The End of Assistant's Answer]

[The Start of Reference Answer]
{answer_b}
[The End of Reference Answer]

your judgement is:
""".strip()
def data_preprocess(input_path, input_file, save_dir='step1'):
    output_path = os.path.join(input_path, save_dir)
    os.makedirs(output_path, exist_ok=True)

    currrent_file_name = input_file.split('/')[-1]
    out_file = os.path.join(output_path, currrent_file_name)
    data = [json.loads(i) for i in open(os.path.join(input_path, input_file), encoding='utf8')]
    for idx, line_info in enumerate(data):
        # question = "\n".join(["{}: {}".format(x["role"], x["content"]) for x in line_info["messages"][:-1]]) + '\nassistant:'
        question = line_info["messages"][-2]["content"][0]["text"]
        output_benchmark = line_info["messages"][-1]["content"][0]["text"]
        predict_result = line_info["predict_result"]
        # id=line_info['id'] 
        id = line_info.get('id', None) 
        # print(line_info)
        instruction = prompt_template.replace("{question}", question).replace("{answer_a}", predict_result).replace("{answer_b}", output_benchmark)
        with open(out_file, 'a', encoding='utf8') as fw:
            json.dump({'idx': idx, 'id':id,'instruction': instruction, 'input': '', 'output': '', 'M': line_info["messages"], 'groundtruth': output_benchmark, 'model_predict': predict_result}, fw, ensure_ascii=False)
            fw.write('\n')

DIMS = [
    'Creativity', 'Richness', 'Visual Perception', 'Logical Coherence',
    'Answer Accuracy', 'Image Relationship Understanding', 'Overall Score'
]


class MetricCalculator:
    def __init__(self):
        self.DIMS = DIMS

    def calculat_metric(self, ans):
        all = defaultdict(lambda: 0)
        tot = defaultdict(lambda: 0)
        valid = defaultdict(lambda: 0)
        for k in ans: #每一个样本
            res = ans[k]['res']
            assert isinstance(res, pd.DataFrame)
            lt = len(res)  #lt=1
            for i in range(lt):
                line = res.iloc[i]
                for k in self.DIMS:
                    tot[k] += 1 #每个指标总预测数量
                    if k in line and line[k] is not None:
                        try:
                            score = int(line[k])
                            score = np.clip(score, 0, 10)
                            all[k] += score #每一列指标求和
                            valid[k] += 1 #每个指标有效数量
                        except Exception as e:
                            print(f'Failed to parse the score: {str(e)}')
        sp1 = {'set': 'all'}
        sp1.update({k: all[k] / tot[k] * 10 for k in self.DIMS}) #总平均分
        sp2 = {'set': 'valid'}
        sp2.update({k: all[k] / valid[k] * 10 for k in self.DIMS}) #剔除无效样本的平均

        return pd.DataFrame([sp1, sp2])
import ast

def extract_result(s,dim,strict_eval):
    """
    Extracts evaluation results from a given string.

    This function prioritizes extracting dictionary-like structures (e.g., {...}) 
    and supports case-insensitive matching. If no dictionary is found, it scans 
    the entire text in reverse to locate key-value pairs. Non-numeric characters 
    in values are automatically filtered, and invalid values (e.g., 'N/A', 'null') 
    are converted to None.

    Args:
        s (str): The input string containing evaluation results.
        dim (list): A list of dimension names to extract (e.g., ['Creativity', 'Richness']).
        strict_eval (bool): If True, enforces strict evaluation. If any dimension 
                            has a value of None, the entire sample is considered invalid.

    Returns:
        dict: A dictionary containing the extracted results for each dimension. 
              If strict_eval is True and any dimension is None, all dimensions 
              will be set to None.

    Example:
        Input:
            s1 = "the answer is: 1. Creativity: :7\n 27. Richness: 0. Visual Perception: 9. Logical Coherence: none, 'Answer Accuracy': 7. Image Relationship Understanding: yes. Overall Score: 7."
            s2 = "Final Scores:\n{'richness':    0, 'Visual Perception': 9, 'Logical Coherence': yy, 'Answer Accuracy': 7, 'Image Relationship Understanding': N/A, 'Overall Score': 7}"
        
        Output:
            For strict_eval = False:
                {'Creativity': None, 'Richness': 0, 'Visual Perception': 9, 
                 'Logical Coherence': None, 'Answer Accuracy': 7, 
                 'Image Relationship Understanding': None, 'Overall Score': 7}
                {'Creativity': None, 'Richness': 0, 'Visual Perception': 9, 'Logical Coherence': None, 'Answer Accuracy': 7, 'Image Relationship Understanding': None, 'Overall Score': 7}
            For strict_eval = True:
                {'Creativity': None, 'Richness': None, 'Visual Perception': None, 
                 'Logical Coherence': None, 'Answer Accuracy': None, 
                 'Image Relationship Understanding': None, 'Overall Score': None}
    """
    fields = dim
    
    result = {field: None for field in fields}
    
    # 尝试匹配字典结构（优先处理）
    dict_match = re.search(r"({.*})", s, re.DOTALL)
    if dict_match:
        dict_content = dict_match.group(1)
        # 处理字典内的键值对（含单引号或双引号）
        kv_pattern = re.compile(r"""['"]?([^'":]+)['"]?\s*:\s*([^,}]+)""")
        for match in kv_pattern.finditer(dict_content):
            key = match.group(1).strip()
            value = match.group(2).strip()
            key = key.title()
            if key in result:
                if value.isdigit():
                    value = int(value)
                    result[key] = value if 0 <= value <= 10 else None  
                else:
                    result[key] = None
    else:
        # 处理点分隔的键值对（如 "Key: Value. Key2: Value2."）
        for field in fields:
            
            escaped_field = re.escape(field)
            pattern = re.compile(
                # fr"{escaped_field}\s*:\s*([^\s.,']+)", 
                fr"(?:{escaped_field}|'{escaped_field}')\s*:\s*([^\s.,']+)",
                re.IGNORECASE
            )
            matches = pattern.findall(s)
            if matches:
                value_str = matches[-1].strip()  # 取最后一个匹配值
                if value_str.isdigit():
                    value = int(value_str)
                    result[field] = value if 0 <= value <= 10 else None  
                else:
                    result[field] = None

    
    if (all([result[x] is not None for x in DIMS])) or not strict_eval:
        return result
    else:
        return {field: None for field in fields}

def MMfin_acc(result_file,strict_eval):
    data = pd.read_json(result_file, lines=True)
    all_result_dict = []
    logs = []
    ans = {}
    for j, item in data.iterrows():
        predict_result = item['predict_result']
        # response = (item)  # 打分
        # print('output:', predict_result)
        # 使用正则表达式匹配目标字典的字符串形式

        ##vlmthkit method
        # match = re.search(r"({.*})", predict_result, re.DOTALL)
        # if match:
        #     target_dict_str = match.group(1)#"{'Creativity': 6, 'Richness': 7, 'Visual Perception': N/A, 'Logical Coherence': 8, 'Answer Accuracy': 6, 'Image Relationship Understanding': N/A, 'Overall Score': 7}"
        #     result_dict = eval(target_dict_str)
        #     print(result_dict)
        # else:
        #     result_dict = {d: None for d in DIMS}

        ##new method
        result_dict=extract_result(predict_result,DIMS,strict_eval)

        all_result_dict.append(result_dict)
        ans[j] = {'res': pd.DataFrame([result_dict])}
        if all([x in result_dict for x in DIMS]):
            logs.append('Succeed')
        else:
            logs.append(
                f'Following Dims are not in results of turn {j}: '
                f'{",".join([x for x in DIMS if x not in result_dict])}'
            )

    # print('all_result_dict:', all_result_dict)
    df = pd.DataFrame(all_result_dict)

    calculator = MetricCalculator()
    metric_df = calculator.calculat_metric(ans)

    return df, metric_df, logs




def evaluation(file_path,**kwargs):
    # res,  score = MMfin_acc(file_path)
    strict_eval = kwargs.get('strict_eval', True)
    df, metric_df, logs = MMfin_acc(file_path,strict_eval)
    res_pth = file_path.replace('.jsonl', '_score.csv')
    res_pth = file_path.replace('.jsonl', '_score.csv')
    metric_pth = file_path.replace('.jsonl', '_metric.csv')
    # logs_pth = file_path.replace('.jsonl', '_logs.txt')

    dump(df, res_pth)
    dump(metric_df, metric_pth)
    all_scores = metric_df[metric_df['set'] == 'all'].drop(columns=['set'])
    score = all_scores.to_dict(orient='records')[0] if not all_scores.empty else {}
    return score
    # return score


if __name__ == "__main__":
    pass