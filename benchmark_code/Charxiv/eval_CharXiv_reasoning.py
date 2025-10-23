import json
from copy import deepcopy
import copy
import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))


from constants import REASONING_GRADING_PREFIX, REASONING_GRADING_INST
from get_stats import get_reasoning_scores,get_stats

def get_reasoning_result_gpt(response):
    try:
        if '```json' or '\\' in response:
            response=response.replace('```json', '').replace('```', '').replace('\\','\\\\').strip('\n')
            content = json.loads(response.strip('\n').strip())
        else:
            content = json.loads(response.strip('\n').strip())
        
        if 'extracted_answer' in response:
            ext, scr = content['extracted_answer'], content['score']
        else:
            ext, scr = content['extract_answer'], content['score']
    except Exception as e:
        print(f"Error: {e}")
        ext, scr = 'Failed to parse response', -1
    return ext, scr


def build_reasoning_grading_queries(reasoning_val, resp):
    """
    根据reasoning_val和response构建用于评分的查询。
    reasoning_val: reasoning_val
    resp: response 生成结果
    return: queries
    """
    queries=[]
    number=0
    for _, data in reasoning_val.items():
        figure_id = str(data['figure_id'])
        # question without instruction, response
        query, response = resp[number]['data_tag']['raw_question'], resp[number]['predict_result']
        # get query for answer type (inst_category), then
        # populate the query with the question, ground truth, and response
        grading_query = REASONING_GRADING_PREFIX + deepcopy(\
            REASONING_GRADING_INST[data['inst_category']])\
            .replace("<|question|>", query)\
            .replace("<|ground_truth|>", data['answer'])\
            .replace("<|response|>", response)
        query = {
            'figure_id': figure_id,
            'grading_query': grading_query,
        }
        queries.append(query)
        number=number+1
    return queries

def build_prompt(query):
    """
    构建用于提取答案的提示模板。

    Args:
        query (str): 问题的字符串。
        prediction (str): 模型的预测答案。

    Returns:
        str: 格式化后的提示模板，用于指导模型匹配答案与选项。
    """

    tmpl = (
        '{}'
    )
    return tmpl.format(query)
upload_pass_template={"messages":[{"role":"user","content":[{"text":"","type":"text"}]}],"choices":[{"index":0,"message":{"role":"assistant","content":[{"text":"","type":"text"}]}}],"model_prompt_tmpl":"","model_prompt_placeholder":[]}

def upload_pass_format(query:str,answer:str,model_prompt_tmpl=None,is_return_dict=True) -> str|dict:
    template=upload_pass_template.copy()
    template["messages"][0]["content"][0]["text"] = query
    if model_prompt_tmpl:
      template['model_prompt_tmpl'] = model_prompt_tmpl 
    template["choices"][0]["message"]["content"][0]["text"] = answer
    if is_return_dict:
        return copy.deepcopy(template)
    else:
        return json.dumps(template,ensure_ascii=False)


def data_preprocess(input_path, input_file, save_dir='step1'):
    """裁判模型的数据预处理函数，用于将原始数据转换为模型可用的格式。

    Args:
        input_path (str): 输入文件所在的目录路径。
        input_file (str): 输入文件的名称。
        save_dir (str): 输出文件的保存目录，默认为 'step1'。

    Returns:
        输出文件到judge/input目录下
    """
    eval_path = os.path.abspath(__file__)
    meta_dir = os.path.dirname(eval_path)
    reasoning_meta_path = os.path.join(meta_dir, "reasoning_val.json")
    with open(reasoning_meta_path, encoding='utf8') as f:
        reasoning_val = json.load(f)

    output_path = os.path.join(input_path,save_dir)
    os.makedirs(output_path, exist_ok=True)
    currrent_file_name = input_file.split('/')[-1]
    out_file = os.path.join(output_path, currrent_file_name)
    data = [json.loads(i) for i in open(os.path.join(input_path, input_file), encoding='utf8')]

    queries = build_reasoning_grading_queries(reasoning_val, data)
    for id, line_info in enumerate(data):
        response = line_info["predict_result"]
        query = queries[id]['grading_query']
        judge_instruction = build_prompt(query)
        answer = line_info["choices"][0]["message"]["content"][0]["text"]
        item = upload_pass_format(judge_instruction,answer)
        item['model_predict'] = response
        item['figure_id']= str(line_info['data_tag']['figure_id'])
        with open(out_file, 'a', encoding='utf8') as fw:
            json.dump(item, fw, ensure_ascii=False)
            fw.write('\n')


def evaluation(input_path,**kwargs):
    """模型评估函数，用于计算模型预测的准确率。

    Args:
        input_path (str): 包含模型预测结果、裁判模型的预测结果和标准答案的文件路径。(默认为judge/output目录下面的文件)
        **kwargs: 可选参数，用于扩展功能。

    Returns:
        float: 模型的准确率（正确预测的数量除以总样本数）。
        默认设置下评估在judge/output目录下，带有evaluation后缀，分数文件也在该目录下，带有score后缀。
    """
    file_name = input_path.split('/')[-1].split('.')[0]
    show_path = os.path.join(os.path.dirname(input_path),file_name+"_score.json")#打分文件
    show_path_1=os.path.join(os.path.dirname(input_path),file_name+"_evaluation.json")#评估文件
    eval_path = os.path.abspath(__file__)
    meta_dir = os.path.dirname(eval_path)
    descriptive_meta_path = os.path.join(meta_dir, "descriptive_val.json")
    with open(descriptive_meta_path, encoding='utf8') as f:
        reasoning_val=json.load(f)
    figure_id_list=[]
    for figure_id, _ in reasoning_val.items():
        figure_id_list.append(figure_id)

    with open(input_path,"r") as f:
        lines = [json.loads(l) for l in f.readlines()]

    queries={}
    for number,line in enumerate(lines):
        predict_result = line['predict_result']
        figure_id = str(line['figure_id'])
        ext, scr = get_reasoning_result_gpt( predict_result)
        if scr==-1:
            print("没有成功解析的样本序号：",number+1)
        query={
            figure_id:
            {   
            'figure_id': figure_id,
            'extracted_answer': ext,
            'score': scr
            }
        }
        queries.update(query)
    with open(show_path, "w") as f:
        json.dump(queries, f, indent=4)
    
    # 读取元数据信息
    eval_path = os.path.abspath(__file__)
    meta_dir = os.path.dirname(eval_path)
    image_meta_path = os.path.join(meta_dir, "image_metadata_val.json")
    descriptive_meta_path = os.path.join(meta_dir, "descriptive_val.json")
    reasoning_meta = os.path.join(meta_dir, "reasoning_val.json")

    image_meta = json.load(open(image_meta_path, 'r', encoding='utf-8'))
    descriptive_meta = json.load(open(descriptive_meta_path, 'r', encoding='utf-8'))
    reasoning_meta = json.load(open(reasoning_meta, 'r', encoding='utf-8'))
    reasoning_stats = get_reasoning_scores(queries, descriptive_meta, reasoning_meta, image_meta)
    reasoning_stats = get_stats(reasoning_stats)
    json.dump(reasoning_stats, open(show_path_1, "w"), indent=4)
    print("### Reasoning Stats ###")
    print(json.dumps(reasoning_stats, indent=4))
    return {'f1_score':reasoning_stats['Overall Score']}


