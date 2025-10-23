import re
from math import isclose
from collections import defaultdict
import json,os
import copy


        

def levenshtein_distance(s1, s2):
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


def anls_compute(groundtruth, prediction, threshold=0.5):
    dist = levenshtein_distance(groundtruth, prediction)
    length = max(len(groundtruth.upper()), len(prediction.upper()))
    value = 0.0 if length == 0 else float(dist) / float(length)
    anls = 1.0 - value
    if anls<=threshold:
        anls = 0.0
    return anls


def is_float_equal(reference, prediction, include_percentage: bool = False, is_close: float = False) -> bool:
    def get_precision(gt_ans: float) -> int:
        precision = 3
        if '.' in str(gt_ans):
            precision = len(str(gt_ans).split('.')[-1])
        return precision

    reference = float(str(reference).strip().rstrip("%").strip())
    try:
        prediction = float(str(prediction).strip().rstrip("%").strip())
    except:
        return False

    if include_percentage:
        gt_result = [reference / 100, reference, reference * 100]
    else:
        gt_result = [reference]
    for item in gt_result:
        try:
            if is_close:
                if isclose(item, prediction, rel_tol=0.01):
                    return True
            precision = max(min(get_precision(prediction), get_precision(item)), 2)
            if round(prediction, precision) == round(item, precision):
                return True
        except Exception:
            continue
    return False


def get_clean_string(s):
    s = str(s).lower().strip()
    if s.endswith("mile"):
        s.rstrip("mile").strip()
    if s.endswith("miles"):
        s.rstrip("miles").strip()
    if s.endswith("million"):
        s.rstrip("million").strip()
    # remove parenthesis
    s = re.sub(r'\s*\([^)]*\)', '', s).strip()
    # remove quotes
    s = re.sub(r"^['\"]|['\"]$", "", s).strip()
    s = s.strip().lstrip("$").strip()
    s = s.strip().rstrip("%").strip()
    return s


def is_exact_match(s):
    flag = False
    # Website
    if "https://" in s:
        flag = True
    # code file
    if s.endswith(".py") or s.endswith("ipynb"):
        flag = True
    if s.startswith("page"):
        flag = True
    # telephone number
    if re.fullmatch(r'\b\d+(-\d+|\s\d+)?\b', s):
        flag = True
    # time
    if "a.m." in s or "p.m." in s:
        flag = True
    # YYYY-MM-DD
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}[-\s]\d{2}\b', s):
        flag = True
    # YYYY-MM
    if re.fullmatch(r'\b\d{4}[-\s]\d{2}\b', s):
        flag = True
    # Email address
    if re.fullmatch(r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}', s):
        flag = True
    return flag


def isfloat(num):
    try:
        float(num)
        return True
    except ValueError:
        return False


def eval_score(gt, pred, answer_type):
    """
    每一条样本的评估函数，计算模型预测结果与标准答案之间的分数。
    :param gt:标签
    :param pred:裁判模型预测结果
    :param answer_type:答案类型
    :return:分数

    """
    if answer_type=="Int":
        try:
            gt, pred = int(gt), int(float(pred))
        except:
            pred = ""
        score = (gt==pred)
    elif answer_type=="Float":
        try:
            gt = float(get_clean_string(str(gt)))
            pred = float(get_clean_string(str(pred)))
        except:
            pred = ""
        score = is_float_equal(gt, pred, include_percentage=True, is_close=True)
    elif answer_type in ["Str", "None"]:
        gt = get_clean_string(gt)
        pred = get_clean_string(pred)
        if is_exact_match(gt):
            score = (gt==pred)
        else:
            score = anls_compute(gt, pred)
    else:
        if isinstance(gt, str) and gt.startswith("["):
            gt = eval(gt)
        if not isinstance(gt, list):
            gt = [gt]
        if isinstance(pred, str) and pred.startswith("["):
            pred = eval(pred)
        if not isinstance(pred, list):
            pred = [pred]
        print(len(gt), len(pred))
        if len(gt)!=len(pred):
            score = 0.0
        else:
            gt = sorted([get_clean_string(a) for a in gt])
            pred = sorted([get_clean_string(a) for a in pred])
            print(gt, pred)
            if isfloat(gt[0]) or is_exact_match(gt[0]):
                score = ("-".join(gt)=="-".join(pred))
            else:
                score = min([anls_compute(gt_v, pred_v) for gt_v, pred_v in zip(gt, pred)])

    return float(score)


def eval_acc_and_f1(samples):
    """
    计算模型预测结果的准确率和F1-score。
    :param samples:模型预测结果列表(包含打分)
    :return:准确率和F1-score
    """
    evaluated_samples = [sample for sample in samples if "eval_result" in sample]
    if not evaluated_samples:
        return 0.0, 0.0
    
    acc = sum([sample['eval_result'] for sample in evaluated_samples])/len(evaluated_samples)
    try:
        recall = sum([sample['eval_result'] for sample in evaluated_samples if sample["choices"][0]["message"]["content"][0]["text"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["choices"][0]["message"]["content"][0]["text"]!="Not answerable"])
        precision = sum([sample['eval_result'] for sample in evaluated_samples if sample["choices"][0]["message"]["content"][0]["text"]!="Not answerable"])/len([sample for sample in evaluated_samples if sample["predict_result"]!="Not answerable"])
        f1 = 2*recall*precision/(recall+precision) if (recall+precision)>0.0 else 0.0
    except:
        f1 = 0.0
    
    return acc, f1


def show_results(samples, show_path=None):
    """
    显示模型评估结果，包括准确率、F1-score等指标。
    :param samples:模型预测结果列表
    :param show_path:结果保存路径
    :return:None
    """
    for sample in samples:
        sample["data_tag"]["evidence_pages"] = eval(str(sample["data_tag"]["evidence_pages"]))
        sample["data_tag"]["evidence_sources"] = eval(str(sample["data_tag"]["evidence_sources"]))
    
    with open(show_path, 'w') as f:
        acc, f1 = eval_acc_and_f1(samples)
        f.write("Overall Acc: {} | Question Number: {}\n".format(acc, len(samples)))
        f.write("Overall F1-score: {} | Question Number: {}\n".format(f1, len(samples)))
        f.write("-----------------------\n")

        #####################
        acc_single_page, _ = eval_acc_and_f1([sample for sample in samples if len(sample["data_tag"]["evidence_pages"])==1])
        acc_multi_page, _ = eval_acc_and_f1([sample for sample in samples if len(sample["data_tag"]["evidence_pages"])!=1 and sample["choices"][0]["message"]["content"][0]["text"]!="Not answerable"])
        acc_neg, _ = eval_acc_and_f1([sample for sample in samples if sample["choices"][0]["message"]["content"][0]["text"]=="Not answerable"])

        f.write("Single-page | Accuracy: {} | Question Number: {}\n".format(
            acc_single_page, len([sample for sample in samples if len(sample["data_tag"]["evidence_pages"])==1])
        ))
        f.write("Cross-page | Accuracy: {} | Question Number: {}\n".format(
            acc_multi_page, len([sample for sample in samples if len(sample["data_tag"]["evidence_pages"])!=1 and sample["choices"][0]["message"]["content"][0]["text"]!="Not answerable"])
        ))
        f.write("Unanswerable | Accuracy: {} | Question Number: {}\n".format(
            acc_neg, len([sample for sample in samples if sample["choices"][0]["message"]["content"][0]["text"]=="Not answerable"])
        ))
        f.write("-----------------------\n")

        #####################
        source_sample_dict, document_type_dict = defaultdict(list), defaultdict(list)
        for sample in samples:
            for answer_source in sample["data_tag"]["evidence_sources"]:
                source_sample_dict[answer_source].append(sample)
            document_type_dict[sample["data_tag"]["doc_type"]].append(sample)
        for type, sub_samples in source_sample_dict.items():
            f.write(
                "Evidence Sources: {} | Accuracy: {} | Question Number: {}\n".format(type, eval_acc_and_f1(sub_samples)[0], len(sub_samples))
            )

        f.write("-----------------------\n")
        for type, sub_samples in document_type_dict.items():
            f.write(
                "Document Type: {} | Accuracy: {} | Question Number: {}\n".format(type, eval_acc_and_f1(sub_samples)[0], len(sub_samples))
            )





def build_prompt(query, prediction):
    """
    构建用于提取答案的提示模板。

    Args:
        query (str): 问题的字符串。
        prediction (str): 模型的预测答案。

    Returns:
        str: 格式化后的提示模板，用于指导模型匹配答案与选项。
    """
    prompt_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(prompt_dir, "prompt_for_answer_extraction.md")
    with open(
            prompt_path,
            'r', encoding='utf-8') as markdown:
        prompt = markdown.read()#提取答案的prompt

    tmpl = (
        prompt+'\n\nQuestion:{}\nAnalysis:{}\n'
    )
    return tmpl.format(query,prediction)
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
    """裁判模型的数据预处理函数，用于将原始数据转换为模型可用的格式，给裁判模型构建输入

    Args:
        input_path (str): 输入文件所在的目录路径。
        input_file (str): 输入文件的名称。
        save_dir (str): 输出文件的保存目录，默认为 'step1'。
        输出文件到input_path/save_dir/input_file
    Returns:
        None
    """
    output_path = os.path.join(input_path,save_dir)
    os.makedirs(output_path, exist_ok=True)
    currrent_file_name = input_file.split('/')[-1]
    out_file = os.path.join(output_path, currrent_file_name)
    data = [json.loads(i) for i in open(os.path.join(input_path, input_file), encoding='utf-8')]
    item_list=[]
    for id, line_info in enumerate(data):
        predict_result = line_info["predict_result"]
        query = line_info['messages'][0]['content'][-1]['text']
        judge_instruction = build_prompt(query,predict_result)
        answer = line_info["choices"][0]["message"]["content"][0]["text"]
        item = upload_pass_format(judge_instruction,answer)
        item['model_predict'] = predict_result
        item['data_tag']=line_info['data_tag']
        item['reference'] = line_info['reference']
        item_list.append(item)
    with open(out_file, 'w', encoding='utf-8') as fw:
        for item in item_list:
            fw.write(json.dumps(item, ensure_ascii=False)+'\n')

    

def evaluation(input_path,**kwargs):
    """模型评估函数，用于计算模型预测的各个指标。

    Args:
        input_path (str): 包含模型预测结果、裁判模型的预测结果和标准答案的文件路径。
        **kwargs: 可选参数，用于扩展功能。

    Returns:
        txt: 模型的各个指标到show_results函数进行展示,更改自己的show_path。
        eg.input_path=/mnt/data/users/fengjunyan/MMLongBench-Doc_evaluation/MMLongBench_Doc_final_v2.jsonl得到
        show_path=/mnt/data/users/fengjunyan/MMLongBench-Doc_evaluation/MMLongBench_Doc_final_v2_evaluation.txt
    """
    file_name = input_path.split('/')[-1].split('.')[0]
    show_path = os.path.join(os.path.dirname(input_path),file_name+"_evaluation.txt")
    with open(input_path,"r") as f:
        lines = [json.loads(l) for l in f.readlines()]
    correct_num = 0
    out=[]
    for line in lines:
        predict_result = line['predict_result']
        try:
            pred_ans = predict_result.split("Answer format:")[0].split("Extracted answer:")[1].strip()
            answer = line["choices"][0]["message"]["content"][0]["text"]
            answer_type=line["reference"][0]["message"]["content"][0]["type"]
            line['final_answer']=pred_ans
            line['eval_result']=eval_score(answer, pred_ans, answer_type)
        except:
            pred_ans = "Failed to extract"
            line['final_answer']=pred_ans
            line['eval_result']=0.0
        out.append(dict(line))
    with open(input_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o,ensure_ascii=False)+"\n")#打分后保存文件
    acc, f1 = eval_acc_and_f1(out)
    show_results(out,show_path=show_path)
    with open(show_path,'r',encoding='utf-8') as result:
        content=result.read()
        print(content)
    return {'acc': acc}


if __name__ == '__main__':
    # input_file=r"/mnt/data/users/fengjunyan/MMLongBench-Doc_evaluation/MMLongBench_Doc_final_v1.jsonl"
    # data_preprocess(input_path=r"/mnt/data/users/fengjunyan/MMLongBench-Doc_evaluation/",input_file="MMLongBench_Doc_final_v1.jsonl")
    evaluation("/mnt/data/users/fengjunyan/lonbench_evaluation/judge/output/MMLongBench_Doc.jsonl")