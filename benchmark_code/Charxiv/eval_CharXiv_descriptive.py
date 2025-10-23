import os
import json
import sys
import copy
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from constants import (
    DESCRIPTIVE_GRADING_PREFIX,
    DESCRIPTIVE_GRADING_QMAP,
    DESCRIPTIVE_GRADING_ICL,
)
from get_stats import get_descriptive_scores, get_stats
import inspect

# 不同的问题使用不同的生成prompt
def get_rubric(qid):
    instruction = None
    if qid in [1]:
        instruction = DESCRIPTIVE_GRADING_ICL["title"]
    if qid in [2, 3, 4, 5, 6, 7]:
        instruction = DESCRIPTIVE_GRADING_ICL["ocr"]
    if qid in [8, 9, 10, 12, 14, 15, 17, 19]:
        instruction = DESCRIPTIVE_GRADING_ICL["quant"]
    if qid in [11]:
        instruction = DESCRIPTIVE_GRADING_ICL["bool"]
    if qid in [13]:
        instruction = DESCRIPTIVE_GRADING_ICL["enum"]
    if qid in [16]:
        instruction = DESCRIPTIVE_GRADING_ICL["trend"]
    if qid in [18]:
        instruction = DESCRIPTIVE_GRADING_ICL["layout"]
    assert instruction is not None, f"Instruction for qid {qid} is not found."
    return instruction


# 验证gpt-4o打分输出是否有问题
def get_descriptive_result_gpt(response, length):
    try:
        if "```json" or "\\" in response:
            response = (
                response.replace("```json", "")
                .replace("```", "")
                .replace("\\", "\\\\")
                .strip("\n")
            )
            content = json.loads(response.strip("\n").strip())
        else:
            content = json.loads(response.strip("\n").strip())
        verify_grading_output(content, length)
    except Exception as e:
        print(f"Error: {e}")
        print(response)
        content = build_dummy_output(length)
    return content


def build_json_keys(length):
    keys = []
    # specify the keys for gpt-4o's json response
    for i in range(1, length + 1):
        keys.append(f"extract_answer_T{i}")
        keys.append(f"score_T{i}")
    return str(keys)


def populate_grading_inputs(batch):
    query = ""
    for i, (_, response, answer) in enumerate(batch):
        # index, response, answer
        curr_query = "T{}:\nResponse {}: {}\nGround Truth {}: {}\n\n".format(
            i + 1, i + 1, response, i + 1, answer
        )
        query += curr_query
    return query


def verify_grading_output(data, length_data):  # 检查打分后是否有字段缺失
    # check the integrity of keys and values
    for i in range(1, length_data + 1):
        assert (
            f"extract_answer_T{i}" in data
        ), f"extract_answer_T{i} is not found in {data}"
        assert f"score_T{i}" in data, f"score_T{i} is not found in {data}"
        assert data[f"score_T{i}"] in [0, 1], f"score_T{i} is not in [0, 1]"
    return True


def build_dummy_output(length_data):  # gpt-4o没有正确解析时的打分
    # if failed to parse the response, return dummy data
    data = {}
    for i in range(1, length_data + 1):
        data[f"extract_answer_T{i}"] = "Failed to parse response"
        data[f"score_T{i}"] = -1
    return data


def preprocess_descriptive_grading_queries(input, resp, num_templates=19
):  # 根据qid将所有问题分组为20组
    # group the responses based on the template id instead of figure id
    groups = {i: [] for i in range(1, num_templates + 1)}
    for _, data in input.items():
        figure_id = data["figure_id"]
        qids = data["qids"]
        for i, qid in enumerate(qids):
            # figure_id with question index
            resp_key = f"{figure_id}_{i}"
            for obj in resp:
                if resp_key == obj["id"]:
                    response = obj["predict_result"]
            answer = data["answers"][i]
            groups[qid].append((resp_key, response, answer))
    return groups


def build_descriptive_grading_queries(groups, nq_per_query=5):  # 生成描述类问题的prompt
    queries = []
    for qid, data in groups.items():
        # batched evaluation based on number of questions per query (nq_per_query)
        for i in range(0, len(data), nq_per_query):
            # batch: list of tuples (resp_key, response, answer)
            batch = data[i : i + nq_per_query]
            # question based on the template id
            question = DESCRIPTIVE_GRADING_QMAP[qid]
            # build the json keys for GPT-4o's response
            json_keys = build_json_keys(len(batch))
            # populate batch size, question, and json keys spec
            prefix = (
                DESCRIPTIVE_GRADING_PREFIX.replace("<|NUM_TRIPLETS|>", str(len(batch)))
                .replace("<|OVERARCHING_QUESTION|>", question)
                .replace("<|JSON_KEYS|>", json_keys)
            )
            # add in-context grading example based on the template id
            rubric_icl = get_rubric(qid)
            # prompt + example + model responses
            grading_query = prefix + rubric_icl + populate_grading_inputs(batch)
            curr_query = {
                "resp_keys": [d[0] for d in batch],
                "grading_query": grading_query,
                "responses": [d[1] for d in batch],
                "answers":  [d[2] for d in batch],
            }
            queries.append(curr_query)
    return queries


def postprocess_descriptive_grading_queries(queries):
    scores = {}
    for query in queries:
        # query contains resp_keys, grading_query, extract_answer and score
        resp_keys = query["resp_keys"]
        for i, resp_key in enumerate(resp_keys):
            # extract the answer and score for each response key
            extracted_answer = query[f"extract_answer_T{i+1}"]
            score = query[f"score_T{i+1}"]
            # store the extracted answer and score
            scores[resp_key] = {
                "resp_id": resp_key,
                "extracted_answer": extracted_answer,
                "score": score,
            }
    return scores


def build_prompt(query):
    """
    构建用于提取答案的提示模板。

    Args:
        query (str): 问题的字符串。
        prediction (str): 模型的预测答案。

    Returns:
        str: 格式化后的提示模板，用于指导模型匹配答案与选项。
    """

    tmpl = "{}"
    return tmpl.format(query)


upload_pass_template = {
    "messages": [{"role": "user", "content": [{"text": "", "type": "text"}]}],
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": [{"text": "", "type": "text"}]},
        }
    ],
    "model_prompt_tmpl": "",
    "model_prompt_placeholder": [],
}


def upload_pass_format(
    query: str, answer: str, model_prompt_tmpl=None, is_return_dict=True
) -> str | dict:
    template = upload_pass_template.copy()
    template["messages"][0]["content"][0]["text"] = query
    if model_prompt_tmpl:
        template["model_prompt_tmpl"] = model_prompt_tmpl
    template["choices"][0]["message"]["content"][0]["text"] = answer
    if is_return_dict:
        return copy.deepcopy(template)
    else:
        return json.dumps(template, ensure_ascii=False)



def get_current_module_path():
    # 获取当前的栈帧
    frame = inspect.currentframe()
    try:
        # 获取调用者的信息
        caller_frame = frame.f_back
        # 获取调用者所在模块的信息
        module = inspect.getmodule(caller_frame)
        # 返回模块的文件路径
        return module.__file__ if module else None
    finally:
        # 避免内存泄漏，删除对栈帧的引用
        del frame
        del caller_frame

def data_preprocess(input_path, input_file, save_dir="step1"):
    """裁判模型的数据预处理函数，用于将原始数据转换为模型可用的格式，给裁判模型构建输入

    Args:
        input_path (str): 输入文件所在的目录路径。
        input_file (str): 输入文件的名称。
        save_dir (str): 输出文件的保存目录，默认为 'step1'。

    Returns:
        None
    """
    # 推理结果文件
    data = [json.loads(i) for i in open(os.path.join(input_path, input_file), encoding="utf-8")]
    # 输出文件，同名，不同目录
    output_path = os.path.join(input_path, save_dir)
    os.makedirs(output_path, exist_ok=True)
    currrent_file_name = input_file.split("/")[-1]
    output_file = os.path.join(output_path, currrent_file_name)
    # 按照问题，将数据分组，总共19个描述性问题
    num_templates = 19
    groups = {i: [] for i in range(1, num_templates + 1)}
    for item in data:
        qid = item["data_tag"]["qid"]
        resp_key = item["id"]
        response = item["predict_result"]
        answer = item['choices'][0]['message']['content'][0]['text']
        groups[qid].append((resp_key, response, answer))
    # 5个一组构造打分prompt
    judge_queries = build_descriptive_grading_queries(groups)
    judge_queries_path = os.path.join(output_path, "judge_queries.json")
    json.dump(judge_queries, open(judge_queries_path, "w", encoding="utf-8"), indent=4)
    item_list = []
    for _, line_info in enumerate(judge_queries):
        judge_instruction = build_prompt(line_info["grading_query"])
        resp_keys_list = line_info["resp_keys"]
        answers = line_info["answers"]
        responses = line_info["responses"]
        item = upload_pass_format(judge_instruction, str(answers))
        item["resp_keys_list"] = resp_keys_list
        item["model_predict"] = str(responses)
        item_list.append(item)

    with open(output_file, "w", encoding="utf-8") as fw:
        for item in item_list:
            fw.write(json.dumps(item, ensure_ascii=False) + "\n")


def evaluation(input_path, **kwargs):
    """模型评估函数，用于计算模型预测的各个指标。

    Args:
        input_path (str): 包含模型预测结果、裁判模型的预测结果和标准答案的文件路径,judge/input。
        **kwargs: 可选参数，用于扩展功能。

    Returns:
        txt: 模型的各个指标到show_results函数进行展示,评估文件就在show_path_1下面。
        eg.默认设置下评估在judge/output目录下，带有evaluation后缀，分数文件也在该目录下，带有score后缀。
    """
    file_name = input_path.split("/")[-1].split(".")[0]
    show_path = os.path.join(os.path.dirname(input_path), file_name + "_score.json")  # 打分文件
    evaluation_path = os.path.join(os.path.dirname(input_path), file_name + "_evaluation.json")  # 评估文件

    with open(input_path, "r") as f:
        lines = [json.loads(l) for l in f.readlines()]

    combined_queries = []
    for item in lines:
        resp_keys = item["resp_keys_list"]
        predict_result = get_descriptive_result_gpt(item["predict_result"], len(resp_keys))
        combined_queries.append({"resp_keys": resp_keys, **predict_result})
    queries = postprocess_descriptive_grading_queries(combined_queries)
    with open(show_path, "w", encoding="utf-8") as f:
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
    descriptive_stats = get_descriptive_scores(queries, descriptive_meta, reasoning_meta, image_meta)
    descriptive_stats = get_stats(descriptive_stats)
    json.dump(descriptive_stats, open(evaluation_path, "w", encoding="utf-8"), indent=4)
    print("### descriptive Stats ###")
    print(json.dumps(descriptive_stats, indent=4))
    return {"f1_score": descriptive_stats["Overall Score"]}


if __name__ == "__main__":
    evaluation(
        r"CharXiv_descriptive.jsonl"
    )
