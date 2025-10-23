import pandas as pd
import json
import os
import subprocess
import uuid
import re
import yaml

# 获取当前脚本所在目录
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(BASE_DIR, '../..'))
target_file = os.path.join(project_root, 'eval_result/swe_bench/swe_bench_prompt/predictions.json')

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def build_prompt(entry):
    prompt = (
        "Attention! Only give the answer, no any explanation needed. "
        "Given github repository issue fixing tasks, please return a patch for the issue above, "
        "note the deliverable must be output by diff format. "
        "Problem: " + entry.problem_statement.strip()
    )
    if entry.hints_text and entry.hints_text.strip():
        prompt += "\n hint: " + entry.hints_text.strip()
    # 限制输出长度
    prompt += (
        "\nPlease output only the patch in unified diff format, "
        "and make sure the patch does not exceed 100 lines. "
        "Do not output any explanation or extra text."
    )
    return prompt

def build_standard_llm_json(input_path: str, input_file=None, save_dir=None, model=None):
    if input_file is not None:
        input_path = os.path.join(input_path, input_file)
    if not input_path:
        raise ValueError("input_path must be a directory")
    if save_dir is None:
        save_dir = os.path.join(project_root, 'eval_result/swe_bench/swe_bench_prompt')

    df = pd.read_parquet(f'{input_path}')
    processed_swebench_json = []
    processed_paas_json = []
    for idx, row in enumerate(df.itertuples()):
        prompt = build_prompt(row)
        swe_bench_json_entry = {
            "instance_id": row.instance_id,
            "model_patch": "",
            "model_name_or_path": model or ""
        }
        inference_entry = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "text": prompt,
                            "type": "text"
                        }
                    ]
                }
            ],
            "choices": [
                {
                    "index": idx,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "text": "",
                                "type": "text"
                            }
                        ]
                    }
                }
            ],
            "swe_bench_json": swe_bench_json_entry
        }
        processed_swebench_json.append(swe_bench_json_entry)
        processed_paas_json.append(inference_entry)

    os.makedirs(save_dir, exist_ok=True)
    output_file_path_paas = os.path.join(save_dir, "swe_bench_dataset_paas.jsonl")
    with open(output_file_path_paas, 'w', encoding='utf-8') as fout:
        for entry in processed_paas_json:
            fout.write(json.dumps(entry, ensure_ascii=False) + '\n')

    output_file_path_swebench = os.path.join(save_dir, "predictions.jsonl")
    with open(output_file_path_swebench, 'w', encoding='utf-8') as fout_2:
        for entry in processed_swebench_json:
            fout_2.write(json.dumps(entry, ensure_ascii=False) + '\n')

def evaluation(pred_file: str, api_key: str = None, run_id: str = 'test_9999') -> dict:
    if not run_id:
        run_id = f'test_{uuid.uuid4()}'
    # 读取JSONL文件并转换为标准JSON格式
    predictions_data = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            prediction_entry = {
                "instance_id": data['swe_bench_json'][0]['instance_id'],
                "model_patch": str(data['predict_result']),
                "model_name_or_path": data['swe_bench_json'][0]['model_name_or_path']
            }
            predictions_data.append(prediction_entry)
    predictions_data = predictions_data[:500]
    # import pdb;pdb.set_trace()
    output_file = os.path.join(project_root, 'eval_result/swe_bench/predictions.json')
    os.makedirs(os.path.join(project_root, 'eval_result/swe_bench/'), exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(predictions_data, f, ensure_ascii=False, indent=2)
    # 调用sb-cli
    call_swebench_api(
        predictions_path=output_file,
        run_id=run_id,
        api_key=api_key
    )
    # 解析评测结果
    try:
        stat_file = os.path.join(project_root, 'eval_result/swe_bench/swe-bench_verified__test__test_9999.json')
        with open(stat_file, 'r', encoding='utf-8') as to_parse:
            data = json.load(to_parse)
            acc_results = parse_acc(data)
            logger.info("acc results is: ", acc_results)
            return {"acc": acc_results}
        logger.error("没有解析到有效的评估结果")
        return {"acc": None}
    except Exception as e:
        logger.error("解析评估结果时出错:", e)
        return {"acc": None}

def parse_acc(acc_dict):
    """
    从评测结果字典中提取准确率（百分比），并转为小数。
    兼容新格式：直接传入字典。
    """
    # 兼容两种格式：
    # 1. 直接有 'acc' 字段（百分比字符串）
    # 2. 有 resolved_instances/total_instances 字段
    if isinstance(acc_dict, dict):
        if 'acc' in acc_dict:
            # 兼容老格式
            try:
                return float(acc_dict['acc'].replace('%','')) / 100
            except Exception:
                pass
        if 'resolved_instances' in acc_dict and 'total_instances' in acc_dict:
            try:
                return acc_dict['resolved_instances'] / acc_dict['total_instances']
            except Exception:
                pass
    return None

def call_swebench_api(predictions_path, run_id, api_key=None, subset="swe-bench_verified", split="test"):
    cmd = [
        "sb-cli", "submit", subset, split,
        "--predictions_path", predictions_path,
        "--run_id", run_id
    ]
    env = os.environ.copy()
    if api_key:
        env['SWEBENCH_API_KEY'] = api_key
    else:
        env['SWEBENCH_API_KEY'] = 'swb_Xl5zvY88SOCnohzRrsTUqkFnjGbCgcKmAKKH-eVObJE_687716f3'  # 兜底老逻辑,不推荐使用
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        logger.error("Error:", result.stderr)
    
    print(result.stdout, end="")

# 示例 main
if __name__ == "__main__":
    # 假设外部引擎会提供 api_key 和 run_id
    # 例如：python eval_trigger_swebench.py <pred_file> <config_path>
    # 其中 <config_path> 包含 api_key 和 run_id
    # 这里为了方便，直接从命令行参数获取，实际应从 config_path 解析
    api_key = "swb_Xl5zvY88SOCnohzRrsTUqkFnjGbCgcKmAKKH-eVObJE_687716f3" # 兜底老逻辑,不推荐使用
    run_id = "test_run_id" # 示例 run_id
    evaluation('/mnt/HithinkOmni/user_workspace/zhangrongjunchen/damien/llm-eval/tests/eval_result/unit_test/SWE-Bench.jsonl', api_key, run_id)
    print(f"api_key={api_key}, run_id={run_id}")