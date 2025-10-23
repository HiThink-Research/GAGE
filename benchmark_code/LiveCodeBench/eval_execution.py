import re
import ast
import json
import os
# from eval_livecodebench_util import calculate_pass_rates
from tqdm import tqdm
from loguru import logger



def extract_expected_value(assertion_str: str) -> str:
    """
    解析python断言，提取==右侧的模型输出值

    参数：
        assertion_str: 包含assert的str
    
    return 大模型输出的test case str
    """
    try:
        parsed_ast = ast.parse(assertion_str, mode="exec")
    except SyntaxError:
        return "Syntax Error"

    if not parsed_ast.body or not isinstance(parsed_ast.body[0], ast.Assert):
        return "NOT an assert statement"

    comparison = parsed_ast.body[0].test

    if not (isinstance(comparison, ast.Compare) and isinstance(comparison.ops[0], ast.Eq)):
        return "NOT an equal comparison assertion"

    return ast.get_source_segment(assertion_str, comparison.comparators[0])



def verify_output(test_str: str, reference_output: str) -> tuple[bool, any, any]:
    """
    验证模型输出case是否与预期输出case相匹配

    参数：
        test_str: 模型输出str，可能包含assert
        reference_output: 预期真值测试用例的json

    return 匹配结果，模型输出测试值，期望真值 三元组ee
    """
    # 处理输入
    if "\n" in test_str:
        for line in test_str.splitlines():
            if not line.startswith("#") and "assert" in line:
                test_str = line.strip()
                break

    if "assert" in test_str:
        output_str = str(extract_expected_value(test_str))
    else:
        output_str = test_str.strip()

    try:
        test_value = eval(output_str)
    except Exception as e:
        return False, f"eval failed: {str(e)}", None

    # 将JSON参考真值转为python对象
    try:
        expected_value = json.loads(reference_output)
    except Exception as e:
        return False, None, f"JSON Parsing Failed: {str(e)}"

    return test_value == expected_value, test_value, expected_value


    

def preprocess_data(input_path: str, input_file=None, save_dir=None, task=None, **kwargs) -> list[dict]:
    """
    预处理execution数据集，与现有框架兼容
    """

    if input_file is not None:
        input_path = os.path.join(input_path, input_file)
    elif not input_path or os.path.isdir(input_path):
        raise ValueError("input_path must be a directory")

    preprocessed_data = []
    with open(input_path, 'r', encoding='utf-8') as f_in:
        data = [json.loads(l) for l in f_in]

    try:
        for d in data:
        # 构造针对这个case的prompt


            prompt = (
                f"ATTENTION! Only give the value of output, no any other explanation needed. Given the function name <{d['function_name']}>, and actually implementation code <{d['code']}>,"
                f"and input cases <{d['input']}>, now predict the output, remember the heads-up, NO EXPLAINATION! "
            )

            # 拿真值期望输出
            expected_output = d['output']

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
                        "messages": {
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
                "label": expected_output
                # "id": question_id,
            }

            # f.write(json.dump(inference_entry, ensure_ascii=False) + '\n') 
            preprocessed_data.append(inference_entry)
        print("preprocessed_data", preprocessed_data)

        os.makedirs(save_dir, exist_ok=True)
        output_file_path = os.path.join(save_dir, "livecodebench_execution_prompt.jsonl")
        with open(output_file_path, 'w', encoding='utf-8') as fout:
            for entry in preprocessed_data:
                # Correctly use json.dumps to get a string, then write it.
                line_to_write = json.dumps(entry, ensure_ascii=False)
                print("line_to_write", line_to_write)
                fout.write(line_to_write + '\n')
        print("Prompt saved to", output_file_path)
    except Exception as e:
        print("Exception occurred in preprocess_data:", e)
        import traceback
        traceback.print_exc()
        
    
    

def normalize_text(text):
    """
    文本归一化方法
    """
    text = re.sub(r'[^\w\s]', '', text)  
    text = re.sub(r'\s+', ' ', text)     
    return text.lower().strip()          

def verify_code(text1, text2):
    """比较文本相似度"""
    if text1 == text2:
        return 1.0
    
    # 归一化文本
    normalized_text1 = normalize_text(text1)
    normalized_text2 = normalize_text(text2)
    
    # 内容一致但格式不一致
    if normalized_text1 == normalized_text2:
        return 0.5
    
    # 内容不符
    return 0.0

def parse_model_output(raw_output: str):
    """
    解析模型输出
    """
    match = re.search(r'(\[.*?\]|\{.*?\}|-?\d+(?:\.\d+)?)', raw_output, re.DOTALL)
    if match:
        snippet = match.group(1)
        try:
            return eval(snippet, {"__builtins__": {}})
        except Exception:
            return snippet.strip()
    return raw_output.strip()

def evaluation(pred_file: str) -> dict:
    """
    评估execution结果和label的匹配率
    """
    total = 0
    correct = 0
    results = []
    with open(pred_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            data['eval_result'] = {"result": "False"}
            label = str(data.get("label", ""))
            pred = str(data.get("predict_result", ""))
            if "assistantfinal" in pred:
                pred = pred.split('assistantfinal')[1]
            if pred == label:
                data['eval_result']['result'] = "True"
                correct += 1
            total += 1
            results.append(data)
    score = correct / total if total > 0 else 0
    print(f"Score: {score:.2f}")

    result_dir = os.path.join(os.path.dirname(__file__), '../../eval_result/LiveCodeBench')
    os.makedirs(result_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pred_file))[0]
    print("base_name value:", base_name)
    stat_path = os.path.join(result_dir, f'{base_name}.eval_execution_results_stat.json')
    with open(stat_path, 'w', encoding='utf-8') as f:
        json.dump({"score": score}, f, ensure_ascii=False, indent=2)
    
    with open(pred_file, 'w', encoding='utf-8') as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    return {"acc": score}
