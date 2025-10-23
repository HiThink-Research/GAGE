import subprocess
import sys
import tempfile
import os
# import response
import inspect
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import shutil
import json
import re
import zlib
import base64
from loguru import logger
from typing import Optional, Tuple, Dict, Any


def extract_python_code(line: str) -> Optional[str]:
    """从一行数据中提取 ```python ``` 部分"""
    start_marker = "```python"
    end_marker = "```"
    
    # 找到起始和结束位置
    start_index = line.find(start_marker)
    end_index = line.find(end_marker, start_index + len(start_marker))
    
    # 如果找到起始和结束标记，提取内容
    if start_index != -1 and end_index != -1:
        return line[start_index + len(start_marker):end_index].strip()
    return None  # 未找到标记时返回 None

def extract_function_name(code: str) -> str:
    """从代码中提取函数名"""
    matches = re.findall(r"def (\w+)\(", code)
    if matches:
        return matches[-1] 
    return "function"

def parse_input_data(input_data: str) -> list:
    """将 input_data 分割为多个参数"""
    parts = input_data.strip().split("\n")
    parsed_args = []
    for part in parts:
        part = part.strip()
        if part.isdigit():
            parsed_args.append(int(part))  # 解析为整数
        elif part.replace(".", "", 1).isdigit():
            parsed_args.append(float(part))  # 解析为浮点数
        else:
            parsed_args.append(part)  # 保留为字符串
    return parsed_args

def run_code(code: str, input_data: str) -> Tuple[str, bool]:
    """安全执行代码并捕获输出"""
    extracted_code = extract_python_code(code)
    if extracted_code:
        code = extracted_code
    try:
        # 检查代码是否包含函数定义
        function_name = extract_function_name(code)
        has_function = function_name is not None
        has_executable_code = any(
            line.strip() and not line.strip().startswith("def ")
            for line in code.splitlines()
        )

        if has_function and not has_executable_code:
            # 解析 input_data 为多个参数
            args = parse_input_data(input_data)

            # 动态获取函数的参数数量
            local_vars = {}
            exec(code, {}, local_vars)
            func = local_vars.get(function_name)
            if not func:
                return ("Function not found", False)

            # 检查参数数量是否匹配
            sig = inspect.signature(func)
            if len(args) != len(sig.parameters):
                return (f"Argument count mismatch: expected {len(sig.parameters)}, got {len(args)}", False)

            # 生成调用代码
            call_code = f"""
{code}
print({function_name}({', '.join(map(repr, args))}))
"""
        else:
            # 直接执行代码
            call_code = code

        # 执行代码
        process = subprocess.run(
            [sys.executable, "-c", call_code],
            input=input_data.encode(),
            capture_output=True,
            timeout=5,
            check=False
        )
        output = process.stdout.decode().strip()
        error = process.stderr.decode().strip()
        return (output, True) if process.returncode == 0 else (error, False)
    except subprocess.TimeoutExpired:
        return ("Execution Timeout", False)
    except Exception as e:
        return (f"Runtime Error: {str(e)}", False)


def preprocess_data(input_path: str, input_file=None, save_dir=None, task=None, **kwargs):
    if input_file is not None:
        input_path = os.path.join(input_path, input_file)
    elif not input_path or os.path.isdir(input_path):
        raise ValueError("input_path must be a directory")

    preprocessed_data = []
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    # 自动识别message格式
    
        for idx, d in enumerate(data):
            # 构造prompt
            try:
                


                prompt = (
                    f"given" 
                    f" question_title <{d['question_title']}>,"
                    f" question_content <{d['question_content']}>," 
                    # f"starter code <{d['starter_code']}>,"
                    f" and example input-output cases <{d['public_test_cases']}>,"
                    f" python implementation code == ??, note only give the python implementation in required data structure, no any other texts needed."
                )


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
                        "label": d['public_test_cases'],
                        "test_case": d['public_test_cases']
                }
                # print("vanilla private_test_cases", d['private_test_cases'])
                private_test_cases = d['private_test_cases']          
                
            except Exception as e:
                logger.error(f"[Warning] Decode failed for entry {idx}: {e}")
                decoded_str = None

            # print("type of inference_entry:", type(inference_entry))
            preprocessed_data.append(inference_entry)


    os.makedirs(save_dir, exist_ok=True)
    output_file_path = os.path.join(save_dir, "livecodebench_generation_prompt.jsonl")
    with open(output_file_path, 'w', encoding='utf-8') as fout:
        for entry in preprocessed_data:
            # print("type of entry before dump:", type(entry))
            line_to_write = json.dumps(entry, ensure_ascii=False)
            logger.info("line_to_write", line_to_write)
            fout.write(line_to_write + '\n')
    logger.info("Prompt saved to", output_file_path)


def normalize_text(text):
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
    match = re.search(r'(\[.*?\]|\{.*?\}|-?\d+(?:\.\d+)?)', raw_output, re.DOTALL)
    if match:
        snippet = match.group(1)
        try:
            return eval(snippet, {"__builtins__": {}})
        except Exception:
            return snippet.strip()
    return raw_output.strip()

def evaluate_single_item(item: Dict[str, Any]) -> Dict[str, Any]:
    """评估单个数据项"""
    try:
        # 获取predict_result
        predict_result = item.get('predict_result', '')
        if not predict_result:
            return {
                'eval_result': 0.0,
                'eval_details': 'No predict_result found',
                'passed_cases': 0,
                'total_cases': 0
            }
        
        # 获取测试用例
        test_cases = item.get('test_case', [])
        if isinstance(test_cases, str):
            try:
                test_cases = json.loads(test_cases)
            except:
                test_cases = []
        
        if not test_cases:
            return {
                'eval_result': 0.0,
                'eval_details': 'No test cases found',
                'passed_cases': 0,
                'total_cases': 0
            }
        
        # 运行测试用例
        passed_cases = 0
        total_cases = len(test_cases)
        case_details = []
        
        for i, test_case in enumerate(test_cases):
            input_data = test_case.get('input', '')
            expected_output = test_case.get('output', '')
            
            # 运行代码
            model_output, run_success = run_code(predict_result, input_data)
            
            # 检查输出是否匹配
            is_correct = False
            if run_success:
                try:
                    # 尝试解析输出
                    if str(model_output).strip() == str(expected_output).strip():
                        is_correct = True
                        passed_cases += 1
                except:
                    pass
            
            case_details.append({
                'case_id': i,
                'input': input_data,
                'expected': expected_output,
                'actual': model_output,
                'run_success': run_success,
                'is_correct': is_correct
            })
        
        # 计算得分
        score = passed_cases / total_cases if total_cases > 0 else 0.0
        
        return {
            'eval_result': True if score >= 0.5 else False,
            'eval_details': f'Passed {passed_cases}/{total_cases} test cases',
            'passed_cases': passed_cases,
            'total_cases': total_cases,
            'case_details': case_details
        }
        
    except Exception as e:
        return {
            'eval_result': "False",
            'eval_details': f'Evaluation error: {str(e)}',
            'passed_cases': 0,
            'total_cases': 0
        }

def evaluation(input_path, task=None):
    """评估函数 - 读取prompt.jsonl文件进行评估"""
    
    logger.info(f"开始评估，输入文件: {input_path}")
    
    # 读取prompt.jsonl文件
    with open(input_path, 'r', encoding='utf-8') as f:
        prompt_data = [json.loads(line) for line in f]
    
    logger.info(f"读取到 {len(prompt_data)} 条prompt数据")
    
    # 对每个prompt进行推理，生成predict_result
    results = []
    total_passed = 0
    total_cases = 0
    
    for i, item in enumerate(prompt_data):
        logger.info(f"处理第 {i+1}/{len(prompt_data)} 条数据")
        
        # 已经有了predict_result，检查
        if 'predict_result' not in item:
            logger.info(f"警告：第 {i+1} 条数据缺少predict_result")
            continue
        
        # 评估predict_result
        eval_result = evaluate_single_item(item)
        item['eval_result'] = eval_result['eval_result']
        item['eval_details'] = eval_result['eval_details']
        item['passed_cases'] = eval_result['passed_cases']
        item['total_cases'] = eval_result['total_cases']
        
        item['eval_result'] = {'result': eval_result['eval_result']}

        if eval_result['eval_result']:
            item['eval_result']['result'] = "True"

        total_passed += eval_result['passed_cases']
        total_cases += eval_result['total_cases']
        results.append(item)
    
    # 计算总体得分
    overall_score = total_passed / total_cases if total_cases > 0 else 0.0
    logger.info(f"总体得分: {overall_score:.2f} ({total_passed}/{total_cases})")
    
    # 保存结果
    with open(input_path, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    # 保存统计结果
    stat_path = input_path.replace('.jsonl', '_eval_results_stat.json')
    stats = {
        'overall_score': overall_score,
        'total_passed': total_passed,
        'total_cases': total_cases,
        'total_samples': len(results)
    }
    with open(stat_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评估完成，结果保存到: {input_path}")
    logger.info(f"统计结果保存到: {stat_path}")

    with open(input_path, 'w', encoding='utf-8') as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    return {"acc": overall_score}
