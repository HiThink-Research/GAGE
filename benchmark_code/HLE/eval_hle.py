import os
import json
from loguru import logger

def extract_label(data):
    # 优先从choices-message-content-text获取label
    if "choices" in data and isinstance(data["choices"], list):
        try:
            return str(data["choices"][0]["message"]["content"][0]["text"])
        except Exception:
            pass
    if "label" in data:
        return str(data["label"])
    if "answer" in data:
        return str(data["answer"])
    return ""

def extract_pred(data):
    if "predict_result" in data:
        return str(data["predict_result"])
    if "prediction" in data:
        return str(data["prediction"])
    if "choices" in data and isinstance(data["choices"], list):
        try:
            return str(data["choices"][0]["message"]["content"][0]["text"])
        except Exception:
            pass
    return ""

def generate_prompt(question: str) -> str:
    """
    生成HLE推理用prompt，要求只返回最终答案。
    """
    prefix = (
        "You are an expert in mathematics and science reasoning. "
        "Please read the following question carefully. "
        "If your answer involves equations or formulas, output them in LaTeX format. "
        "Return the final answer only, do not return any progress and/or explanation.\n\n"
    )
    return prefix + question

def evaluation(pred_file: str) -> dict:
    """
    评估HLE结果和label的匹配率，自动适配常见字段
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
            label = extract_label(data)
            pred = extract_pred(data)
            if pred == label:
                data['eval_result']['result'] = "True"
                correct += 1
            total += 1
            results.append(data)
    score = correct / total if total > 0 else 0
    logger.info(f"准确率: {score:.2f}")

    result_dir = os.path.join(os.path.dirname(__file__), '../../eval_result/HLE')
    os.makedirs(result_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pred_file))[0]
    stat_path = os.path.join(result_dir, f'{base_name}.eval_hle_results_stat.json')
    with open(stat_path, 'w', encoding='utf-8') as f:
        json.dump({"score": score}, f, ensure_ascii=False, indent=2)

    with open(pred_file, 'w', encoding='utf-8') as fout:
        for item in results:
            fout.write(json.dumps(item, ensure_ascii=False) + '\n')

    return {"acc": score}

def add_prompt_to_questions(input_file: str, output_file: str):
    """
    读取json或jsonl数据集，将每条数据的question字段加上prompt前缀，写入新文件。
    """
    results = []
    with open(input_file, 'r', encoding='utf-8') as f:
        first_char = f.read(1)
        f.seek(0)
        if first_char == '[':
            data = json.load(f)
            for item in data:
                try:
                    for c in item['messages'][0]['content']:
                        if c.get('type') == 'text':
                            c['text'] = generate_prompt(c['text'])
                except Exception as e:
                    logger.error(f"处理样本出错: {e}")
                results.append(item)
        else:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                try:
                    for c in item['messages'][0]['content']:
                        if c.get('type') == 'text':
                            c['text'] = generate_prompt(c['text'])
                except Exception as e:
                    logger.error(f"处理样本出错: {e}")
                results.append(item)

    with open(output_file, 'w', encoding='utf-8') as f:
        for item in results:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    logger.info(f"已处理{len(results)}条，结果写入{output_file}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) == 3:
        add_prompt_to_questions(sys.argv[1], sys.argv[2])
    elif len(sys.argv) == 2:
        evaluation(sys.argv[1])
    else:
        print("用法: python eval_hle.py <预测结果文件pred_file>")
        print("或:   python eval_hle.py <输入json/jsonl> <输出jsonl>") 

