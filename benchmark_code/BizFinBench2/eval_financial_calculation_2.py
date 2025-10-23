import json
import re
from utils import JsonPaser


def extract_numeric_value(text):
    """
    从文本中提取数字部分（包括小数和负数）
    返回提取到的数字，如果找不到数字则返回None
    """
    if text is None:
        return None
    
    # 匹配数字模式（包括小数、负数和科学计数法）
    pattern = r'-?\d+\.?\d*|\.\d+'
    matches = re.findall(pattern, str(text))
    
    if matches:
        try:
            # 返回第一个匹配的数字
            return float(matches[0])
        except ValueError:
            return None
    return None

def evaluation(input_path, **kwargs):
    corrects = []
    total_scores = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    
    out = []
    for d in data:
        try:
            # 提取正确答案
            choices = d.get("choices", [])
            if choices and len(choices) > 0:
                message = choices[0].get("message", {})
                content = message.get("content", [])
                if isinstance(content, list) and len(content) > 0:
                    correct_answer = content[0].get("text", "")
                else:
                    correct_answer = content if isinstance(content, str) else ""
            else:
                correct_answer = d.get("messages", [])[-1].get("content", "")
                if isinstance(correct_answer, list):
                    correct_answer = correct_answer[0].get("text", "") if correct_answer else ""

            predicted_answer = None
            try:
                predict_result_str = d.get("predict_result", "")

                j_paser = JsonPaser()
                
                predict_data = j_paser.extract_json_from_text(predict_result_str)

                if predict_data:
                    predicted_answer = predict_data.get("answer", "")
                    

                
            except json.JSONDecodeError:
                try:
                    json_pattern = r'"answer"\s*:\s*([^,}\s]+)'
                    match = re.search(json_pattern, predict_result_str)
                    if match:
                        predicted_answer_str = match.group(1)
                        if predicted_answer_str.startswith('"') and predicted_answer_str.endswith('"'):
                            predicted_answer = predicted_answer_str[1:-1]
                        else:
                            predicted_answer = predicted_answer_str
                except Exception as e:
                    print(f"Error extracting answer with regex: {e}")

            # 提取数字部分进行比较
            correct_numeric = extract_numeric_value(correct_answer)
            predicted_numeric = extract_numeric_value(predicted_answer)
            
            # 比较数字部分
            if correct_numeric is not None and predicted_numeric is not None:
                # 数值比较（允许极小误差）
                score = 1.0 if abs(predicted_numeric - correct_numeric) < 1e-4 else 0
            else:
                # 如果无法提取数字，回退到完整字符串比较
                score = 1.0 if str(predicted_answer).strip() == str(correct_answer).strip() else 0

            d['eval_result'] = {
                "result": "True" if score == 1 else "False",
                "correct_answer": correct_answer,
                "predicted_answer": predicted_answer,
                "correct_numeric": correct_numeric,
                "predicted_numeric": predicted_numeric
            }
            d['score'] = score
            corrects.append(score)
            total_scores.append(1)
            
        except Exception as e:
            d['eval_result'] = {"result": "False", "error": str(e)}
            d['score'] = 0
            total_scores.append(1)
            print(f"Error processing data: {e}")
        
        out.append(d)
    
    with open(input_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    
    total_correct = sum(corrects)
    total_possible = sum(total_scores)
    overall_score = total_correct / total_possible if total_possible > 0 else 0
    return {"acc": overall_score}

