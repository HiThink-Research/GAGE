import json
import re
from utils import JsonPaser


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
                predict_data = re.sub('<think[\s\S]*think>', '', predict_result_str).strip()
                # print(predict_data)

                # j_paser = JsonPaser()
                
                # predict_data = j_paser.extract_json_from_text(predict_result_str)

                # if predict_data:
                #     predicted_answer = predict_data.get("answer", "")

            except json.JSONDecodeError:
                print("JSONDecodeError")

            # 提取正确答案与模型答案
            correct_answer = json.loads(correct_answer)
            correct_answer_dict = {item['description_id']: item['answer'] for item in correct_answer}
            #获取正确答案总数量
            total_answer = len(correct_answer)
            correct_count = 0

            false_desc_id = []

            # 检查模型答案中的每个项目
            predicted_answer_list = json.loads(predict_data)
            # print(predicted_answer_list)
            for predicted_item in predicted_answer_list:
                desc_id = predicted_item['description_id']
                predicted_answer = sorted([int(ans) for ans in predicted_item['answer']])

                
                # 检查description_id是否在标准答案中
                if desc_id in correct_answer_dict:
                    # 检查答案是否正确
                    if predicted_answer == sorted(correct_answer_dict[desc_id]):
                        correct_count += 1
                    else:
                        false_desc_id.append(desc_id)

                    # 如果答案不匹配，不增加计数（相当于错误）
                # 如果description_id不在标准答案中，不增加计数（相当于错误）

            score = correct_count / total_answer
            if score == 1:
                result = "True"
            elif 0 < float(score) < 1:
                result = "Partially Correct"
            else:
                result = "False"
            
            false_desc_ids = ','.join(str(x) for x in false_desc_id)
            d['eval_result'] = {
                "result": result,
                "false_desc_id": false_desc_ids
            }
            d['score'] = score
            corrects.append(score)
            total_scores.append(1)
            
        except Exception as e:
            d['eval_result'] = {"result": "False", "error": str(e)}
            d['score'] = 0
            total_scores.append(1)
            # print(d["messages"][0]["content"][0]["text"][378:478])
            # print(f"Error processing data: {e}")
        
        out.append(d)
    
    with open(input_path, 'w', encoding='utf-8') as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    
    total_correct = sum(corrects)
    total_possible = sum(total_scores)
    overall_score = total_correct / total_possible if total_possible > 0 else 0
    return {"acc": overall_score}

