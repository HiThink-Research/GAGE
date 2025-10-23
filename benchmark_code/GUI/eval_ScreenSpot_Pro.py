import json
import re
import os
def parse_coordinate(coord_str):
    """从字符串中提取出坐标"""
    # 匹配单点 (x, y) 或 矩形 (x1, y1, x2, y2)
    # pattern = r'\(\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+)\s*,\s*([\d.]+))?\s*\)'
    pattern = r'[\(\[]\s*([\d.]+)\s*,\s*([\d.]+)(?:\s*,\s*([\d.]+)\s*,\s*([\d.]+))?\s*[\)\]]'
    match = re.search(pattern, coord_str)
    
    if match:
        values = [float(v) for v in match.groups() if v is not None]
        if len(values) == 2:
            return tuple(values)  # 返回单点坐标 (x, y)
        elif len(values) == 4:
            return values  # 返回矩形 bbox [x1, y1, x2, y2]
    
    return None


def is_in_bbox(point, bbox):
    """判断一个点是否在bbox内"""
    if len(bbox) != 4:
        print("错误：bbox 格式不正确", bbox)
        return False
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max

def evaluate(data):
    """评估预测结果"""
    choices = data.get('choices', [])
    message = choices[0].get('message', [])
    predict_result = data.get('predict_result', '')
    
    bbox = json.loads(message['content'][0]["text"])
    
    if not bbox or len(bbox) != 4:
        print("错误：未能正确解析 bbox", bbox)
        return False

    print(f"评估：bbox={bbox}, predict_result={predict_result}")
    point = extract_answer(predict_result)
    if not point:
        print("错误：未能正确解析 predict_result", predict_result)
        point = (0, 0)  # 如果解析失败，默认返回 (0, 0)
        # return False
    print(point)


    # print(f"评估：bbox={bbox}, point={point}")
    return is_in_bbox(point, bbox)

def extract_answer(predict_result):
    """解析预测结果中的坐标"""
    predicted_coords = parse_coordinate(predict_result)
    if isinstance(predicted_coords, tuple):
        return predicted_coords
    elif isinstance(predicted_coords, list) and len(predicted_coords) == 4:
        x1, y1, x2, y2 = predicted_coords
        return (x1 + x2) / 2, (y1 + y2) / 2  # 返回中心点坐标
    return None

def evaluation(input_file_path):
    total_score = 0
    correct_count = 0
    total_count = 0

    base_path, ext = os.path.splitext(input_file_path)
    output_file_path = f"{base_path}_results{ext}"

    with open(input_file_path, 'r', encoding='utf-8') as input_file, \
         open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            data = json.loads(line.strip())
            eval_result = evaluate(data)

            total_count += 1
            if eval_result:
                correct_count += 1

            data['eval_result'] = eval_result
            output_file.write(json.dumps(data, ensure_ascii=False) + '\n')

    total_score = correct_count / total_count if total_count > 0 else 0
    return f"{total_score:.2f}"