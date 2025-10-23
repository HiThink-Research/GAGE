import json
import re
import re
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

# UIVision 原生逻辑
# def parse_coordinate(s):
#     # First try to parse JSON to find "bbox_2d"
#     try:
#         # Look for a JSON code block in case the string is wrapped in triple backticks
#         json_block = None
#         m = re.search(r"```json(.*?)```", s, re.DOTALL)
#         if m:
#             json_block = m.group(1).strip()
#         else:
#             # If no explicit JSON block is found, assume the entire string might be JSON
#             json_block = s.strip()

#         data = json.loads(json_block)
#         # If the data is a list, look for a dictionary with the "bbox_2d" key
#         if isinstance(data, list):
#             for item in data:
#                 if isinstance(item, dict) and "bbox_2d" in item:
#                     bbox = item["bbox_2d"]
#                     if isinstance(bbox, list) and len(bbox) == 4:
#                         return (bbox[0], bbox[1]), (bbox[2], bbox[3])
#         elif isinstance(data, dict) and "bbox_2d" in data:
#             bbox = data["bbox_2d"]
#             if isinstance(bbox, list) and len(bbox) == 4:
#                 return (bbox[0], bbox[1]), (bbox[2], bbox[3])
#     except Exception:
#         # If JSON parsing fails, we'll fall back to regex extraction
#         pass

    # Regex patterns to match bounding boxes in different formats
    patterns = [
        # Pattern 1: <|box_start|>[x, y, x, y]<|box_end|>
        r"<\|box_start\|\>\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]<\|box_end\|\>",
        # Pattern 2: <|box_start|>(x, y),(x, y)<|box_end|>
        r"<\|box_start\|\>\(\s*(\d+),\s*(\d+)\s*\),\(\s*(\d+),\s*(\d+)\s*\)<\|box_end\|\>",
        # Pattern 3: (x: number, y: number) format
        r"\(x:\s*(\d+),\s*y:\s*(\d+)\)",
        # Pattern 4: Top-left corner: (x, y) ... Bottom-right corner: (x, y)
        r"Top-left corner:\s*\(\s*(\d+),\s*(\d+)\s*\).*?Bottom-right corner:\s*\(\s*(\d+),\s*(\d+)\s*\)",
        # Pattern 5: General (x, y) pairs
        r"\(\s*(\d+),\s*(\d+)\s*\)",
        # Pattern 6: [x, y, x, y] format
        r"\[\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\s*\]",
        # Pattern 7: x: number, y: number format
        r"x:\s*(\d+),\s*y:\s*(\d+)"
    ]

    for pattern in patterns:
        matches = re.findall(pattern, s, re.DOTALL | re.IGNORECASE)
        if matches:
            if len(matches[0]) == 4:  # Four coordinates found
                coords = [int(x) for x in matches[0]]
                return (coords[0], coords[1]), (coords[2], coords[3])
            elif len(matches[0]) == 2:  # Two coordinates found
                if len(matches) >= 2:  # If we have two pairs, treat as top-left and bottom-right
                    x1, y1 = int(matches[0][0]), int(matches[0][1])
                    x2, y2 = int(matches[1][0]), int(matches[1][1])
                    return (x1, y1), (x2, y2)
                else:  # Single pair, return as is
                    return (int(matches[0][0]), int(matches[0][1]))

    # If nothing was found, return None
    return None


def is_in_bbox(point, bbox):
    """判断一个点是否在bbox内"""
    if len(bbox) != 4:
        print("错误：bbox 格式不正确", bbox)
        return False
    x, y = point
    x_min, y_min, x_max, y_max = bbox
    return x_min <= x <= x_max and y_min <= y <= y_max



# def eval_sample_positive_gt(sample, response):
#     bbox = sample["bbox"]
#     bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]  # x1, y1, x2, y2
#     if bbox[0] > bbox[2]:
#         bbox[0], bbox[2] = bbox[2], bbox[0]
#     if bbox[1] > bbox[3]:
#         bbox[1], bbox[3] = bbox[3], bbox[1]
#     img_size = sample["image_size"]
#     bbox = [bbox[0] / img_size[0], bbox[1] / img_size[1], bbox[2] / img_size[0], bbox[3] / img_size[1]]
    
#     try:
#         click_point = response["point"]  # may be none
#     except:
#         click_point = None
#     if click_point is None:
#         return "wrong_format"
#     # Check if the predicted point falls in the ground truth box
#     if (bbox[0] <= click_point[0] <= bbox[2]) and (bbox[1] <= click_point[1] <= bbox[3]):
#         return "correct"
#     else:
#         return "wrong"


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
    
    if predicted_coords is None:
        return None
    
    # Handle different return formats from parse_coordinate
    if isinstance(predicted_coords, tuple):
        if len(predicted_coords) == 2:
            # Check if it's two points (top-left, bottom-right) or a single point
            if isinstance(predicted_coords[0], tuple) and isinstance(predicted_coords[1], tuple):
                # Two points: calculate center
                x1, y1 = predicted_coords[0]
                x2, y2 = predicted_coords[1]
                return (x1 + x2) / 2, (y1 + y2) / 2
            else:
                # Single point
                return predicted_coords
    elif isinstance(predicted_coords, list) and len(predicted_coords) == 4:
        # Four coordinates: [x1, y1, x2, y2]
        x1, y1, x2, y2 = predicted_coords
        return (x1 + x2) / 2, (y1 + y2) / 2
    
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