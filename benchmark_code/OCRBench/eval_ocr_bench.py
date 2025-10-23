import json
from tqdm import tqdm


def evaluation(input_path,**kwargs):
    """模型评估函数，用于评估模型在OCR任务中的表现。

    Args:
        input_path (str): 输入数据集的路径，文件应为JSON格式，每行包含模型的预测结果和标准答案。
        **kwargs: 可选参数，用于传递模型的子路径或其他配置参数。

    Returns:
        float: 返回模型的最终评估分数，归一化后的值。

    Examples:
        >>> score = evaluation("data/input.json")
        >>> print(score)
        0.85
    """
    OCRBench_score = {
            'Regular Text Recognition': 0,
            'Irregular Text Recognition': 0,
            'Artistic Text Recognition': 0,
            'Handwriting Recognition': 0,
            'Digit String Recognition': 0,
            'Non-Semantic Text Recognition': 0,
            'Scene Text-centric VQA': 0,
            'Doc-oriented VQA': 0,
            'Key Information Extraction': 0,
            'Handwritten Mathematical Expression Recognition': 0,
        }
    with open(input_path,"r") as f:
        lines = [json.loads(l) for l in f.readlines()]
    for i in tqdm(range(len(lines))):
        line = lines[i]
        predict = line['predict_result']

        answers = eval(line["choices"][0]["message"]['content'][0]['text'])
        line['eval_result'] = False
        # category = line['choices'][0]['message']['content'][1]['text']   #category
        category = line['category']
        if category == 'Handwritten Mathematical Expression Recognition':
            for j in range(len(answers)):
                answer = answers[j].strip().replace('\n', ' ').replace(' ', '')
                predict = predict.strip().replace('\n', ' ').replace(' ', '')
                if answer in predict:
                    OCRBench_score[category] += 1
                    line['eval_result'] = True
                    break
        else:
            for j in range(len(answers)):
                answer = answers[j].lower().strip().replace('\n', ' ')
                predict = predict.lower().strip().replace('\n', ' ')
                if answer in predict:
                    OCRBench_score[category] += 1
                    line['eval_result'] = True
                    break

    final_score_dict = {}
    final_score_dict['Text Recognition'] = \
        (OCRBench_score['Regular Text Recognition'] + OCRBench_score['Irregular Text Recognition']
        + OCRBench_score['Artistic Text Recognition'] + OCRBench_score['Handwriting Recognition']
        + OCRBench_score['Digit String Recognition'] + OCRBench_score['Non-Semantic Text Recognition'])
    final_score_dict['Scene Text-centric VQA'] = OCRBench_score['Scene Text-centric VQA']
    final_score_dict['Doc-oriented VQA'] = OCRBench_score['Doc-oriented VQA']
    final_score_dict['Key Information Extraction'] = OCRBench_score['Key Information Extraction']
    final_score_dict['Handwritten Mathematical Expression Recognition'] = \
        (OCRBench_score['Handwritten Mathematical Expression Recognition'])
    final_score_dict['Final Score'] = \
        (final_score_dict['Text Recognition'] + final_score_dict['Scene Text-centric VQA']
        + final_score_dict['Doc-oriented VQA'] + final_score_dict['Key Information Extraction']
        + final_score_dict['Handwritten Mathematical Expression Recognition'])
    final_score_dict['Final Score Norm'] = (float(final_score_dict['Final Score']) / 1000)
    with open(input_path,"w") as f:
        for l in lines:
            f.write(json.dumps(l,ensure_ascii=False)+"\n")
    # return final_score_dict['Final Score Norm']
    return {"acc":final_score_dict['Final Score Norm']}