from typing import Dict, Any, List

import jieba
from rouge_chinese import Rouge

import json
def evaluation(input_path,**kwargs):
    all_res_list =[]
     
    with open(input_path, "r", encoding="utf-8") as reader_pred:
        for line_pred in reader_pred:
            obj_pred = json.loads(line_pred)
            result = process_single_evaluation(obj_pred)
            if result:
                all_res_list.append(result)
    
    return compute_evaluation_metrics(all_res_list)


def compute_evaluation_metrics(results: List[Any]) -> Dict[str, Any]:
    """
    汇总计算评估指标

    Args:
        results (list): 处理后的结果列表，每个元素为`process_single_evaluation`方法返回的对象；注意处理单行返回对象为None的场景

    Returns:
        dict: 评估指标，key为指标名，值类型不确定

    {'关联关系中的亲属关系句子': (1, 0, 0), '人物亲属关系（某某和某某为夫妻关系、某某、某某夫妇）': (3, 4, 2), '关联关系中具有亲属关系的人名': (5, 3, 0)}
    """
    if not results:
        return {"f1_score": 0.0}
    res_dict = {}
    res_dict_f1 = {}

    for res_dict_tmp in results:
        if not res_dict_tmp:
            continue
        for res_key, res_num in res_dict_tmp.items():
            if res_dict.get(res_key):
                tp = res_dict[res_key][0] + res_num[0]
                fp = res_dict[res_key][1] + res_num[1]
                fn = res_dict[res_key][2] + res_num[2]
                count = res_dict[res_key][3]
                res_dict[res_key] = [tp, fp, fn, count if res_num[2] == 0 and res_num[1] == 0 and res_num[
                    0] == 0 else count + 1]
            else:

                res_dict[res_key] = [res_num[0], res_num[1], res_num[2],
                                     0 if res_num[2] == 0 and res_num[1] == 0 and res_num[0] == 0 else 1]

    F1_sum = 0
    count = 0
    sum_count = 0
    F1_sum_mic = 0
    recall = 0
    for key, value in res_dict.items():
        f = 0.0
        p = 0.0
        r = 0.0
        if (value[0] + value[1]) != 0.0:
            p = value[0] / (value[0] + value[1])
        if (value[0] + value[2]) != 0.0:
            r = value[0] / (value[0] + value[2])
        if (p + r) != 0.0:
            f = 2 * p * r / (p + r)
        F1_sum += f
        if value[3] != 0:
            count += 1
        F1_sum_mic += f * value[3]
        sum_count += value[3]
        recall += r
    
    # res_dict_f1.update({
    #     'F1-AVG':str(F1_sum / (count+0.001))[:4],
    #     'F1-AVG-mic':str(F1_sum_mic / (sum_count+0.001))[:4],
    #     'Recall':str(recall / (count+0.001))[:4]
    # })

    res_dict_f1 = {"f1_score": F1_sum_mic / (sum_count+0.001)}
    return res_dict_f1

def process_single_evaluation(data: Dict[str, Any]) -> Any:
    """
    处理单行评测集数据

    Args:
        data (dict): 单行评测集数据，通常包含三个字段：
            - instruction (str): 指令
            - output (str): 期望输出
            - predict_result (str): 模型预测输出

    Returns:
        object: 处理后的结果，类型不确定；若处理错误返回 None
    """

    def get_position_flat_entity(obj):
        flat_obj = {}
        for group in obj:
            for key, values in group.items():
                position_values = []
                for value in values:
                    position_values += [str(x) for x in value]
                if flat_obj.get(key):
                    old_result = flat_obj[key]
                    old_result.extend(position_values)
                    # 去重
                    flat_obj[key] = list(set(old_result))
                # 为空
                else:
                    flat_obj[key] = list(set(position_values))
        # print(flat_obj)
        return flat_obj

    def get_rouge(refer, pred, rouge,rouge_type="rouge-1"):
        """
        rouge_type: rouge-l rouge-2 rouge-l
        """
        pred = ' '.join(jieba.cut(pred))
        refer = ' '.join(jieba.cut(refer))
        scores = rouge.get_scores(pred, refer,avg=True)
        return scores[rouge_type]

    rouger = Rouge()

    if not "output" in data:
        data["output"] = str(data["choices"][0]["message"]["content"][0]["text"])
    
    try:
        if isinstance(data['predict_result'], str):
            data['predict_result'] = eval(data['predict_result'])

        if isinstance(data['output'], str):
            data['output'] = eval(data['output'])



        label_ner_dict = get_position_flat_entity(data['output'])
        preds_ner_dict = get_position_flat_entity(data['predict_result'])

        labels = list(label_ner_dict.keys())
        res_dict = {}
        F_avg = 0
        contain = True
        for label in labels:
            TP = 0
            FP = 0
            FN = 0
            # for pre, gold in zip(pre_lines, gold_lines):

            pre = preds_ner_dict.get(label, {})
            gold = label_ner_dict.get(label, {})

            if "-" in gold:
                gold.remove("-")
            if "-" in pre:
                pre.remove("-")
            for i in pre:
                is_match = False
                # 去掉标点符号
                for g in gold:
                    if get_rouge(g, i, rouger,rouge_type="rouge-1")["f"] >= 0.7:
                        is_match = True
                        break
                    if contain:
                        if g in i or i in g:
                            is_match = True
                            break
                if is_match:
                    TP += 1
                else:
                    FP += 1

            for i in gold:
                is_match = False

                for p in pre:
                    if get_rouge(i, p, rouger,rouge_type="rouge-1")["f"] >= 0.7:
                        is_match = True
                        break
                    if contain:
                        if g in i or i in g:
                            is_match = True
                            break
                if is_match == False:
                    FN += 1

            res_dict[label] = (TP, FP, FN)
       
        return res_dict
    except Exception as e:
        print(f"Error processing data: {e}")
        return None
    

if __name__ == "__main__":
    data = {
        "instruction": "",
        "output": "[{'利润表': [[6], [7]], '资产负债表': [[8]], '现金流量表': [[11]]}]",
        "predict_result": "[{'利润表': [[6], [3]], '资产负债表': [[8]], '现金流量表': [[11]]}]"
    }
    # print(process_single_evaluation(data))
    print(evaluation(input_path = "/mnt/data/kg/数据处理_新体系/hxdatasets/7b_predict/20241113/7b-qianwen-extraction-news2/表格类_美股10q_完整定位_test.json"))