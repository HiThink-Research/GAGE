from typing import Dict, Any, List

import jieba
from rouge_chinese import Rouge

from typing import Dict, Any, List
import json

def evaluation(input_path,**kwargs):
    all_res_list =[]
     
    original_lines = []
    with open(input_path, "r", encoding="utf-8") as reader_pred:
        for line_pred in reader_pred:
            obj_pred = json.loads(line_pred)
            result = process_single_evaluation(obj_pred)
            # TODO

            if result:
                all_res_list.append(result)
            obj_pred["eval_result"] = {"result":str(compute_evaluation_metrics([result]))}
            original_lines.append(obj_pred)
        reader_pred.close()
    
    # TODO 
    with open(input_path, "w") as w_f:
        
        output = "\n".join([json.dumps(x, ensure_ascii=False) for x in original_lines])
        w_f.write(output)
        w_f.close()

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


def process_single_evaluation(data: Dict[str, Any], note=False, return_diff = False) -> Any:
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

    def get_row_flat_entity(obj):
        flat_obj = {}
        for group in obj:
            for key, output_dict in group.items():
                if key == "Row":
                    if output_dict["行索引"] and output_dict["行分类"]:
                        flat_obj["行分类"] = flat_obj.get("行分类", []) + [output_dict["行索引"] + "_" + output_dict["行分类"]]

                
        flat_obj["行分类"] = list(set(flat_obj["行分类"]))
        return flat_obj

    def filter_note_predict(output_obj, predict_obj):
        # 附注表评测时，只用标注的行索引评测
        output_idx = set()
        for group in output_obj:
            for key, output_dict in group.items():
                if key == "Row":
                    if output_dict["行索引"] and output_dict["行分类"]:
                        output_idx.add(output_dict["行索引"])

        flat_obj = {}
        for group in predict_obj:
            for key, output_dict in group.items():
                if key == "Row":
                    if output_dict["行索引"] and output_dict["行分类"] and output_dict["行索引"] in output_idx:
                        flat_obj["行分类"] = flat_obj.get("行分类", []) + [output_dict["行索引"] + "_" + output_dict["行分类"]]

                
        flat_obj["行分类"] = list(set(flat_obj["行分类"]))
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



        label_ner_dict = get_row_flat_entity(data['output'])
        if note:
            preds_ner_dict = filter_note_predict(data['output'], data['predict_result'])
        else:
            preds_ner_dict = get_row_flat_entity(data['predict_result'])

        
        print("label_ner_dict: ", label_ner_dict)
        print("preds_ner_dict: ", preds_ner_dict)

        print("--------\n")

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
                # for g in gold:
                #     if get_rouge(g, i, rouger,rouge_type="rouge-1")["f"] >= 0.7:
                #         is_match = True
                #         break
                #     if contain:
                #         if g in i or i in g:
                #             is_match = True
                #             break
                for g in gold:
                    if g == i:
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
        "output": """[{'Row': {'行索引': '5', '科目公布名称': 'Marketing', '行分类': 'selling and marketing expenses'}},
 {'Row': {'行索引': '6', '科目公布名称': 'Technology and development', '行分类': 'research and development expenses'}},
 {'Row': {'行索引': '8', '科目公布名称': 'General and administrative', '行分类': 'general and administrative expenses'}},
 {'Row': {'行索引': '10', '科目公布名称': 'Interest expense', '行分类': 'interest expense'}},
 {'Row': {'行索引': '11', '科目公布名称': 'Interest and other income', '行分类': 'interest income'}},
 {'Row': {'行索引': '13', '科目公布名称': 'Provision for income taxes', '行分类': 'income tax expense'}}]""",

        "predict_result": """[{'PosRow': {'行索引': '3', '科目公布名称': 'Revenues', '位置类科目': 'revenue'}},
 {'PosRow': {'行索引': '4', '科目公布名称': 'Cost of revenues', '位置类科目': 'cost of sales'}},
 {'PosRow': {'行索引': '8', '科目公布名称': 'Operating income', '位置类科目': 'operating income'}},
 {'PosRow': {'行索引': '12', '科目公布名称': 'Income before income taxes', '位置类科目': 'pretax income from continuing operations'}},
 {'PosRow': {'行索引': '14', '科目公布名称': 'Net income', '位置类科目': 'net income'}},
 {'Row': {'行索引': '5', '科目公布名称': 'Marketing', '行分类': 'selling and marketing expenses'}},
 {'Row': {'行索引': '6', '科目公布名称': 'Technology and development', '行分类': 'research and development expenses'}},
 {'Row': {'行索引': '7', '科目公布名称': 'General and administrative', '行分类': 'general and administrative expenses'}},
 {'Row': {'行索引': '10', '科目公布名称': 'Interest expense', '行分类': 'interest expense'}},
 {'Row': {'行索引': '11', '科目公布名称': 'Interest and other income', '行分类': 'interest income'}},
 {'Row': {'行索引': '13', '科目公布名称': 'Provision for income taxes', '行分类': 'income tax expense'}},
 {'Column': {'列索引': 'A', '列日期': '', '列分类': '公布科目列'}},
 {'Column': {'列索引': 'B', '列日期': 'Three Months Ended September 30,2022', '列分类': '本年日期列'}},
 {'Column': {'列索引': 'C', '列日期': 'Three Months Ended September 30,2021', '列分类': '上年日期列'}},
 {'Column': {'列索引': 'D', '列日期': 'Nine Months Ended September 30,2022', '列分类': '本年日期列'}},
 {'Column': {'列索引': 'E', '列日期': 'Nine Months Ended September 30,2021', '列分类': '上年日期列'}}]"""
    }
    # print(process_single_evaluation(data, return_diff = True))
    print(evaluation(input_path = "/mnt/data/kg/数据处理_新体系/hxdatasets/7b_predict/20241113/7b-qianwen-extraction-news2/美股新体系利润表_主表_test.json"))