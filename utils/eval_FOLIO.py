import json
import re

def choice(text):
    predict_result = []
    if re.search(r'\\boxed\{.*?\}', text):
        predict_result = re.findall(r'\\boxed\{.*?\}', text)
    elif re.search(r'\*\*Answer:\*\*',text):
        print(1)
        predict_result.append(text.split('**Answer:**')[-1].strip().strip('.'))
    elif re.search(r'\*\*Answer:\s*\w+.*\*\*',text):
        print(2)
        predict_result = re.findall(r'\*\*Answer:\s*(\w+).*\*\*', text)
    elif re.search(r'\*\*Final judgment:\*\*',text):
        print(5)
        predict_result.append(text.split('**Final judgment:**')[-1].strip().strip('.'))
    elif re.search(r'\*\*Judgment:\*\*',text):
        print(6)
        predict_result.append(text.split('**Judgment:**')[-1].strip().strip('.'))
    elif re.search(r'\*\*Final Judgment:\*\*',text):
        print(7)
        predict_result.append(text.split('**Final Judgment:**')[-1].strip().strip('.'))
    elif re.search(r'\*\*Final Answer:\*\*',text):
        print(3)
        predict_result.append(text.split('**Final Answer:**')[-1].strip())
    elif re.search(r'\*\*False|True|Uncertain\*\*',text):
        print(4)
        predict_result = re.findall(r'\*\*(False|True|Uncertain)\*\*', text)
    return predict_result


def evaluation(file_path):
    """

    参数:
    file_path (str): 包含预测结果的文件路径

    返回:
    float: 准确率
    """
    correct_count = 0.
    total_count = 0.
    nofind=[]
    with open(file_path, 'r', encoding='utf-8') as f:
        for number,line in enumerate(f):
            data = json.loads(line)
            output = data['choices'][0]['message']['content'][0]['text']
            predict_result = choice(data['predict_result'].strip())
            if predict_result ==[]:
                nofind.append(number)
                result=''
            else:
                text=predict_result[-1]
                # print(text)
                if "text" in text:
                    result = re.findall(r'\\text{(.*?)\}',text)[-1]
                elif "boxed" in text:
                    result = re.findall(r'\{(.*?)\}',text)[-1]
                else:
                    result = text
                if result not in ['False','True','Uncertain']:
                    print(result)
            if output == result:
                correct_count += 1
            total_count += 1

    accuracy = correct_count / total_count if total_count > 0 else 0
    print('未成功匹配的字符串:',nofind)
    print(f"FOLIO数据集的准确率: {accuracy:.2%}")
    print(accuracy)
    return {"acc":accuracy}
    # return accuracy,nofind

if __name__ == "__main__":
    test_file_path = "/mnt/data/users/fengjunyan/FOLIO/test_result5/FOLIO.jsonl"  # 文件路径
    accuracy = evaluation(test_file_path)
