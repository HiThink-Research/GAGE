import json
import os
import pandas as pd
import traceback
import requests
import yaml
from tqdm import tqdm
from io import BufferedReader
import re
from loguru import logger
import base64

def get_task_res_dict(config):
    tasks = list(config.get('tasks').keys())
    task_res_dict = {}
    for task in tasks:
        task_res_dict[config["tasks"][task]["data_path"]] = task
    return task_res_dict

def statistic(path,config):
    is_submit = config.get("is_submit", "1")
    submit_status = False if is_submit == "0" else True
    logs = os.listdir(path)
    results = {}
    submitters = []
    task_res_dict = get_task_res_dict(config)
    for l in logs:
        if ".log" not in l:
            continue
        with open(os.path.join(path,l),'r',encoding='utf-8') as f:
            result = json.load(f)
            task = l.split('.')[0]
            for task_res in task_res_dict:
                if l.split('.')[0] in task_res:
                    task = task_res_dict[task_res]
                    break
            results[task] = result[-1]['Average']
        
        # 结果统一为List传入提交函数
        if isinstance(results[task], dict):
            scores = results[task]
        elif isinstance(results[task], float):
            scores = {"score":results[task]}

    with open(os.path.join(path,"statistic.jsonl"),'w',encoding='utf-8') as f:
        f.write(json.dumps(results,ensure_ascii=False)+'\n')


if __name__ == "__main__":
    pass
