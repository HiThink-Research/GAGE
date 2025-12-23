import json
import re

def read_jsonl(file):
    with open(file, 'r') as f:
        data = [json.loads(line) for line in f]
    return data

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)
