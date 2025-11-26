import importlib
import yaml

from argparse import Namespace
from typing import List
from transformers import AutoTokenizer, PreTrainedTokenizer

from gage_eval.assets.datasets.preprocessors.preprocess import convert_sample_to_input_ids as convert_sample_to_input_ids_


PROMPT = {
    'none': '{}',
    'Hithink0': 'Human: {} \n\nAssistant:',  # 原 "belle" prompt
    'Hithink': 'Human: {} \n\nAssistant:\n',
    'qwen': '<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n',
    'llama3': '<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n',
    'chat_template': None
}


# Adapted from: https://github.com/hsiehjackson/RULER/tree/dbc6a83c2f60d034dfaab8e7af42dde1a5d2e3dc
from utils.ruler.synthetic.constants import TASKS as TASKS_BASE


with open('testsets/ruler/synthetic/synthetic.yaml') as f:
    TASKS_CUSTOMIZED = yaml.safe_load(f)


task2samples = {}


class HFTokenizer:
    """
    Tokenizer from HF models
    """
    def __init__(self, tokenizer) -> None:
        self.tokenizer = tokenizer
    
    def text_to_tokens(self, text: str) -> List[str]:
        tokens = self.tokenizer.tokenize(text)
        return tokens

    def tokens_to_text(self, tokens: List[int]) -> str:
        text = self.tokenizer.convert_tokens_to_string(tokens)
        return text


def convert_sample_to_input_ids(s: dict, prompt_type: str, tokenizer: PreTrainedTokenizer):
    if 'task' not in s:
        return convert_sample_to_input_ids_(s, prompt_type, tokenizer)

    chunk_idx = 0
    chunk_amount = 1

    config = TASKS_CUSTOMIZED.get(s['task'])
    config.update(TASKS_BASE[config['task']])

    task_template = config['template']
    answer_prefix = config['answer_prefix'] if 'answer_prefix' in config else ''

    if PROMPT.get(prompt_type) is None:  # 使用 tokenizer 带的 chat_template
        messages = [{'role': 'user', 'content': task_template}]
        template = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:  # 使用 PROMPT 中定义的文本模板
        template = PROMPT[prompt_type].format(task_template)
    config['template'] = template + answer_prefix

    # Split task into multiple chunks 
    num_samples = s['num_samples']
    chunks = [(num_samples // chunk_amount) + (1 if i < num_samples % chunk_amount else 0) for i in range(chunk_amount)]
    num_samples = chunks[chunk_idx]
    pre_samples = sum(chunks[:chunk_idx])

    random_seed = 42 + chunk_idx

    args = Namespace(**config['args'])
    args.template = config['template']
    args.tokens_to_generate = config['tokens_to_generate']
    args.max_seq_length = s['max_seq_length']
    args.remove_newline_tab = False
    args.num_samples = num_samples
    args.random_seed = random_seed
    if config['task'] == 'qa':
        args.pre_samples = pre_samples

    key = '_'.join([config["task"], str(args)])
    samples = task2samples.get(key)
    if samples is None:
        module = importlib.import_module(f'utils.ruler.synthetic.{config["task"]}')
        generate_samples_func = getattr(module, 'generate_samples')
        samples = generate_samples_func(args, HFTokenizer(tokenizer))
        task2samples[key] = samples

    d = samples[s['sample_idx']]
    s.update(d)  # 将生成的input和output等内容加入原始的样本，用于保存结果和评估
    s['max_new_tokens'] = config['tokens_to_generate']

    input_ids = tokenizer.encode(d['input'])
    return input_ids


# 注意：移除了可执行入口，避免导入时访问硬路径或进入调试
