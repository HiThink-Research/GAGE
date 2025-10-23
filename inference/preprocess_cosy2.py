import re
import json
from transformers import AutoProcessor, PreTrainedTokenizer


PROMPT = {
    'none': '{}',
    'Hithink': 'Human: {} \n\nAssistant:',
    "qwen":"<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
    'llama3':'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n',
    'llama2':'User:\n\n{}Assistant:\n\n',
    'chat_template': None,
    'deepseek':'<｜begin▁of▁sentence｜>User: {s_in}\nAssistant: '
}


MESSAGESPROMPT = {
    'none': '{system}{s_in}',
    'Hithink': '<<SYS>>\n{system}\n<</SYS>>\n\n{s_in}',
    'qwen': '<|im_start|>system\n{system}<|im_end|>\n\n{s_in}',
    'llama3':'<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system}<|eot_id|>{s_in}',
    'llama2':'{system}\n\n{s_in}',
    'deepseek':'<｜begin▁of▁sentence｜>{system}\n\n{s_in}',
}
HumanPROMPT = {
    'none': '{human_in}',
    'Hithink': 'Human:{human_in}',
    'qwen': '<|im_start|>user\n{human_in}\n<|im_end|>',
    'llama3':'<|start_header_id|>user<|end_header_id|>\n\n{human_in}<|eot_id|>',
    'llama2':'User:\n\n{human_in}',
    'deepseek':'User: {human_in}',
}
AssistantPROMPT = {
    'none': '{ai_in}',
    'Hithink': 'Assistant:{ai_in}',
    'qwen': '<|im_start|>assistant\n{ai_in}\n<|im_end|>',
    'llama3':'<|start_header_id|>assistant<|end_header_id|>\n\n{ai_in}<|eot_id|>',
    'llama2':'Assistant:\n\n{ai_in}',
    'deepseek':'Assistant: {ai_in}<｜end▁of▁sentence｜>',
}
ContinuePROMPT = {
    'none': '',
    'Hithink': 'Assistant:',
    'qwen': '<|im_start|>assistant\n',
    'llama3': '<|start_header_id|>assistant<|end_header_id|>\n\n',
    'llama2':'Assistant:\n\n',
    'deepseek':'Assistant: ',
}

def process_messages(tokenizer, sample, prompt_type, is_multi_modal=False) -> str:
    """
    将messages格式的数据拼接成完整的输入
    """
    messages = sample['messages']
    text_messages = list()
    for turn in messages:
        if isinstance(turn['content'], list):
            text = '\n'.join([x['text'] for x in turn['content'] if x['type'] == 'text'])
        elif isinstance(turn['content'], str):
            text = turn['content']
        else:
            raise ValueError(f'"content" must be either list or str: {turn["content"]}')
        text_messages.append(text)
    return ' '.join(text_messages)

   
def convert_sample_to_input_ids(s: dict, prompt_type: str, tokenizer: PreTrainedTokenizer):
    if 'token_ids' in s:
        return s['token_ids']

    if 'messages' in s:  # 多轮对话格式
        is_multi_modal, images, audios = False, [], []
        s_in = process_messages(tokenizer, s, prompt_type, is_multi_modal)
        input_ids = tokenizer.encode(s_in, allowed_special='all')
        # print("wnq debug:", input_ids)
        return input_ids
    else:
        raise ValueError("Only message format is supported!")


if __name__ == '__main__':
    pass