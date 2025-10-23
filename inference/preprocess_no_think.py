import json
from transformers import AutoTokenizer, PreTrainedTokenizer


PROMPT = {
    'chat_template': None,
}


def process_messages(
    tokenizer,
    sample,
    prompt_type,
    is_multi_modal=False,
    remove_last_assistant=True,
    add_generation_prompt=True,
):
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
        text_messages.append({'role': turn['role'], 'content': text})
        if t := turn.get('tool_calls'):
            text_messages[-1]['tool_calls'] = t

    if text_messages[-1]['role'] == 'assistant':
        # 若最后是答案则去掉
        last_turn_text = text_messages[-1]
        last_turn = messages[-1]
        text_messages = text_messages[:-1]
        messages = messages[:-1]

    # 替换最后一轮（PaaS平台格式）
    if 'model_prompt_tmpl' in sample and sample['model_prompt_tmpl'] and text_messages[-1]['role'] == 'user':
        model_prompt_tmpl = sample['model_prompt_tmpl']
        model_prompt_placeholder = sample['model_prompt_placeholder']
        for placeholder_item in model_prompt_placeholder:
            key = placeholder_item['key']
            value = placeholder_item['value']
            model_prompt_tmpl = model_prompt_tmpl.replace('{{' + '{}'.format(key) + '}}', value)
        text_messages = text_messages[:-1] + [{'role': 'user', 'content': model_prompt_tmpl}]
        # TODO: messages的最后一轮是否需要替换？messages在输入多模态数据时会用到，暂不替换

    if not remove_last_assistant:
        text_messages.append(last_turn_text)
        messages.append(last_turn)

    tools = sample.get('tools')
    if isinstance(tools, str) and tools:
        tools = json.loads(tools)
    final_prompt = tokenizer.apply_chat_template(
        messages if is_multi_modal else text_messages,
        tools=tools or None,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )

    return final_prompt


def check_for_multi_modal(messages):
    """
    判断在messages中是否有multi-modal数据
    如果有的话进行处理并返回 True, List[Image path], List[Audio path]
    """
    # 处理所有 'user' 消息，去除其中的jpg和url 
    user_message = [msg for msg in messages['messages'] if msg['role'] == 'user']

    if not user_message:
        return False, None, None

    # 检查是否有 type 不为 'text' 的内容
    images = []
    audios = []
    for msg in user_message:
        # 检查 content 是否存在且为列表，否则的话，这是一条纯文本格式的消息
        if 'content' not in msg or not isinstance(msg['content'], list):
            continue
        for item in msg['content']:
            if item.get('type') == 'image_url':
                images.append(item['image_url']['url'])
            if item.get('type') == 'audio_url':
                audios.append(item['audio_url']['url'])
    is_multi_modal = bool(images or audios)
    return is_multi_modal, images, audios


def convert_sample_to_input_ids(
    s: dict,
    prompt_type: str,
    tokenizer: PreTrainedTokenizer,
    remove_last_assistant: bool = True,
    add_generation_prompt: bool = True,
):
    if 'token_ids' in s:
        return s['token_ids']

    if 'messages' in s:  # 多轮对话格式
        is_multi_modal, images, audios = check_for_multi_modal(s)
        s_in = process_messages(
            tokenizer, s, prompt_type, is_multi_modal, remove_last_assistant, add_generation_prompt
        )
        if is_multi_modal:
            inputs = {
                "prompt": s_in,
                "multi_modal_data": {}
            }
            if images:
                inputs["multi_modal_data"]["image"] = images
            if audios:
                inputs["multi_modal_data"]["audio"] = audios
            return inputs

    elif 'target' in s:  # 输入格式：input/source + target
        s_in = s['source'] if 'source' in s else s['input']
    else:  # 输入格式：instruction + input + output
        s_in = s.get('instruction', '')
        if not s_in:
            s_in = s.get('system', '')
        if s.get('input'):
            if s_in:
                s_in += '\n'
            s_in += s['input']

    if prompt_type in PROMPT.keys():
        kwargs = {}
        if "messages" in s or "model_prompt_tmpl" in s:
            # messages 格式
            prompt = s_in
        else:
            if not isinstance(s_in, str):
                s_in = str(s_in)
            s_in = [{'role': 'user', 'content': s_in}]
            prompt = tokenizer.apply_chat_template(
                s_in, tokenize=False, add_generation_prompt=add_generation_prompt, enable_thinking=False)
            kwargs['add_special_tokens'] = False

        #处理当输入是多模态Processor的时候tokenizer异常
        if "Processor" in type(tokenizer).__name__:
            tokenizer = tokenizer.tokenizer
        input_ids = tokenizer.encode(prompt, **kwargs)
    else: 
        raise ValueError("prompt type must be in: {}".format(PROMPT.keys()))

    return input_ids


if __name__ == '__main__':
    tokenizer = AutoTokenizer.from_pretrained('/mnt/data/llm/models/chat/Qwen3-32B')
    # s = {"instruction": "你是谁？"}
    s = {'messages': [{'role': 'user', 'content': '你是谁？'}]}

    input_ids = convert_sample_to_input_ids(s, 'chat_template', tokenizer)
    print('-' * 50)
    print(input_ids)
    print('-' * 50)
    print(tokenizer.decode(input_ids))
    print('-' * 50)
    import pdb; pdb.set_trace()
