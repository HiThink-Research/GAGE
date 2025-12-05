import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from eval_hle import generate_prompt
from transformers import AutoProcessor, PreTrainedTokenizer

def convert_sample_to_input_ids(
    s:dict,
    prompt_type: str,
    tokenizer: PreTrainedTokenizer
):
    question = s.get("question")
    image = s.get("image", "")

    if image:
        question = "<|vision_start|><|image_pad|><|vision_end|>"+question
        inputs = {
            "prompt": generate_prompt(question),
            "multi_modal_data": {"image": [image]}
        }
        return inputs
    else:
        contents = [
            {"type": "text", "text": generate_prompt(question)}
        ]
        messages = [
            {
                "role": "user",
                "content": contents
            }
        ]
        final_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )
        #处理当输入是多模态Processor的时候tokenizer异常
        if "Processor" in type(tokenizer).__name__:
            tokenizer = tokenizer.tokenizer
        input_ids = tokenizer.encode(final_prompt)
        return input_ids