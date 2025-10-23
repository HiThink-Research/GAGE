import os
import subprocess
import time
from vllm.lora.request import LoRARequest
from typing import Optional, Union
from vllm.inputs import ProcessorInputs, PromptType
# from vllm.inputs.parse import split_enc_dec_inputs
from hyperpyyaml import load_hyperpyyaml
import json

cur_path = os.path.dirname(os.path.abspath(__file__))

from vllm.v1.engine.processor import Processor

def prepare_cosyvoice_env(model_dir):
    print(f"[prepare_cosyvoice_env]cur_path: {cur_path}")
    
    print("copy model file to vllm")
    sub_run_file = os.path.join(cur_path, "copy_model_to_vllm.py")
    env = os.environ.copy()
    cmd = [
            'python', sub_run_file,
    ]
    result = subprocess.Popen(cmd, cwd=os.path.dirname(sub_run_file), env=env)
    result.wait()

    #print("monkey patching...")
    #original_validate_model_inputs = Processor._validate_model_inputs
    #Processor._validate_model_inputs = new_validate_model_inputs
    
    update_cosyvoice_config(model_dir)

    disable_cudnn()

    print('[Done] Prepare env done.')

def update_cosyvoice_config(model_dir):
    print("[INFO] update cosyvoice config")
    config_filename = 'config.json'
    config_filepath = os.path.join(model_dir, config_filename)
    try:
        with open(config_filepath, 'r', encoding='utf-8') as file:
            config = json.load(file)
    except FileNotFoundError:
        print("错误: config.json文件未找到！")
    except json.JSONDecodeError:
        print("错误: 无法解析JSON文件！")
    except Exception as e:
        print(f"错误: 发生了一个未知错误: {e}")
    print('[SRC] config is:', config)
    
    config.pop('architectures')

    config['architectures'] = ['CosyVoice2Model']
    print('[Updated] config is:', config)
    with open(config_filepath, 'w', encoding='utf-8') as file:
        json.dump(config, file, ensure_ascii=False, indent=4)

def disable_cudnn():
    print("[INFO]disable cudnn")
    import torch
    torch.backends.cudnn.enabled = False

def load_cosyvoice2_tokenizer(model_dir):
    with open('{}/cosyvoice2.yaml'.format(model_dir), 'r') as f:
        # configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': args.model_dir + "/CosyVoice-BlankEN"})
        configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': model_dir})
    tokenizer = configs["get_tokenizer"]()
    print("[INFO] load cosyvoice2 tokenizer done.")
    return tokenizer
