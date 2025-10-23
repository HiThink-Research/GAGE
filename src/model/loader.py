import os
import requests
from typing import TYPE_CHECKING, Optional, Tuple

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.integrations import is_deepspeed_zero3_enabled
from transformers.utils.versions import require_version
from loguru import logger
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from ..hparams import FinetuningArguments, ModelArguments

# from trl import AutoModelForCausalLMWithValueHead
from ..extras.misc import count_parameters,get_current_device
from peft import LoraConfig, PeftModel, TaskType, get_peft_model

# from .adapter import init_adapter
# from .patcher import patch_config, patch_model, patch_tokenizer, patch_valuehead_model
# from .utils import load_valuehead_params, register_autoclass

    

# require_version("transformers>=4.36.2", "To fix: pip install transformers>=4.36.2")
# require_version("datasets>=2.14.3", "To fix: pip install datasets>=2.14.3")
# require_version("accelerate>=0.21.0", "To fix: pip install accelerate>=0.21.0")
# require_version("peft>=0.7.0", "To fix: pip install peft>=0.7.0")
# require_version("trl>=0.7.6", "To fix: pip install trl>=0.7.6")


class RemoteModel:
    """模型通过独立的推理进程加载，通过HTTP请求提交inputs和返回output"""

    OUTPUT_TYPES = ['text', 'reward', 'loss', 'next_token_prob']

    def __init__(self, url) -> None:
        self.url = url
        self.sess = requests.Session()

    def run_batch(self, input_ids, output_type='text', next_token_ids=None):
        assert output_type in self.OUTPUT_TYPES, f'"output_type" must be one of {self.OUTPUT_TYPES}!'
        # 支持多模态模型 输入改为inputs
        inputs = {'inputs': input_ids, 'output_type': output_type}
        if output_type == 'next_token_prob':
            inputs['next_token_ids'] = next_token_ids
        while True:
            try:
                r = self.sess.post(self.url, json=inputs)
                r = r.json()
                break
            except OSError:  # 有时会出现 ConnectionResetError: [Errno 104] Connection reset by peer，超过系统连接数限制？尝试再次连接
                # traceback.print_exc()
                continue
        # import pdb; pdb.set_trace()
        return r['output']


def load_model_and_tokenizer(
    model_args: "ModelArguments",
    finetuning_args: Optional[FinetuningArguments]=None,
    is_trainable: Optional[bool] = False,
    add_valuehead: Optional[bool] = False,
) -> Tuple["PreTrainedModel", "PreTrainedTokenizer"]:
    r"""
    Loads pretrained model and tokenizer.

    Support both training and inference.
    """

    # try_download_model_from_ms(model_args)

    config_kwargs = {
        "trust_remote_code": True,
        "cache_dir": model_args.cache_dir,
        "revision": model_args.model_revision,
        "token": model_args.hf_hub_token,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=False,
        split_special_tokens=model_args.split_special_tokens,
        padding_side="left",
        **config_kwargs,
    )
    # patch_tokenizer(tokenizer)

    remote_model_url = os.environ.get('REMOTE_MODEL_URL')
    if remote_model_url:
        model = RemoteModel(remote_model_url)
        return model, tokenizer

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, **config_kwargs)
    # patch_config(config, tokenizer, model_args, config_kwargs, is_trainable)

    model = None
    if is_trainable and model_args.use_unsloth:
        require_version("unsloth", "Follow the instructions at: https://github.com/unslothai/unsloth")
        from unsloth import FastLlamaModel, FastMistralModel  # type: ignore

        unsloth_kwargs = {
            "model_name": model_args.model_name_or_path,
            "max_seq_length": model_args.model_max_length,
            "dtype": model_args.compute_dtype,
            "load_in_4bit": model_args.quantization_bit == 4,
            "token": model_args.hf_hub_token,
            "device_map": get_current_device(),
            "rope_scaling": getattr(config, "rope_scaling", None),
        }
        if getattr(config, "model_type", None) == "llama":
            model, _ = FastLlamaModel.from_pretrained(**unsloth_kwargs)
        elif getattr(config, "model_type", None) == "mistral":
            model, _ = FastMistralModel.from_pretrained(**unsloth_kwargs)
        else:
            logger.warning("Unsloth does not support model type {}.".format(getattr(config, "model_type", None)))
            model_args.use_unsloth = False

        if model_args.adapter_name_or_path:
            model_args.adapter_name_or_path = None
            logger.warning("Unsloth does not support loading adapters.")

    if model is None:
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            config=config,
            device_map=get_current_device() if model_args.data_parallel else "auto",
            torch_dtype=model_args.compute_dtype,
            **config_kwargs,
        )
    if model_args.adapter_name_or_path != None:
        print(model_args.adapter_name_or_path)
        model = PeftModel.from_pretrained(model, model_args.adapter_name_or_path[0])
        model = model.merge_and_unload()
        logger.info("Merged {} adapter(s).".format(model_args.adapter_name_or_path))
    # patch_model(model, tokenizer, model_args, is_trainable)
    # register_autoclass(config, model, tokenizer)

    # model = init_adapter(model, model_args, finetuning_args, is_trainable)

    # if add_valuehead:
    #     model: "AutoModelForCausalLMWithValueHead" = AutoModelForCausalLMWithValueHead.from_pretrained(model)
    #     patch_valuehead_model(model)

    #     if model_args.adapter_name_or_path is not None:
    #         vhead_path = model_args.adapter_name_or_path[-1]
    #     else:
    #         vhead_path = model_args.model_name_or_path

    #     vhead_params = load_valuehead_params(vhead_path, model_args)
    #     if vhead_params is not None:
    #         model.load_state_dict(vhead_params, strict=False)
    #         logger.info("Loaded valuehead from checkpoint: {}".format(vhead_path))

    if not is_trainable:
        
        model.requires_grad_(False)
        
        # TODO when use gpu ,delete the code
        model_args.data_parallel = False
        # print(model_args.data_parallel)
        
        if model_args.data_parallel:
            model = torch.nn.DataParallel(model)
            model.to("cuda")
        # model = model.to(model_args.compute_dtype) if not getattr(model, "quantization_method", None) else model
        # else:
        #     ds_engine = deepspeed.init_inference(model)
        #     model = ds_engine.module
        model.eval()
    else:
        model.train()

    trainable_params, all_param = count_parameters(model)
    logger.info(
        "trainable params: {:d} || all params: {:d} || trainable%: {:.4f}".format(
            trainable_params, all_param, 100 * trainable_params / all_param
        )
    )

    if not is_trainable:
        logger.info("This IS expected that the trainable params is 0 if you are using model for inference only.")

    return model, tokenizer
