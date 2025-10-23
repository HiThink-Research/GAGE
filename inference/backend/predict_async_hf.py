import os
import sys

os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

sys.path.append(os.path.realpath(os.path.dirname(os.path.abspath(__file__))))
from adaptor_base import set_device, EngineAdaptorBase

set_device()
print(os.environ)
import argparse
import asyncio
import inspect
import torch
import transformers

from multimodal import load_multimodal_data


OUTPUT_TYPES = ['text', 'reward', 'loss', 'prompt_tokens', 'next_token_prob', 'turn_taking']


class HFEngineAdaptor(EngineAdaptorBase):
    """调用HuggingFace transformers实现的推理代码"""

    def should_stop_adding_sample(self, n_added):
        return (
            n_added >= 2  # 一次最多添加 2 条样本
            or len(self.idx2task) >= 8  # 超过 8 条样本在队列中，暂停添加新样本
        )

    def prepare_inputs(self, r: dict):
        otype = r.get('output_type', args.output_type)
        params = r.get('generation_params', {})
        if otype == 'text':
            kwargs = dict(
                do_sample=(args.temperature != 0.),
                temperature=args.temperature if args.temperature != 0 else None,
                top_p=args.top_p,
                repetition_penalty=args.repetition_penalty,
                # stop=args.stop,  # Qwen2-Audio-7B-Instruct @ transformers 4.51.3报错，待修复
                max_new_tokens=params.get('max_new_tokens', args.max_new_tokens),
            )
        elif otype == 'turn_taking':
            kwargs = dict()
        else:
            raise NotImplementedError(f'Unsupported output_type: "{otype}"')

        return dict(
            inputs=r['inputs'],
            kwargs=kwargs,
            request_id=r['idx'],
            otype=otype,
        )

    async def predict_sample(
        self,
        inputs: list[int] | dict,
        kwargs: dict,
        request_id: str,
        otype: str,
    ):
        """
        单条样本推理

        Args:
            model (`transformers.PreTrainedModel`):
                通过transformers加载的模型
            inputs (`list[int] | dict`):
                模型输入，文本输入的token ids（`list[int]`），或多模态的输入（`dict`）
            kwargs (`dict`):
                采样参数，调用model.generate()时传入
            request_id (`str`):
                请求的唯一id
            otype (`str`):
                模型输出类型
        """
        model: transformers.PreTrainedModel = self.model
        processor = self.processor
        if isinstance(inputs, dict):  # multimodal
            # images, audios = load_multimodal_data(processor, **inputs['multi_modal_data'])
            images, audios = await asyncio.get_running_loop().run_in_executor(  # 使用多线程加载多模态数据
                None,
                load_multimodal_data,
                processor,
                inputs['multi_modal_data'].get('image'),
                inputs['multi_modal_data'].get('audio'),
            )
            inputs = dict(
                text=[inputs['prompt']],
                **{k: v for k, v in [('images', images), ('audios', audios)] if v is not None},
                return_tensors="pt",
            )
            if audios:
                try:
                    inputs['sampling_rate'] = processor.feature_extractor.sampling_rate  # 传入默认值，否则一些模型会打印warning
                except AttributeError:
                    pass
                # 一些模型如Qwen2.5-Omni，以及transformers 4.54.0之后不再使用audios，改为audio
                if (
                    'audios' not in inspect.signature(processor).parameters
                    and 'audio' in inspect.signature(processor).parameters
                ):
                    inputs['audio'] = inputs.pop('audios')

            input_tensors = processor(**inputs).to(model.device)
            input_len =  input_tensors['input_ids'].shape[1]

        else:  # list[int]
            input_len = len(inputs)  # input_ids
            input_tensors = dict(
                inputs=torch.tensor([inputs], device=model.device),
                attention_mask=torch.ones((1, input_len), dtype=torch.long, device=model.device)
            )

        if 'return_audio' in inspect.signature(model.generate).parameters:  # Qwen2.5-Omni，不输出音频
            kwargs['return_audio'] = False

        eos_token_id = getattr(model.config, 'text_config', model.config).eos_token_id
        if eos_token_id is not None:
            kwargs['pad_token_id'] = eos_token_id

        if otype == 'text':
            output_ids = model.generate(
                **input_tensors,
                **kwargs,
            )  # input + output, shape: (batch_size, total_len)

            output = output_ids[0].tolist()[input_len:]

        elif otype == 'turn_taking':
            model = getattr(model, 'thinker', model)  # Qwen2.5-Omni
            outputs = model(
                **input_tensors,
                output_logits=False,
                is_turn_taking=True,
            )
            audio_token_id = model.config.audio_token_index
            mask = (input_tensors['input_ids'][0] == audio_token_id).cpu().numpy()[::-1]
            i1 = mask.argmax()            # 最后一段连续的audio_token的结束位置
            i0 = mask[i1:].argmin() + i1  # 最后一段连续的audio_token的开始位置
            output = outputs.logits[0, -i0: -i1].squeeze(-1).tolist()

        return output

    def load_model(self):
        """加载模型，以及processor（tokenizer）"""
        args: argparse.Namespace = self.args
        config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=True)
        processor = transformers.AutoProcessor.from_pretrained(args.model, trust_remote_code=True)

        kwargs = dict(device_map='cuda', torch_dtype='auto', attn_implementation='flash_attention_2')
        if args.dtype == 'fp16':
            kwargs.update(torch_dtype=torch.float16)
        elif args.dtype == 'bf16':
            kwargs.update(torch_dtype=torch.bfloat16)

        print('loading model...')
        exception = None

        model_classes =  getattr(config, 'architectures', [])
        for s in model_classes:
            if s.endswith('Model'):
                s = s[:-5] + 'ForConditionalGeneration'  # Qwen2_5OmniModel -> Qwen2_5OmniForConditionalGeneration
                if s not in model_classes:
                    model_classes.append(s)
        model_classes.extend(['AutoModelForCausalLM', 'AutoModel'])

        for model_class in model_classes:
            try:
                model_class = getattr(transformers, model_class)
            except AttributeError:
                continue
            try:
                model = model_class.from_pretrained(args.model, trust_remote_code=True, **kwargs)
                exception = None
                break
            except ValueError as e:
                exception = e
                continue
        if exception is not None:
            raise exception

        if hasattr(model.config, 'use_cache') and not model.config.use_cache:
            print('setting model.config.use_cache = True')
            model.config.use_cache = True

        elif args.dtype == 'fp16':
            model.half()

        self.model, self.processor = model, processor


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model file path")
    parser.add_argument("--server_addr", type=str, required=True, help="Data server address")
    parser.add_argument("--server_port", type=int, required=True, help="Data server port")
    parser.add_argument("--output_type", type=str, default='text', choices=OUTPUT_TYPES, help="Output type")
    parser.add_argument("--max_length", type=int, default=None, help="Max number of tokens (input and output)")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max number of tokens to generate")
    parser.add_argument("--sample_n", "--beam_size", type=int, default=1, help="generate n outputs by sampling, or beam_size when output_type=beam search")
    parser.add_argument("--temperature", type=float, default=0., help="0.0 means greedy decoding")
    parser.add_argument("--top_p", type=float, default=1., help="1.0 means sampling from all tokens")
    parser.add_argument("--repetition_penalty", type=float, default=1., help="1.0 means no penalty")
    parser.add_argument("--presence_penalty", type=float, default=0., help="0.0 means no penalty")
    parser.add_argument("--stop", type=str, default=None, help="token/word at which generation will be stopped")
    parser.add_argument("--dtype", type=str, default='auto', help="dtype of the loaded model")
    parser.add_argument("--low_vram", action='store_true', help="Lower gpu memory usage")
    args = parser.parse_args()

    assert args.sample_n, "sample_n > 1 not supported yet with hf backend!"
    args.stop = args.stop.replace('\\n', '\n').split(',') if args.stop else None

    asyncio.run(HFEngineAdaptor(args).run_predict_until_complete())
