import os
import sys

os.environ['LOG_LEVEL'] = 'CRITICAL'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['NCCL_NVLS_ENABLE'] = '0'  # 如不设置，tensor_parallel=8时，腾讯云报错：misc/socket.cc:484 NCCL WARN socketStartConnect: Connect to 10.170.1.33<37999> failed : Cannot assign requested address
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'

sys.path.append(os.path.realpath(os.path.dirname(os.path.abspath(__file__))))
from adaptor_base import set_device, get_local_worker_id, EngineAdaptorBase

set_device()

if not os.getenv('VLLM_PORT'):
    os.environ['VLLM_PORT'] = str(1033 + int(get_local_worker_id()) * 100)  # 设为1000到6000之间，以避免与平台端口冲突

import argparse
import asyncio
import time
import torch
import transformers
import vllm

from packaging import version
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm.sampling_params import RequestOutputKind
from vllm.outputs import RequestOutput

from loguru import logger

import vllm.inputs

vllm_prompt_type = 0
if hasattr(vllm.inputs, 'PromptInputs') or hasattr(vllm.inputs, 'PromptType'):  # vllm 0.5.0之后，LLMEngine.generate 入参改为 PromptInputs，0.6.3之后改为 PromptType
    vllm_prompt_type = 1

is_v1_engine = version.parse(vllm.__version__) >= version.parse('0.8.0') and os.getenv('VLLM_USE_V1', '1') != '0'
is_vllm_v_0_9 = version.parse(vllm.__version__) >= version.parse('0.9.0')

from multimodal import load_multimodal_data


OUTPUT_TYPES = ['text', 'reward', 'loss', 'prompt_tokens', 'next_token_prob', 'cosyvoice2']


class VLLMEngineAdaptor(EngineAdaptorBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.waiting = set()
        self.idx2stime = {}  # 推理过程中最新step的时间戳

    def should_stop_adding_sample(self, n_added):
        return (
            n_added >= 2  # 一次最多添加 2 条样本
            or len(self.waiting) >= 2  # 超过 2 条样本在排队，暂停添加新样本
            or any(time.time() - v > 10. for v in self.idx2stime.values())  # 任意请求token延迟超过10秒，暂停添加新样本
        )

    def prepare_inputs(self, r: dict):
        params = r.get('generation_params', {})
        sample_n = params.get('n', args.sample_n)
        otype = r.get('output_type', args.output_type)
        if otype == 'beam_search' and sample_n == 1:
            otype = 'text'
        r['output_type'] = otype
        if otype == 'loss':
            sampling_params = SamplingParams(
                temperature=0,
                prompt_logprobs=1,
                max_tokens=1
            )
        elif otype == 'prompt_tokens':
            sampling_params = SamplingParams(
                temperature=0,
                prompt_logprobs=20,  # 默认top-20，后期需要自定义可改为入参传入
                max_tokens=1
            )
        elif otype == 'next_token_prob':
            sampling_params = SamplingParams(
                temperature=0,
                max_tokens=1,
                logprobs=params.get('top_logprobs_num'),
                # logprob_token_ids=r.get('next_token_ids')
            )
        elif otype == 'cosyvoice2':
            sampling_params = SamplingParams(
                temperature = 1,  # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
                top_p = 1,       # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
                top_k = 25,
                # "min_tokens": 80,       # 不支持设置最小的tokens数量设置，开启后vllm直接崩溃，无法启动
                # "presence_penalty": 1.0,    # 不支持设置
                # "frequency_penalty": 0.0,   # 不支持设置
                max_tokens = 2048,
                detokenize = False,          # 目前 vllm 0.7.3 v1版本中设置无效，待后续版本更新后减少计算
                ignore_eos = False,
                output_kind = RequestOutputKind.DELTA  # 设置为DELTA，如调整该参数，请同时调整llm_inference的处理代码
            )
            sampling_params.stop_token_ids = [6561, 6563]
        else:
            sampling_params = SamplingParams(
                max_tokens=params.get('max_new_tokens', args.max_new_tokens),
                temperature=params.get('temperature', args.temperature),
                top_p=params.get('top_p', args.top_p),
                stop=params.get('stop', args.stop),
                repetition_penalty=params.get('repetition_penalty', args.repetition_penalty),
                presence_penalty=params.get('presence_penalty', args.presence_penalty),
            )
        self.waiting.add(r['idx'])
        return dict(
            inputs=r['inputs'],
            sampling_params=sampling_params,
            request_id=r['idx'],
            otype=otype,
            sample_n=sample_n,
        )

    async def predict_sample(
        self,
        inputs: list[int] | dict,
        sampling_params: SamplingParams,
        request_id: str,
        otype: str,
        sample_n: int,
    ):
        if self.args.output_type == 'cosyvoice2':
            results_generator = await self.model.generate(inputs, request_id=request_id)
        elif sample_n > 1:
            self.waiting.discard(request_id)
            return await asyncio.gather(*[
                self.predict_sample(inputs, sampling_params, f'{request_id}_{i}', otype, sample_n=1)
                for i in range(sample_n)
            ])
        else:
            self.waiting.add(request_id)
            results_generator = await self.generate(inputs, request_id, sampling_params, otype)
        try:
            final_output = await asyncio.wait_for(
                self.get_generator_result(results_generator, request_id), self.args.max_time
            )
        except asyncio.TimeoutError:
            print(f'单条样本超时（{self.args.max_time}秒），已跳过，返回预测结果为None：{request_id}')
            await self.model.abort(request_id)
            final_output = None
        self.waiting.discard(request_id)
        self.idx2stime.pop(request_id, None)
        return final_output

    async def generate(
        self,
        inputs: list[int] | list[list[int]] | dict,
        request_id: str,
        sampling_params: SamplingParams,
        otype: str,
    ):
        """
        单条样本推理

        Args:
            model (`AsyncLLMEngine`):
                通过vllm加载的模型
            inputs (`list[int] | dict`):
                模型输入，文本输入的token ids（`list[int]`），或多模态的输入（`dict`）
            request_id (`str`):
                请求的唯一id
            sampling_params (`SamplingParams`):
                采样参数，参考vllm.sampling_params
        """
        if isinstance(inputs, dict):  # multimodal
            # inputs['multi_modal_data'] = load_multimodal_data(processor, **inputs['multi_modal_data'], return_dict=True)
            inputs['multi_modal_data'] = await asyncio.get_running_loop().run_in_executor(  # 使用多线程加载多模态数据
                None,
                load_multimodal_data,
                self.processor,
                inputs['multi_modal_data'].get('image'),
                inputs['multi_modal_data'].get('audio'),
                True,  # return_dict=True
            )

        else:  # list[int]
            inputs = dict(prompt=None, prompt_token_ids=inputs)

        if vllm_prompt_type == 1:  # vllm >= 0.5.3
            return self.model.generate(
                inputs,
                sampling_params=sampling_params,
                request_id=request_id,
            )
        else:
            return self.model.generate(
                **inputs,
                sampling_params=sampling_params,
                request_id=request_id,
            )

    async def get_generator_result(self, result_generator, request_id) -> RequestOutput | list[RequestOutput]:
        if isinstance(result_generator, list):
            final_output = await asyncio.gather(*[self.get_generator_result(g, request_id) for g in result_generator])
            return final_output

        final_output = None
        async for request_output in result_generator:
            self.waiting.discard(request_id)
            self.idx2stime[request_id] = time.time()
            final_output = request_output
        self.idx2stime.pop(request_id, None)
        return final_output

    def convert_final_output(self, inputs: dict, final_output: RequestOutput | list[RequestOutput], otype: str):
        """
        将vllm的输出转换为返回data server的数据格式

        Args:
            inputs (`dict`):
                通过调用data server获取的模型输入
            final_output (`RequestOutput | list[RequestOutput]`):
                vllm的输出
            otype (`str`):
                输出结果的类型，支持的类型见`OUTPUT_TYPES`
        """
        if final_output is None:
            return None
        if isinstance(final_output, list):
            return [self.convert_final_output(inputs, o, otype) for o in final_output]

        if otype == 'reward':
            return final_output.outputs[0].rewards
        elif otype == 'text':
            output_ids = [
                o.token_ids if isinstance(o.token_ids, (tuple, list)) else tuple(o.token_ids)  # vllm 0.5.5: array -> tuple
                for o in final_output.outputs
            ]
            return output_ids[0] if len(output_ids) == 1 else output_ids
        elif otype == 'beam_search':
            return final_output
        elif otype == 'loss':
            logprobs = [
                logprobs[token_id] if isinstance(logprobs[token_id], float) else logprobs[token_id].logprob  # vllm 0.2.5 是 float，vllm 0.4.0 是 Logprob
                for token_id, logprobs in zip(inputs['inputs'], final_output.prompt_logprobs)
                if logprobs
            ]
            return - sum(logprobs) / len(logprobs)
        elif otype == 'prompt_tokens':
            return [
                {
                    token_id: round(logprobs_j if isinstance(logprobs_j, float) else logprobs_j.logprob, 3)  # vllm 0.2.5 是 float，vllm 0.4.0 是 Logprob
                    for token_id, logprobs_j in logprobs_i.items()
                }
                for logprobs_i in final_output.prompt_logprobs
                if logprobs_i
            ]
        elif otype == 'next_token_prob':
            try:
                return [
                    [token_id, logprob if isinstance(logprob, float) else logprob.logprob]  # vllm 0.2.5 是 float，vllm 0.4.0 是 Logprob
                    for token_id, logprob in final_output.outputs[0].logprobs[0].items()
                ]
            except IndexError:
                return []
        elif otype == 'cosyvoice2':
            return final_output
        else:
            raise NotImplementedError(f'Unsupported output_type: "{otype}"')

    def load_model(self):
        """加载模型，以及processor（tokenizer）"""
        args: argparse.Namespace = self.args
        if  args.output_type == 'cosyvoice2':
            model_dir = args.model
            print(f"load cosyvoice2 model: {model_dir}")
            from utils.cosyvoice.model import CosyVoice2Model
            from hyperpyyaml import load_hyperpyyaml

            print(f"load cosyvoice2 yaml: {model_dir}/cosyvoice2.yaml")
            with open('{}/cosyvoice2.yaml'.format(model_dir), 'r') as f:
                # configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': args.model_dir + "/CosyVoice-BlankEN"})
                configs = load_hyperpyyaml(f, overrides={'qwen_pretrain_path': model_dir})
            # print("configs:", configs)
            print("model init...")
            model = CosyVoice2Model(
                    model_dir,
                    configs['flow'],
                    configs['hift'],
                    args.fp16
            )
            # print('wnq_debug::model is:', model)
            model.load(
                '{}/flow.pt'.format(model_dir),
                '{}/hift.pt'.format(model_dir),
            )
            if True or args.load_jit:
                model.load_jit('{}/flow.encoder.{}.zip'.format(model_dir, 'fp16' if args.fp16 is True else 'fp32'))
            if args.load_trt:
                model.load_trt('{}/flow.decoder.estimator.{}.mygpu.plan'.format(model_dir, 'fp16' if args.fp16 is True else 'fp32'),
                                    '{}/flow.decoder.estimator.fp32.onnx'.format(model_dir),
                                args.fp16)
            tokenizer = configs["get_tokenizer"]()
            self.model, self.processor = model, tokenizer
            return

        args.trust_remote_code = True
        config = transformers.AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)
        processor = transformers.AutoProcessor.from_pretrained(args.model, trust_remote_code=args.trust_remote_code)

        if args.output_type == 'reward':
            args.max_new_tokens = 1
            assert {
                'LlamaForRewardModelWithGating',
                'LlamaForSequenceClassification',
                'Qwen2ForRewardModel',
                'RewardModel',
            }.intersection(getattr(config, "architectures", [])), '不是Reward模型！'

        if hasattr(config, 'text_config'):  # e.g. Qwen2-Audio
            config = config.text_config

        rope_scaling = getattr(config, 'rope_scaling', {}) or {}
        rope_scaling_factor = 1. if rope_scaling.get('rope_type') == 'llama3' else rope_scaling.get('factor', 1.)
        max_model_length = int(
            getattr(config, 'max_position_embeddings', 0) * rope_scaling_factor
            or getattr(config, 'seq_length', 0)
        )

        try:
            import vllm.config.model as vllm_config
        except ModuleNotFoundError:  # vllm < v0.11
            import vllm.config as vllm_config
        get_config = vllm_config.get_config

        def _get_config(model, *_args, **kwargs):
            config = get_config(model, *_args, **kwargs)
            if 'RewardModel' in config.architectures:  # OpenRLHF训练的RewardModel
                if config.model_type == 'llama':
                    config.architectures.append('LlamaForRewardModel')
                elif config.model_type == 'qwen2':
                    config.architectures.append('Qwen2ForRewardModel')
                else:
                    raise ValueError('不支持当前Reward模型！')
            if any(config.__class__.__name__.startswith(custom_model) for custom_model in ['Dongwu', 'Hithinkgpt']):
                if getattr(config, 'qkv_bias', False):
                    config = transformers.Qwen2Config.from_pretrained(model)
                    config.architectures.append('Qwen2ForCausalLM')
                else:
                    config.architectures.append('MistralForCausalLM')
            text_config = getattr(config, 'text_config', config)
            if args.max_length and args.max_length != max_model_length:
                max_pe = int(args.max_length / rope_scaling_factor)
                print(f'Overriding max_position_embeddings: {text_config.max_position_embeddings} -> {max_pe}')
                text_config.max_position_embeddings = max_pe
            return config

        vllm_config.get_config = _get_config

        if is_v1_engine:
            if args.max_length is not None:
                args.max_model_len = args.max_length

        if args.low_vram:
            max_length = args.max_length or max_model_length
            if args.block_size is None:
                args.block_size = 32  # 默认值
            num_cache_blocks = int((max_length or 8192) * 1.1 / args.block_size)
            for key in ['num_gpu_blocks', 'num_cpu_blocks', 'forced_num_gpu_blocks', 'num_gpu_blocks_override']:
                if hasattr(args, key) and getattr(args, key) is None:
                    setattr(args, key, num_cache_blocks)
            if not is_vllm_v_0_9:
                args.gpu_memory_utilization = 1.0  # 防止出现类似报错：The model's max seq len (8192) is larger than the maximum number of tokens that can be stored in KV cache (7984). Try increasing `gpu_memory_utilization` or decreasing `max_model_len` when initializing the engine.
        # else:
        #     args.gpu_memory_utilization = 0.8  # 默认0.9，遇到过70B模型cache分配异常的情况
        if hasattr(args, 'disable_custom_all_reduce') and not is_v1_engine:
            args.disable_custom_all_reduce = True
        if hasattr(args, 'enforce_eager') and getattr(config, 'dual_chunk_attention_config', None):
            args.enforce_eager = True

        args.tensor_parallel_size = args.tensor_parallel
        if args.tensor_parallel_size > 1 and hasattr(args, 'enforce_eager') and int(torch.version.cuda.split('.')[0]) < 12:  # https://github.com/vllm-project/vllm/issues/7548
            args.enforce_eager = True

        if args.pipeline_parallel > 1:
            assert hasattr(args, 'pipeline_parallel_size')
            args.pipeline_parallel_size = args.pipeline_parallel
            # 减少DeepSeek-R1推理卡住的机率
            args.enable_chunked_prefill = True
            args.enable_prefix_caching = True
            args.max_num_batched_tokens = 8192

        if args.tensor_parallel_size > 1 and hasattr(config, 'moe_intermediate_size') and hasattr(args, 'enable_expert_parallel'):
            args.enable_expert_parallel = True  # MoE模型使用专家并行

        if "Processor" in type(processor).__name__:
            args.limit_mm_per_prompt = {"image":10}
        engine_args = AsyncEngineArgs.from_cli_args(args)
        logger.info(engine_args)
        model = AsyncLLMEngine.from_engine_args(engine_args)
        engine = getattr(model, 'engine_core', None) or model.engine
        model.log_requests = False
        engine.log_stats = False
        self.model, self.processor = model, processor


if __name__ == "__main__":
    if is_vllm_v_0_9:
        from vllm.utils import FlexibleArgumentParser as ArgumentParser  # vllm >= 0.9
        add_argument = ArgumentParser.add_argument
        def _add_argument(self, dest, *args, **kwargs):
            dest = dest.replace('_', '-')
            return add_argument(self, dest, *args, **kwargs)
        ArgumentParser.add_argument = _add_argument
    else:
        ArgumentParser = argparse.ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("--server_addr", type=str, required=True, help="Data server address")
    parser.add_argument("--server_port", type=int, required=True, help="Data server port")
    parser.add_argument("--output_type", type=str, default='text', choices=OUTPUT_TYPES, help="Output type")
    parser.add_argument("--max_length", type=int, default=None, help="Max number of tokens (input and output)")
    parser.add_argument("--max_time", type=int, help="Max number of seconds per sample")
    parser.add_argument("--max_new_tokens", type=int, default=2048, help="Max number of tokens to generate")
    parser.add_argument("--sample_n", "--beam_size", type=int, default=1, help="generate n outputs by sampling, or beam_size when output_type=beam search")
    parser.add_argument("--temperature", type=float, default=0., help="0.0 means greedy decoding")
    parser.add_argument("--top_p", type=float, default=1., help="1.0 means sampling from all tokens")
    parser.add_argument("--repetition_penalty", type=float, default=1., help="1.0 means no penalty")
    parser.add_argument("--presence_penalty", type=float, default=0., help="0.0 means no penalty")
    parser.add_argument("--stop", type=str, default=None, help="token/word at which generation will be stopped")
    parser.add_argument("--low_vram", action='store_true', help="Lower gpu memory usage")
    parser.add_argument("--tensor_parallel", type=int, default=1, help="Number of tensor parallel gpus")
    parser.add_argument("--pipeline_parallel", type=int, default=1, help="Number of pipeline stages")
    parser.add_argument("--load_jit", action='store_true', help="Load jit")
    parser.add_argument("--load_trt", action='store_true', help="Load trt")
    parser.add_argument("--fp16", action='store_true', help="fp16")
    parser = AsyncEngineArgs.add_cli_args(parser)
    args = parser.parse_args()

    args.stop = args.stop.replace('\\n', '\n').split(',') if args.stop else None

    asyncio.run(VLLMEngineAdaptor(args).run_predict_until_complete())
