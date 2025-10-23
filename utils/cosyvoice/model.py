import sys
import os
import torch
torch.backends.cudnn.enabled = False
import torchaudio
from typing import AsyncGenerator, Generator, List, Union
from hyperpyyaml import load_hyperpyyaml
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.sampling_params import SamplingParams
from vllm import ModelRegistry
from collections import OrderedDict
from torch.nn import functional as F
import uuid
from vllm.outputs import RequestOutput 

from types import SimpleNamespace
#from vllm_use_cosyvoice2_model import CosyVoice2Model as CosyVoice2LLM
from vllm.sampling_params import RequestOutputKind
from vllm.model_executor.models.cosyvoice2 import CosyVoice2Model as CosyVoice2LLM
ModelRegistry.register_model("CosyVoice2Model", CosyVoice2LLM)

from cosyvoice.flow.flow import CausalMaskedDiffWithXvec
from cosyvoice.hifigan.generator import HiFTGenerator
from cosyvoice.utils.common import fade_in_out
from cosyvoice.utils.file_utils import convert_onnx_to_trt
import numpy as np
import asyncio
import os
import time
import queue
from vllm.lora.request import LoRARequest
from typing import Optional, Union
from vllm.inputs import ProcessorInputs, PromptType
from vllm.inputs.parse import split_enc_dec_inputs
# v0.7.2
#from vllm.engine.llm_engine import LLMEngine
#from vllm.transformers_utils.tokenizer import AnyTokenizer
#def new_validate_token_prompt(self, prompt: PromptType,
#                               tokenizer: AnyTokenizer):
#    pass
#LLMEngine._validate_token_prompt = new_validate_token_prompt

def new_validate_model_inputs(self,
                               inputs: ProcessorInputs,
                               lora_request: Optional[LoRARequest] = None):
    encoder_inputs, decoder_inputs = split_enc_dec_inputs(inputs)

    # For encoder-decoder multimodal models, the max_prompt_len
    # restricts the decoder prompt length
    if self.model_config.is_multimodal_model:
        prompt_inputs = decoder_inputs
    else:
        prompt_inputs = encoder_inputs or decoder_inputs

    prompt_ids = prompt_inputs["prompt_token_ids"]

    if prompt_ids is None or len(prompt_ids) == 0:
        raise ValueError("Prompt cannot be empty")

    max_input_id = max(prompt_ids)
    max_allowed = self.tokenizer.get_lora_tokenizer(
            lora_request).max_token_id
    #if max_input_id > max_allowed:
    #    print("Token id {} is out of vocabulary".format(max_input_id))
    #raise ValueError(
    #        "Token id {} is out of vocabulary".format(max_input_id))

    if len(prompt_ids) >= self.model_config.max_model_len:
        raise ValueError(
                f"Prompt length of {len(prompt_ids)} is longer than the "
                f"maximum model length of {self.model_config.max_model_len}.")

    if self.model_config.is_multimodal_model:
        max_prompt_len = self.model_config.max_model_len

        if len(prompt_ids) > max_prompt_len:
            raise ValueError(
                    f"The prompt (total length {len(prompt_ids)}) is too long "
                    f"to fit into the model (context length {max_prompt_len}). "
                    "Make sure that `max_model_len` is no smaller than the "
                    "number of text tokens plus multimodal tokens. For image "
                    "inputs, the number of image tokens depends on the number "
                    "of images, and possibly their aspect ratios as well.")

        # TODO: Find out how many placeholder tokens are there so we can
        # check that chunked prefill does not truncate them
        # max_batch_len = self.scheduler_config.max_num_batched_tokens
from vllm.v1.engine.processor import Processor
original_validate_model_inputs = Processor._validate_model_inputs
Processor._validate_model_inputs = new_validate_model_inputs


ENGINE_ARGS = {
    # "enforce_eager": True,
    "gpu_memory_utilization": 0.8,
    "max_num_batched_tokens": 1024,
    "max_model_len": 1024,
    "max_num_seqs": 256,
    "disable_log_requests": False,
    "disable_log_stats": False,
    "dtype": "float16",
}

SAMPLING_PARAMS = {
    "temperature": 1,  # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
    "top_p": 1,       # 不能低于0.8, 否则会生成非常多的空音频，或者无法正常生成语音Token
    "top_k": 25,
    # "min_tokens": 80,       # 不支持设置最小的tokens数量设置，开启后vllm直接崩溃，无法启动
    # "presence_penalty": 1.0,    # 不支持设置
    # "frequency_penalty": 0.0,   # 不支持设置
    "max_tokens": 1024,
    #"detokenize": False,          # 目前 vllm 0.7.3 v1版本中设置无效，待后续版本更新后减少计算
    #"ignore_eos": False,
    "output_kind": RequestOutputKind.DELTA  # 设置为DELTA，如调整该参数，请同时调整llm_inference的处理代码
}

class CosyOutput:
    def __init__(self, outputs, request_id):
        self.outputs = outputs
        self.index = 0
    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.outputs):
            raise StopAsyncIteration
        result = self.outputs[self.index]
        self.index += 1
        return result

class LRUCache(OrderedDict):
    """LRU缓存容器，继承自OrderedDict"""

    def __init__(self, max_size=100_000):
        super().__init__()
        self.max_size = max_size

    def __getitem__(self, key):
        # 访问时移动到末尾（表示最新）
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        # 插入时检查容量，超限则移除最旧项
        if key in self:
            self.move_to_end(key)
        else:
            if len(self) >= self.max_size:
                self.popitem(last=False)
        super().__setitem__(key, value)

def tensor_to_list(tensor: torch.tensor):
    return tensor.view(-1).cpu().numpy().tolist()

class CosyVoice2Model:
    def __init__(self,
         model_dir: str,
         flow: CausalMaskedDiffWithXvec | torch.nn.Module,
         hift: HiFTGenerator | torch.nn.Module,
         fp16: bool,
         mix_ratio: List[int] = None,
         use_flow_cache: bool = False, sample_rate=24000, spk_id='xiaohe'):
        # vllm engine 的参数配置
        engine_args = AsyncEngineArgs(
            model=model_dir,
            **ENGINE_ARGS,
        )
        self.llm_engine: AsyncLLMEngine = AsyncLLMEngine.from_engine_args(engine_args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.thread_count = 10
        self.stream_pool = queue.Queue()
        for _ in range(self.thread_count):
            stream = torch.cuda.Stream(self.device)
            self.stream_pool.put(stream)

        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.mel_overlap_dict = {}
        self.flow_cache_dict = {}
        self.hift_cache_dict = {}

        self.flow = flow
        self.hift = hift
        self.fp16 = fp16
        self.flow.fp16 = fp16
        if self.fp16 is True:
            self.flow.half()
        # self.token_hop_len = 2 * self.flow.input_frame_rate
        self.token_hop_len = 25
        self.flow_decoder_required_cache_size = 0 if use_flow_cache is False else 1 * self.token_hop_len * self.flow.token_mel_ratio

        self.use_flow_cache = use_flow_cache

        # here we fix flow encoder/decoder decoding_chunk_size, in the future we will send it as arguments, or use cache
        self.flow.encoder.static_chunk_size = 2 * self.flow.input_frame_rate
        self.flow.decoder.estimator.static_chunk_size = 2 * self.flow.input_frame_rate * self.flow.token_mel_ratio
        # hift cache
        self.mel_cache_len = 8
        self.source_cache_len = int(self.mel_cache_len * 480)
        # speech fade in out
        self.speech_window = np.hamming(2 * self.source_cache_len)
        # rtf and decoding related
        self.stream_scale_factor = 1
        self.llm_context = torch.cuda.stream(torch.cuda.Stream(self.device)) if torch.cuda.is_available() else nullcontext()

        self.mix_ratio = mix_ratio or [5, 15]

        # self.lock = asyncio.Lock()  # 改为异步锁

        # dict used to store session related variable
        self.tts_speech_token_dict = {}
        self.llm_end_dict = {}
        self.hift_cache_dict = {}

        # 与vllm中的模型保持一致
        self.speech_token_size = 6564
        self.llm_token_size = 151936
        self.sos_eos_token_id = self.speech_token_size + self.llm_token_size + 1
        self.task_token_id = self.sos_eos_token_id + 1
        self.zero_token_id = self.task_token_id + 1

        # load speaker info:
        spk_info_path =  f"{model_dir}/spk2info.pt"
        self.spk_infos = torch.load(spk_info_path, map_location=self.device, weights_only=False)
        self.spk_info_cache = LRUCache(max_size=10000)
        self.spk_id=spk_id
        for spk_id, info in self.spk_infos.items():
            self.spk_info_cache[self.spk_id] = info
        #print("spk_info:", self.spk_infos)
        self.spk_embedding = self.spk_info_cache[spk_id]['embedding']
        assert self.spk_embedding is not None

        self.sample_rate = sample_rate

        self.save_dir = os.environ.get('COSY_AUDIO_OUTPUT_DIR')
        assert self.save_dir is not None, f"audio save dir env {COSY_AUDIO_OUTPUT_DIR} is none!!!"


    def load(self, flow_model, hift_model):
        self.flow.load_state_dict(torch.load(flow_model, weights_only=True, map_location=self.device), strict=True)
        self.flow.to(self.device).eval()
        # in case hift_model is a hifigan model
        hift_state_dict = {k.replace('generator.', ''): v for k, v in torch.load(hift_model, weights_only=True, map_location=self.device).items()}
        self.hift.load_state_dict(hift_state_dict, strict=True)
        self.hift.to(self.device).eval()

    def load_jit(self, flow_encoder_model):
        flow_encoder = torch.jit.load(flow_encoder_model, map_location=self.device)
        self.flow.encoder = flow_encoder

    def load_trt(self, flow_decoder_estimator_model, flow_decoder_onnx_model, fp16):
        assert torch.cuda.is_available(), 'tensorrt only supports gpu!'
        if not os.path.exists(flow_decoder_estimator_model):
            convert_onnx_to_trt(flow_decoder_estimator_model, flow_decoder_onnx_model, fp16=fp16)
        if os.path.getsize(flow_decoder_estimator_model) == 0:
            raise ValueError('{} is empty file, delete it and export again!'.format(flow_decoder_estimator_model))
        del self.flow.decoder.estimator
        import tensorrt as trt
        with open(flow_decoder_estimator_model, 'rb') as f:
            self.flow.decoder.estimator_engine = trt.Runtime(trt.Logger(trt.Logger.INFO)).deserialize_cuda_engine(f.read())
        if self.flow.decoder.estimator_engine is None:
            raise ValueError('failed to load trt {}'.format(flow_decoder_estimator_model))
        self.flow.decoder.estimator = EstimatorWrapper(self.flow.decoder.estimator_engine, estimator_count=ESTIMATOR_COUNT)

    def token2wav(self, token, prompt_token, prompt_feat, embedding, this_uuid, token_offset, finalize=False, speed=1.0):
        self.tts_speech_token_dict[this_uuid], self.llm_end_dict[this_uuid] = [], False
        self.mel_overlap_dict[this_uuid] = torch.zeros(1, 80, 0)
        self.flow_cache_dict[this_uuid] = torch.zeros(1, 80, 0, 2)
        self.hift_cache_dict[this_uuid] = None
        self.flow_cache_dict[this_uuid] = self.init_flow_cache()

        torch.cuda.current_stream().synchronize() # 将当前流进行同步了再处理后续逻辑
        stream = self.stream_pool.get()
        with torch.cuda.stream(stream):
            tts_mel, self.flow_cache_dict[this_uuid] = self.flow.inference(token=token.to(self.device),
                                             token_len=torch.tensor([token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_token=prompt_token.to(self.device),
                                             prompt_token_len=torch.tensor([prompt_token.shape[1]], dtype=torch.int32).to(self.device),
                                             prompt_feat=prompt_feat.to(self.device),
                                             prompt_feat_len=torch.tensor([prompt_feat.shape[1]], dtype=torch.int32).to(self.device),
                                             embedding=embedding.to(self.device),
                                             cache=self.flow_cache_dict[this_uuid],
                                             finalize=finalize)
            tts_mel = tts_mel[:, :, token_offset * self.flow.token_mel_ratio:]
            # append hift cache
            if self.hift_cache_dict[this_uuid] is not None:
                hift_cache_mel, hift_cache_source = self.hift_cache_dict[this_uuid]['mel'], self.hift_cache_dict[uuid]['source']
                tts_mel = torch.concat([hift_cache_mel, tts_mel], dim=2)
            else:
                hift_cache_source = torch.zeros(1, 1, 0)
            # keep overlap mel and hift cache
            if finalize is False:
                tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
                if self.hift_cache_dict[this_uuid] is not None:
                    tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[this_uuid]['speech'], self.speech_window)
                self.hift_cache_dict[this_uuid] = {'mel': tts_mel[:, :, -self.mel_cache_len:],
                                              'source': tts_source[:, :, -self.source_cache_len:],
                                              'speech': tts_speech[:, -self.source_cache_len:]}
                tts_speech = tts_speech[:, :-self.source_cache_len]
            else:
                if speed != 1.0:
                    assert self.hift_cache_dict[this_uuid] is None, 'speed change only support non-stream inference mode'
                    tts_mel = F.interpolate(tts_mel, size=int(tts_mel.shape[2] / speed), mode='linear')
                # print("tts_mel:", tts_mel)
                # print("cache:", hift_cache_source)
                tts_speech, tts_source = self.hift.inference(speech_feat=tts_mel, cache_source=hift_cache_source)
                if self.hift_cache_dict[this_uuid] is not None:
                    tts_speech = fade_in_out(tts_speech, self.hift_cache_dict[this_uuid]['speech'], self.speech_window)
            torch.cuda.synchronize(torch.cuda.current_stream())
            self.stream_pool.put(stream)
            return tts_speech

    def init_flow_cache(self):
        encoder_cache = {'offset': 0,
                         'pre_lookahead_layer_conv2_cache': torch.zeros(1, 512, 2).to(self.device),
                         'encoders_kv_cache': torch.zeros(6, 1, 8, 0, 64 * 2).to(self.device),
                         'upsample_offset': 0,
                         'upsample_conv_cache': torch.zeros(1, 512, 4).to(self.device),
                         'upsample_kv_cache': torch.zeros(4, 1, 8, 0, 64 * 2).to(self.device)}
        decoder_cache = {'offset': 0,
                         'down_blocks_conv_cache': torch.zeros(10, 1, 2, 832, 2).to(self.device),
                         'down_blocks_kv_cache': torch.zeros(10, 1, 4, 2, self.flow_decoder_required_cache_size, 512, 2).to(self.device),
                         'mid_blocks_conv_cache': torch.zeros(10, 12, 2, 512, 2).to(self.device),
                         'mid_blocks_kv_cache': torch.zeros(10, 12, 4, 2, self.flow_decoder_required_cache_size, 512, 2).to(self.device),
                         'up_blocks_conv_cache': torch.zeros(10, 1, 2, 1024, 2).to(self.device),
                         'up_blocks_kv_cache': torch.zeros(10, 1, 4, 2, self.flow_decoder_required_cache_size, 512, 2).to(self.device),
                         'final_blocks_conv_cache': torch.zeros(10, 2, 256, 2).to(self.device)}
        if self.fp16 is True:
            for cache in [encoder_cache, decoder_cache]:
                for k, v in cache.items():
                    if isinstance(v, torch.Tensor):
                        cache[k] = v.half()
        cache = {'encoder_cache': encoder_cache, 'decoder_cache': decoder_cache}
        return cache

    async def async_tts(self, text, flow_embedding, llm_embedding=torch.zeros(0, 192),
            prompt_text=torch.zeros(1, 0, dtype=torch.int32),
            llm_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32),
            prompt_speech_feat=torch.zeros(1, 0, 80), stream=False, speed=1.0, 
            stop_token_ids=None, request_id=None, **kwargs):

        # this_uuid is used to track variables related to this inference thread
        # queue: asyncio.Queue[int|None] = asyncio.Queue()
        # print("async_tts_text:", text)
        # print("async_tts_flow", flow_embedding)
        text = tensor_to_list(text + torch.tensor(6564))
        # print("llm generate input(text):", text)
        prompt_text = tensor_to_list(prompt_text + torch.tensor(6564))
        # print("llm generate input(prompt_text):", text)
        # print("llm prompt speech token: ", llm_prompt_speech_token)
        llm_prompt_speech_token = tensor_to_list(llm_prompt_speech_token)
        # print("llm prompt speech token: ", llm_prompt_speech_token)

        prompt_token_ids = [self.sos_eos_token_id] + prompt_text + text + \
                               [self.task_token_id] + llm_prompt_speech_token
        max_tokens = len(text) * 20

        #llm engine:
        sampling_params = SamplingParams(**SAMPLING_PARAMS)
        sampling_params.stop_token_ids = stop_token_ids or [6561, 6563]

        if max_tokens:
            sampling_params.max_tokens = max_tokens
        # print("prompt_token_ids:", prompt_token_ids)
        ret_tokens = []
        async for output in self.llm_engine.generate(
                {
                    "prompt_token_ids": prompt_token_ids,
                },
                sampling_params=sampling_params,
                request_id=request_id or f"{time.time()}",
        ):  
            result = output.outputs[0]
            finished = output.finished
            # print("result:", result)
            # print("finished:", finished)
            if result.token_ids[-1] >= 6561:
                need_add_tokens = result.token_ids[:-1]
            else:
                need_add_tokens = result.token_ids
            ret_tokens.extend(need_add_tokens)
            if finished:
                break

        this_tts_speech_token = torch.tensor(ret_tokens).unsqueeze(dim=0)
        yield {"tts_speech_token": this_tts_speech_token}
        #print("this_tts_speech_token:", this_tts_speech_token)
        #this_uuid = str(uuid.uuid1())
        #this_tts_speech = self.token2wav(this_tts_speech_token,
        #                flow_prompt_speech_token,
        #                prompt_speech_feat,
        #                flow_embedding,
        #                this_uuid,
        #                0, True, speed)
        #print("this_tts:", this_tts_speech)
        #yield {'tts_speech': this_tts_speech.cpu()}

    async def generate(self, model_inputs, request_id):
        # print("generate inputs:", model_inputs)
        tts_text_token = torch.tensor([model_inputs], dtype=torch.int32)
        tts_text_token_len = torch.tensor([tts_text_token.shape[1]], dtype=torch.int32)
        model_input = {'text': tts_text_token, 
                        'text_len': tts_text_token_len, 
                        'llm_embedding': self.spk_embedding, 
                        'flow_embedding': self.spk_embedding
                    }
        # print("model_input:", model_input)

        i = 0
        async for j in self.async_tts(**model_input):
            this_uuid = str(request_id)
            flow_prompt_speech_token=torch.zeros(1, 0, dtype=torch.int32)
            prompt_speech_feat=torch.zeros(1, 0, 80)
            stream=False
            speed=1.0
            
            this_tts_speech = self.token2wav(j["tts_speech_token"],
                          flow_prompt_speech_token,
                          prompt_speech_feat,
                          self.spk_embedding,
                          this_uuid,
                          0, True, speed)
            # print("this_tts_speech:", this_tts_speech)
            output_path = os.path.join(self.save_dir, f'sft_{i}_{request_id}.wav')
            # print("output_path is:", output_path)
            # torchaudio.save('sft_{}.wav'.format(request_id), this_tts_speech.cpu(), sample_rate)
            torchaudio.save(output_path, this_tts_speech.cpu(), self.sample_rate)
            ret = CosyOutput(
                outputs = [output_path],
                request_id = request_id
            )
            # print("ret: ", ret)
            return ret
            #print("this_tts_speech:", this_tts_speech.cpu())
            #torchaudio.save('sft_{}.wav'.format(i), this_tts_speech.cpu(), self.sample_rate)
            i += 1
            