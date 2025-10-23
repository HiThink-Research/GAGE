import os
import sys

os.environ['SGLANG_DISABLE_REQUEST_LOGGING'] = 'true'
os.environ['SGLANG_SUPPORT_CUTLASS_BLOCK_FP8'] = 'true'

sys.path.append(os.path.realpath(os.path.dirname(os.path.abspath(__file__))))
from adaptor_base import set_device, EngineAdaptorBase

set_device()

import argparse
import asyncio
import subprocess
import time
import ujson as json
import requests
import aiohttp

from sglang.utils import launch_server_cmd, terminate_process

OUTPUT_TYPES = ['text', 'next_token_prob']


class SGlangEngineAdaptor(EngineAdaptorBase):

    async def run_predict_until_complete(self):

        try:
            if self.is_worker_node:
                await self.run_worker_until_complete()
            else:
                await super().run_predict_until_complete()
        except:
            terminate_process(self.server_proc)
            raise
        else:
            terminate_process(self.server_proc)

    async def run_worker_until_complete(self):
        """使用跨节点的模型并行时，worker节点（rank > 1）不需要做其他事情，只需要等待中止信号"""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout()) as self.session:  # 取消超时限制（默认5分钟）
            while True:
                try:
                    async with self.session.get(f'{self.server_url}/info') as r:
                        r = await r.text()
                        r = json.loads(r)
                except (
                    aiohttp.client_exceptions.ClientConnectionError,
                    asyncio.exceptions.TimeoutError
                ):
                    await asyncio.sleep(3)
                else:
                    if r['terminated']:
                        break
                    else:
                        await asyncio.sleep(1)

    def should_stop_adding_sample(self, n_added):
        return (
            n_added >= 2  # 一次最多添加 2 条样本
            or len(self.idx2task) >= (16 if self.args.tensor_parallel < 8 else 256)  # 超过 16 条样本在队列中，暂停添加新样本（如果tp >= 8，意味着dp=1，不需要关心负载均衡，可使用更大队列长度）
        )

    def prepare_inputs(self, r: dict):
        otype = r.get('output_type', args.output_type)
        params = r.get('generation_params', {})
        sample_n = params.get('n', args.sample_n)
        if otype == 'text':
            sampling_params = dict(
                n=sample_n,
                max_new_tokens=params.get('max_new_tokens', args.max_new_tokens),
                temperature=params.get('temperature', args.temperature),
                top_p=params.get('top_p', args.top_p),
                stop=params.get('stop', args.stop),
                presence_penalty=params.get('presence_penalty', args.presence_penalty),
            )
            params = dict(sampling_params=sampling_params)
        elif otype == 'next_token_prob':
            sampling_params = dict(
                max_new_tokens=1,
            )
            params = dict(
                sampling_params=sampling_params,
                return_logprob=True,
                top_logprobs_num=params.get('top_logprobs_num'),
                logprob_token_ids=r.get('token_ids_logprob'),
            )
        else:
            raise NotImplementedError(f'Unsupported output_type: "{otype}"')

        return dict(
            inputs=r['inputs'],
            params=params,
            request_id=r['idx'],
            otype=otype,
        )

    async def predict_sample(self, inputs: list[int] | dict, params: dict, request_id: str, otype: str) -> dict:
        """
        单条样本推理

        Args:
            inputs (`list[int] | dict`):
                模型输入，文本输入的token ids（`list[int]`），或多模态的输入（`dict`）
            sampling_params (`dict`):
                采样参数，调用/generate时传入
            request_id (`str`):
                请求的唯一id
        """
        if isinstance(inputs, dict):  # multimodal
            raise NotImplementedError('暂未接入SGLang多模态推理')

        else:  # list[int]
            if (n := params['sampling_params'].pop('n', 1)) > 1:
                return await asyncio.gather(*[
                    self.predict_sample(inputs, params, request_id, otype)
                    for _ in range(n)
                ])
            inputs = dict(input_ids=inputs, **params)
            max_retries = 10
            for i in range(max_retries):
                try:
                    async with self.session.post(f'http://127.0.0.1:{self.port}/generate', json=inputs) as r:
                        r = await r.text()
                        r = json.loads(r)
                    break
                except aiohttp.client_exceptions.ClientOSError:
                    if i < max_retries - 1:
                        await asyncio.sleep(10)
                    else:
                        raise
            return r

    def convert_final_output(self, inputs: dict, final_output, otype: str):
        if otype == 'next_token_prob':
            try:
                final_output = final_output['meta_info']['output_top_logprobs'][0]
            except (KeyError, IndexError):
                final_output = []
            else:
                final_output = [[token_id, logprob] for logprob, token_id, token_str in final_output]
        return final_output

    def load_model(self):
        """加载模型，启动推理引擎"""
        args = self.args
        cmd = [
            'python3', '-m', 'sglang.launch_server',
            '--model-path', args.model,
            '--log-level', 'warning',
            '--trust-remote-code',
        ]
        tp = args.tensor_parallel
        nnodes = tp // self.get_gpu_count()
        timeout = 3600
        self.is_worker_node = False

        if nnodes > 1:  # 多机，e.g. 2 * 8 * H100 启动DeepSeek-R1时，tp=16
            # 修复多节点加载模型超时的问题
            import sglang
            subprocess.run(f'sed -i s/\'UNBALANCED_MODEL_LOADING_TIMEOUT_S = 300\'/'
                        f'\'UNBALANCED_MODEL_LOADING_TIMEOUT_S = 1800\'/g '
                        f'{os.path.join(sglang.__path__[0], "srt/model_executor/model_runner.py")}', shell=True)

            cmd.extend([
                '--dist-init-addr', ':'.join([args.server_addr, os.environ['MASTER_PORT']]),
                '--nnodes', str(nnodes),
                '--node-rank', os.environ['RANK'],
                '--dist-timeout', str(timeout),
                '--enable-dp-attention',  # 提升高并发下的吞吐 https://docs.sglang.ai/references/deepseek.html#data-parallelism-attention
                '--dp', str(nnodes),
            ])
            if os.environ['RANK'] != '0':
                self.is_worker_node = True

        cmd.extend(['--tp', str(tp)])
        if args.max_length is not None:
            cmd.extend(['--context-length', str(args.max_length)])
        cmd.extend(self.additional_args)

        cmd = ' '.join(cmd)
        print('cmd:', cmd)
        self.server_proc, self.port = launch_server_cmd(cmd)

        # 等待模型server启动
        start_time = time.time()
        try:
            while True:
                try:
                    if not os.popen(f'ps auxww | grep sglang.launch_server | grep "\-\-port {self.port}" | grep -v grep').read():
                        raise RuntimeError('SGLang启动异常')
                    if self.is_worker_node:
                        break
                    r = requests.get(f"http://127.0.0.1:{self.port}/v1/models", headers={"Authorization": "Bearer None"})
                    if r.status_code == 200:
                        time.sleep(5)
                        break
                    if timeout and time.time() - start_time > timeout:
                        raise TimeoutError('SGLang启动超时')
                except requests.exceptions.RequestException:
                    time.sleep(1)
        except:
            terminate_process(self.server_proc)
            raise


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
    parser.add_argument("--tensor_parallel", type=int, default=1, help="Number of tensor parallel gpus")
    parser.add_argument("--low_vram", action='store_true', help="Lower gpu memory usage")
    args, engine_args = parser.parse_known_args()

    args.stop = args.stop.replace('\\n', '\n').split(',') if args.stop else None

    asyncio.run(SGlangEngineAdaptor(args, engine_args).run_predict_until_complete())
