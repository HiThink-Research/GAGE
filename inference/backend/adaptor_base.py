import os
import sys
import argparse
import asyncio
import time
import ujson as json
import aiohttp
import subprocess

from collections import OrderedDict


def set_device():
    """根据启动参数（`--device`），设置当前进程可用的GPU"""
    for i in range(len(sys.argv)):
        if sys.argv[i] == '--device':
            os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[i + 1]
            del sys.argv[i: i + 2]
            break


def get_local_worker_id() -> str:
    if devices := os.getenv('CUDA_VISIBLE_DEVICES'):
        return devices.split(',')[0]
    return '0'


class EngineAdaptorBase:
    """
    引擎适配器基类。作为DataServer和具体推理引擎（vLLM/SGLang/...）之间的桥梁，负责数据（模型输入/输出）的传输以及引擎生命周期的管理

                  GET /data                   .generate()
    DataServer <-------------- EngineAdaptor -------------> Engine (HF/vLLM/SGLang/...)
                 POST /result
    """

    def __init__(self, args: argparse.Namespace, additional_args: list[str] = None, max_concurrent_fetch: int = 4):
        self.args = args
        self.additional_args = additional_args
        self.max_concurrent_fetch = max_concurrent_fetch  # 请求推理数据的最大并发数
        self.server_url = f'http://{self.args.server_addr}:{self.args.server_port}'
        self.n_concurrent_fetch = 0
        self.idx2task = OrderedDict()
        self.r_latest = {}
        self.worker_info = {'rank': os.getenv('RANK', 'unknown'), 'worker_id': get_local_worker_id()}
        print('loading model...')
        self.load_model()

    async def run_predict_until_complete(self):
        """引擎适配器的主要运行函数，异步请求data server获取待预测样本，提交模型推理，最后把结果推送给data server，直到所有数据推理完成"""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout()) as self.session:  # 取消超时限制（默认5分钟）
            await self.add_sample_until_full()
            while True:
                if self.r_latest.get('status_code', 0) == 2:  # terminated
                    break
                elif self.idx2task:
                    await asyncio.gather(*self.idx2task.values())
                else:
                    await self.add_sample_until_full()
                    await asyncio.sleep(1)

    async def add_sample_until_full(self):
        """从data server获取待预测样本，直到填满推理队列"""
        if self.n_concurrent_fetch >= self.max_concurrent_fetch:
            return
        self.n_concurrent_fetch += 1
        n_added = 0
        while True:
            if self.should_stop_adding_sample(n_added):
                break
            try:
                async with self.session.get(f'{self.server_url}/data', params=self.worker_info) as r:
                    r = await r.text()
                    r = json.loads(r)
                    self.r_latest.update(r)
                    if r['status_code'] > 0:  # 1: all done, 2: terminated
                        break
                    # print('add:', r['idx'])
                    predict_coro = self.predict_sample(**self.prepare_inputs(r))
                    self.idx2task[r['idx']] = asyncio.create_task(
                        self.process_request(r, predict_coro),
                        name=str(time.time())
                    )
                    n_added += 1
            except (
                aiohttp.client_exceptions.ClientConnectionError,
                asyncio.exceptions.TimeoutError
            ):
                # print('get err')
                await asyncio.sleep(3)

        self.n_concurrent_fetch -= 1

    def should_stop_adding_sample(self, n_added: int):
        """负载均衡策略（推理队列较满时停止添加新样本），此基类不做限制，派生类可提供具体实现"""
        return False

    def prepare_inputs(self, r: dict):
        """将data_server返回的数据r转换为模型输入，派生类可覆盖此方法以定制输入"""
        return dict(inputs=r['inputs'])

    async def process_request(self, inputs, predict_coro):
        """将预测完成的样本推送给data server"""
        final_output = await predict_coro

        idx = inputs['idx']
        # print('done:', idx)
        otype = inputs.get('output_type', self.args.output_type)
        output = self.convert_final_output(inputs, final_output, otype=otype)
        data = {'idx': idx, 'output_type': otype, 'output': output}
        data.update(self.worker_info)

        while True:
            try:
                async with self.session.post(f'{self.server_url}/result', data=json.dumps(data)) as r:
                    r = await r.text()
                    r = json.loads(r)
                    break
            except OSError:  # 有时会出现 ConnectionResetError: [Errno 104] Connection reset by peer，超过系统连接数限制？尝试再次连接
                # traceback.print_exc()
                await asyncio.sleep(1.)
        await self.add_sample_until_full()  # 尝试添加新样本的推理队列
        task_done = self.idx2task.pop(idx)

    def convert_final_output(self, inputs: dict, final_output, otype: str):
        """
        将推理引擎的输出转换为返回data server的数据格式，此基类不做处理，派生类可覆盖此方法以适配输出

        Args:
            inputs (`dict`):
                通过调用data server获取的模型输入
            final_output:
                推理引擎的输出
            otype (`str`):
                输出结果的类型，支持的类型见`OUTPUT_TYPES`
        """
        return final_output

    @staticmethod
    def get_gpu_count() -> int:
        """获取当前进程可用的GPU数量"""
        device = os.environ.get('CUDA_VISIBLE_DEVICES')
        if device is None:
            ngpus = int(subprocess.run(  # all available gpus
                'nvidia-smi --query-gpu=name --format=csv,noheader | wc -l', shell=True, stdout=subprocess.PIPE
            ).stdout)
        else:
            assert device, 'No available gpus! (CUDA_VISIBLE_DEVICES=)'
            ngpus = len(device.split(','))
        return ngpus
