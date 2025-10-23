import os
import sys
import argparse
import random
import ujson as json
import asyncio
import time
import importlib
import inspect
import traceback
import uuid
import math
import datasets

from collections import deque
from dataclasses import dataclass, field
from functools import partial
from transformers import AutoProcessor, PreTrainedTokenizer
from sanic import Sanic, response
from sanic.request import Request
from typing import Callable, List, Dict, Deque, Optional, Union


OUTPUT_TYPES = ['text', 'reward', 'loss', 'prompt_tokens', 'next_token_prob', 'embedding', 'retrieval', 'cosyvoice2', 'turn_taking']

app = Sanic("DataServer")
app.config.RESPONSE_TIMEOUT = 60 * 60 * 24  # 24 hours

global_args: list[str] = None
data_args : list[str] = sys.argv[1:]
is_args_loaded = asyncio.Event()  # 确保参数已加载后再处理请求
node_info: dict = {}  # 各个节点的运行状态

args: argparse.Namespace = None
tokenizer: PreTrainedTokenizer = None
stop: str = None
preprocessors: Dict[str, Callable] = {}  # preprocess -> convert_sample_to_inputs
postprocessors: Dict[str, Callable] = {}

info = {  # 全局信息
    'i_pred': 0,   # 当前预测的data序号
    'n_queue': 0,  # 已提交到推理队列的样本数量
    'n_pred': 0,   # 已返回预测结果的样本数量
    'n_read': 0,   # 已提交到推理队列的文件数量
    'n_skip': 0,   # （因为超过最大长度）跳过的样本数量
    'n_done': 0,   # 完成写入的文件数量
    'n_token': 0,   # 已生成的token数量
    't_start': 0.,  # 开始预测的时间戳
    't_curr': 0.,   # 当前的时间戳
    'error': 0,
    'terminated': False,  # 下游发送终止信号
    'post_start': False  # 下游发送启动信号
}


@dataclass
class FileData:
    input_path: str  # './input/a.json'
    output_path: str  # './output/a.json'
    file_type: str = 'jsonl'
    preprocess: Optional[str] = None
    postprocess: Optional[str] = None
    output_type: Optional[str] = None
    output_key: Optional[str] = None
    samples: Optional[List[dict]] = None
    results: Optional[List[dict]] = None
    results_new: Optional[List[dict]] = None  # 仅用于no_order为True时，保存预测完成待写入的样本
    subsample_ids: Optional[List[int]] = None  # 仅用于args.subsample不为None时，采样一部分数据进行预测
    j_pred: int = 0   # 当前预测的样本序号
    n_pred: int = 0   # 已有预测结果的数量
    n_write: int = -1   # 已写入文件的样本数量
    read: Optional[asyncio.Task] = None
    write: tuple[asyncio.Semaphore, dict[int, asyncio.Task]] = field(
        default_factory=lambda : (asyncio.Semaphore(), {})
    )
    overwrite: bool = None
    is_done: bool = False  # 已完成预测并写入结果
    is_aborted: bool = False  # 下游请求终止文件推理
    preprocess_kwargs: Optional[dict] = None
    generation_params: Optional[dict] = None


@dataclass
class Sample:
    inputs: list[int] | dict
    output_type: Optional[str] = None
    output: Optional[Union[str, float, dict]] = None
    generation_params: Optional[dict] = None
    next_token_ids: Optional[List[int]] = None  # only used when output_type == 'next_token_prob'
    uid: str = field(default_factory=lambda : str(time.time()))
    done_event: asyncio.Event = field(default_factory=asyncio.Event)


file_list: List[FileData] = []   # 待预测的文件
file_path2idx: Dict[str, int] = {}

input_queue: Deque[Sample] = deque()  # 通过HTTP请求添加的待预测样本
uid2sample: Dict[str, Sample] = {}


def is_all_done():
    """所有推理都已完成（包括文件和HTTP请求的样本）"""
    return len(file_list) == get_num_done() and not uid2sample


def is_file_all_read(d: FileData):
    """判断数据（文件）是否已全部读取并提交推理"""
    return d.read is not None and d.read.done() and d.j_pred >= get_num_samples(d)


def is_file_all_done(d: FileData, written=False):
    """判断数据（文件）是否已全部完成推理"""
    if d.is_done:
        return True
    n_curr = d.n_write if written else d.n_pred
    return d.read is not None and d.read.done() and n_curr >= get_num_samples(d)


def get_num_samples(d: FileData, remaining=False):
    """返回单个文件需要推理的样本数量"""
    samples = d.samples if args.subsample is None else d.subsample_ids
    n_total = len(samples) if samples else 0
    if remaining:
        n_remain = max(0, n_total - d.j_pred)
        return n_remain
    else:
        return n_total


def get_num_done():
    """返回已完成推理的数据（文件）数量"""
    n = 0
    for d in file_list:
        if not d.is_done:
            d.is_done = is_file_all_done(d, written=True)
        if d.is_done:
            n += 1
    return n


@app.get("/info")
async def get_info(request: Request):
    """供主进程（`predict_multi_gpu.py`）调用，返回全局信息，包括已完成的样本/文件数量等"""
    info['n_file'] = len(file_list)
    info['n_done'] = get_num_done()
    info['n_queue'] = sum(d.j_pred for d in file_list)
    info['n_read'] = sum(1 for d in file_list if is_file_all_read(d))
    info['t_curr'] = time.time()
    info['node_info'] = node_info
    if global_args:
        info['global_args'] = global_args
    r = info
    if (rank := request.args.get('rank')) and should_terminate_rank(rank, check_running=True):
        r = r.copy()
        r['terminated'] = True
    return response.json(r)


@app.post("/node")
async def post_node(request: Request):
    """供主进程（`predict_multi_gpu.py`）调用，传入节点信息"""
    rank = request.json['rank']
    if global_args:
        if ' '.join(global_args) != ' '.join(request.json['args']):
            info['error'] += 1
            raise ValueError(f'global_args mismatch! server={global_args} client={request.json["args"]}')

    node_info.setdefault('ranks', {})[rank] = {'ip': request.client_ip}
    return response.json({
        'status_code': 0,
        'status_msg': 'ok'
    })


@app.get("/data")
async def get_data(request: Request):
    """供worker（predict_async.py）调用，返回待预测的inputs"""
    try:
        if (
            (rank := request.args.get('rank')) not in node_info.get('ranks', {})  # node_info中不包含当前rank，说明是旧的推理worker（等待重启）
            or should_terminate_rank(rank, check_running=False)  # 用户发送终止rank的信号
        ):
            return response.json({'status_code': 1})
        else:
            rank_info = node_info['ranks'][rank]
            rank_info.setdefault('n_queue', 0)  # 模型加载完成信号
            worker_id = request.args.get('worker_id')
            worker_info = rank_info.setdefault('workers', {}).setdefault(worker_id, {})

        if info['terminated']:
            return response.json({'status_code': 2, 'status_msg': 'terminated'})

        if not info['t_start']:
            info['t_start'] = time.time()

        if should_throttle_worker(rank, worker_id):
            return response.json({'status_code': 1})

        while input_queue:
            s = input_queue.popleft()
            if args.max_length and isinstance(s.inputs, list) and len(s.inputs) >= args.max_length:
                s.output = ''
                s.done_event.set()
                info['n_skip'] += 1
                continue
            r = {
                'status_code': 0,
                'inputs': s.inputs,
                'output_type': s.output_type,
                'generation_params': s.generation_params or {},
                'idx': s.uid
            }
            if s.next_token_ids is not None:
                r['next_token_ids'] = s.next_token_ids
            rank_info['n_queue'] = rank_info.get('n_queue', 0) + 1
            worker_info['n_queue'] = worker_info.get('n_queue', 0) + 1
            return response.json(r)

        while True:
            i, d = get_current_file()
            if d is None:
                return response.json({'status_code': 1})  # all done!

            if d.read is None:  # 开始读取数据（异步）
                d.read = asyncio.get_running_loop().run_in_executor(
                    None, load_file, i
                )

            idx = d.j_pred
            try:
                s = d.samples[idx if args.subsample is None else d.subsample_ids[idx]]
            except (TypeError, IndexError):
                await asyncio.sleep(0.1)  # 等待读取
                continue
            d.j_pred = idx + 1

            output_key = d.output_key or args.output_key
            if (  # 跳过已有推理结果的样本
                has_enough_answer_by_model(s)
                or (args.reuse and (s.get(output_key) and not s[output_key].startswith('ERROR')))
            ):
                set_result(i, idx, s)
                continue

            # print('get:', d.input_path, idx)
            kwargs = d.preprocess_kwargs or {}
            kwargs['data_path'] = d.input_path  # 用于加载多模态数据
            convert_sample_to_inputs, kwargs = get_preprocess_func(d.preprocess, output_type=d.output_type, kwargs=kwargs)
            inputs = convert_sample_to_inputs(s, args.prompt, tokenizer, **kwargs)
            if args.max_length and isinstance(inputs, list) and inputs and isinstance(inputs[0], int) and len(inputs) >= args.max_length:
                r = s.copy()
                r[output_key] = ''
                set_result(i, idx, r)
                info['n_skip'] += 1
                continue

            r = {
                'status_code': 0,
                'inputs': inputs,
                'output_type': d.output_type,
                'idx': '_'.join([str(i), str(idx)]),
                'generation_params': d.generation_params or {},
            }
            if 'max_new_tokens' in s:  # 支持针对单条样本设置生成长度
                r['generation_params']['max_new_tokens'] = s['max_new_tokens']
            rank_info['n_queue'] = rank_info.get('n_queue', 0) + 1
            worker_info['n_queue'] = worker_info.get('n_queue', 0) + 1
            return response.json(r)
    except:
        info['error'] += 1
        raise


@app.post("/result")
async def post_result(request: Request):
    """供worker（predict_async.py）调用，传入预测结果"""
    try:
        otype = request.json['output_type']
        output = request.json['output']
        if otype in ['text', 'beam_search']:
            output = convert_output_to_text(output)
        elif otype == 'next_token_prob':
            output = [[decode([token_id]), logprob] for token_id, logprob in output]
        elif otype == 'reward':
            if isinstance(output, dict):
                output['rewards'] = [round(s, 4) for s in output['rewards']]
                output['gating_output'] = [round(s, 4) for s in output['gating_output']]

        idx = request.json['idx']
        # print('post:', idx)
        if idx in uid2sample:
            s = uid2sample.pop(idx)
            s.output = output
            s.done_event.set()
        else:
            i_data, i_sample = map(int, idx.split('_'))
            d = file_list[i_data]
            i = i_sample if args.subsample is None else d.subsample_ids[i_sample]
            r = d.samples[i]
            if isinstance(d.samples, list):  # 释放内存（如果是Dataset则不允许修改）
                d.samples[i] = None
            output_key = d.output_key or args.output_key
            r[output_key] = output
            set_result(i_data, i_sample, r)
        info['n_pred'] += 1
        rank = request.json.get('rank')
        if (rank_info := node_info.get('ranks', {}).get(rank)) is not None:
            worker_info = rank_info.setdefault('workers', {}).setdefault(request.json.get('worker_id'), {})
            rank_info['n_pred'] = rank_info.get('n_pred', 0) + 1
            worker_info['n_pred'] = worker_info.get('n_pred', 0) + 1
        if info['terminated'] or should_terminate_rank(rank, check_running=True):
            return response.json({'status_code': 2, 'status_msg': 'terminated'})
        else:
            return response.json({'status_code': 0})
    except:
        info['error'] += 1
        raise


@app.post("/v1/chat/completions")
async def openai_v1_chat_completions(request: Request):
    """兼容OpenAI协议的API"""
    try:
        messages = request.json.get('messages')
        uid = f'chatcmpl-{str(uuid.uuid4().hex)}'
        otype = 'text'
        t0 = time.time()

        sample = {'messages': messages}
        if (tools := request.json.get('tools') or request.json.get('functions')):
            sample['tools'] = tools
        tool_choice = request.json.get('tool_choice') or request.json.get('function_call') \
            or ('auto' if tools else 'none')

        kwargs = {}
        if chat_template_kwargs := request.json.get('chat_template_kwargs'):
            kwargs['chat_template_kwargs'] = chat_template_kwargs
        if reasoning_effort := request.json.get('reasoning_effort'):
            kwargs.setdefault('chat_template_kwargs', {})['reasoning_effort'] = reasoning_effort
        await is_args_loaded.wait()
        convert_sample_to_inputs, kwargs = get_preprocess_func(output_type=otype, kwargs=kwargs)
        inputs = convert_sample_to_inputs(sample, args.prompt, tokenizer, **kwargs)

        generation_params = dict(
            n=request.json.get('n'),
            max_new_tokens=request.json.get('max_completion_tokens') or request.json.get('max_tokens'),
            temperature=request.json.get('temperature'),
            top_p=request.json.get('top_p'),
            stop=request.json.get('stop'),
            repetition_penalty=request.json.get('repetition_penalty'),
            presence_penalty=request.json.get('presence_penalty'),
        )
        generation_params = {k: v for k, v in generation_params.items() if v is not None}

        s = Sample(
            inputs=inputs,
            output_type=otype,
            generation_params=generation_params,
            uid=uid,
        )
        input_queue.append(s)
        uid2sample[s.uid] = s

        await s.done_event.wait()

        outputs = s.output if isinstance(s.output, list) else [s.output]
        choices = []
        for i, o in enumerate(outputs):
            c = {
                "index": i,
                "message": {
                    "role": "assistant",
                    "content": o,
                },
            }
            content = o if o is not None else ""
            if args.reasoning_parser:
                reasoning_content, content = args.reasoning_parser.extract_reasoning_content(content, request=None)
                c['message'].update(
                    reasoning_content=reasoning_content,
                    content=content,
                )
            if tool_choice != 'none' and args.tool_call_parser:
                tc = args.tool_call_parser.extract_tool_calls(content, request=None)
                if tc.tools_called:
                    c['message'].update(
                        content=tc.content,
                        tool_calls=[t.model_dump() for t in tc.tool_calls],
                    )
            choices.append(c)

        return response.json({
            "id": uid,
            "object": "chat.completion",
            "created": int(t0),
            "model": request.json.get('model'),
            "choices": choices,
        })

    except:
        return response.json({
            'object': 'error',
            'code': -1,
            'message': traceback.format_exc()
        })


@app.post("/model")
async def post_inputs(request: Request):
    """供下游任务调用，传入待预测的样本"""
    try:
        prompt : dict | str = request.json.get('prompt')
        otype = request.json.get('output_type')
        if otype is None:
            otype = args.output_type
        elif otype not in OUTPUT_TYPES:
            return response.json({
                'status_code': -1,
                'status_msg': f'"output_type" must be one of {OUTPUT_TYPES}!'
            })

        if prompt:
            if isinstance(prompt, str):
                prompt = {'messages': [{'role': 'user', 'content': prompt}]}
            kwargs = request.json.get('preprocess_kwargs') or {}
            if chat_template_kwargs := request.json.get('chat_template_kwargs'):
                kwargs['chat_template_kwargs'] = chat_template_kwargs
            await is_args_loaded.wait()
            convert_sample_to_inputs, kwargs = get_preprocess_func(output_type=otype, kwargs=kwargs)
            inputs = convert_sample_to_inputs(prompt, args.prompt, tokenizer, **kwargs)
        else:
            inputs = request.json.get('inputs')
        if not inputs:
            return response.json({
                'status_code': -1,
                'status_msg': f'"inputs" is missing!'
            })
        if isinstance(inputs[0], int):
            is_batched = False
            inputs = [inputs]
        else:
            is_batched = True

        samples = []
        for inputs_i in inputs:
            s = Sample(
                inputs=inputs_i,
                output_type=otype,
                generation_params=request.json.get('generation_params'),
                next_token_ids=request.json.get('next_token_ids') if otype == 'next_token_prob' else None
            )
            samples.append(s)
            input_queue.append(s)
            uid2sample[s.uid] = s

        await asyncio.gather(*[s.done_event.wait() for s in samples])
        output = [s.output for s in samples] if is_batched else samples[0].output
        return response.json({'status_code': 0, 'output': output, 'output_type': otype})

    except:
        return response.json({
            'status_code': -1,
            'status_msg': traceback.format_exc()
        })


@app.post("/file")
async def post_file(request: Request):
    """供下游任务调用，传入待预测的文件"""
    try:
        input_path = request.form.get('input_path')
        if not input_path:
            return response.json({
                'status_code': -1,
                'status_msg': f'"input_path" is missing!'
            })
        output_dir = request.form.get('output_dir')
        output_path = request.form.get('output_path')
        output_type = request.form.get('output_type')
        if output_type is None:
            output_type = args.output_type
        overwrite = request.form.get('overwrite')
        if isinstance(overwrite, str):
            overwrite = eval(overwrite)
        if not (output_dir or output_path):
            return response.json({
                'status_code': -1,
                'status_msg': f'"output_dir" or "output_path" is missing!'
            })
        if output_type not in OUTPUT_TYPES:
            return response.json({
                'status_code': -1,
                'status_msg': f'"output_type" must be one of {OUTPUT_TYPES}!'
            })
        if preprocess_kwargs := request.form.get('preprocess_kwargs'):
            preprocess_kwargs = json.loads(preprocess_kwargs)
        else:
            preprocess_kwargs = {}
        if chat_template_kwargs := request.form.get('chat_template_kwargs'):
            preprocess_kwargs['chat_template_kwargs'] = eval(f'dict({chat_template_kwargs})')

        await is_args_loaded.wait()
        added, existed = add_file_for_prediction(
            input_path,
            output_dir=output_dir,
            output_path=output_path,
            overwrite=overwrite,
            output_type=output_type,
            preprocess_kwargs=preprocess_kwargs,
            generation_params=request.form.get('generation_params'),
            preprocess=request.form.get('preprocess'),
            postprocess=request.form.get('postprocess'),
            output_key=request.form.get('output_key'),
        )

        return response.json({
            'status_code': 0,
            'n_added': len(added),
            'input_path': [d.input_path for d in (added + existed)],
            'output_path': [d.output_path for d in (added + existed)]
        })
    except:
        return response.json({
            'status_code': -1,
            'status_msg': traceback.format_exc()
        })


@app.get("/file")
async def get_file(request: Request):
    """供下游任务调用，返回文件信息（是否已完成、已完成的样本数等）"""
    try:
        input_path = request.args.get('input_path')
        d = get_file_info(
            input_path,
            request.args.get('output_path')
        )
        if d is None:
            return response.json({
                'status_code': -1,
                'status_msg': f'"{input_path}" is *NOT* in queue!'
            })
        if request.args.get('abort'):
            d.is_aborted = True
        return response.json({
            'status_code': 0,
            'is_done': is_file_all_done(d, written=True),
            'n_samples': len(d.samples or []) if args.subsample is None else len(d.subsample_ids or []),
            'n_pred': d.n_pred,
        })
    except:
        return response.json({
            'status_code': -1,
            'status_msg': traceback.format_exc()
        })


def get_file_info(input_path, output_path):
    """根据文件的输入输出路径，获取推理状态信息"""
    if output_path is None:
        for f in reversed(file_list):
            if f.input_path == input_path:
                return f
    else:
        i = file_path2idx.get((input_path, output_path))
        if i is not None:
            return file_list[i]


@app.post("/start")
async def post_start(request: Request):
    """供下游任务调用，启动推理进程（当manual_start为True时）"""
    info['post_start'] = True
    return response.json({
        'status_code': 0,
        'status_msg': 'ok'
    })


@app.post("/terminate")
async def post_terminate(request: Request):
    """供下游任务调用，退出推理，结束所有相关进程"""
    if rank := request.args.get('rank'):  # 终止个别rank
        node_info.setdefault('ranks', {}).setdefault(rank, {})['terminated'] = True
    else:
        info['terminated'] = True  # 全局退出
    return response.json({
        'status_code': 0,
        'status_msg': 'ok'
    })


@app.post("/restart")
async def post_restart(request: Request):
    """供下游任务调用，更新全局args，用于重启推理进程（更换模型/推理参数）"""
    if not is_all_done():
        return response.json({
            'status_code': -1,
            'status_msg': 'Restart not allowed, inference is still running!'
        })
    global global_args
    global_args = request.json['args']
    is_args_loaded.clear()
    # 重置数据队列和统计信息
    node_info.clear()
    file_list.clear()
    file_path2idx.clear()
    for k, v in info.items():
        if isinstance(v, int):
            info[k] = 0
        elif isinstance(v, bool):
            info[k] = False
    return response.json({
        'status_code': 0,
        'status_msg': 'ok'
    })


@app.post("/data_args")
async def post_data_args(request: Request):
    """供主进程（`predict_multi_gpu.py`）调用，更新并加载data_args"""
    try:
        data_args[:] = request.json['args']
        parse_args()
    except:
        info['error'] += 1
        raise
    return response.json({
        'status_code': 0,
        'status_msg': 'ok'
    })


def should_terminate_rank(rank, check_running):
    """用户发送信号，终止特定rank的推理进程"""
    rank_info = node_info.get('ranks', {}).get(rank, {})
    if rank_info.get('terminated'):
        if not check_running:
            return True
        else:  # check_running: 仅当所有worker都已完成推理时才退出
            if not any(d.get('n_queue', 0) - d.get('n_pred', 0) for d in rank_info.get('workers', {}).values()):
                return True


def should_throttle_worker(rank, worker_id):
    """如果剩余样本较少，使用更严格的负载均衡策略，强制每个worker平分剩余样本"""
    if any(d.read is None for d in file_list):  # 还有未读取的文件
        return
    n_remain = len(input_queue) + sum(get_num_samples(d, remaining=True) for d in file_list)
    if n_remain:
        worker2running = {  # 各个worker正在处理的数量
            (r, w): n_running
            for r, v in node_info.get('ranks', {}).items() for w, d in v.get('workers', {}).items()
            if (n_running := d.get('n_queue', 0) - d.get('n_pred', 0))
        }
        worker2running.setdefault((rank, worker_id), 0)
        n_sample_per_worker = math.ceil((n_remain + sum(worker2running.values())) / len(worker2running))  # 剩余样本平分给各个worker
        if n_sample_per_worker < 100 and worker2running[(rank, worker_id)] >= n_sample_per_worker:
            return True


def has_enough_answer_by_model(s: dict) -> bool:
    """检查choices字段下model_name对应的答案数量是否已达到限制"""
    if args.num_outputs_per_model and args.model_name:
        n = 0
        choices = s.get('choices')
        if isinstance(choices, list):
            for c in choices:
                if isinstance(c, dict):
                    if c.get('model_name') == args.model_name:
                        n += 1
        return n >= args.num_outputs_per_model


def get_current_file() -> tuple[int, FileData]:
    if args.file_schedule == 'fifo':
        return get_current_file_fifo()
    elif args.file_schedule == 'rr':
        return get_current_file_rr()
    else:
        raise NotImplementedError


def get_current_file_fifo() -> tuple[int, FileData]:
    """返回当前正在推理的一个数据文件（先进先出调度策略）"""
    i = info['i_pred']
    try:
        d = file_list[i]
    except IndexError:
        d = None
    while d is None or is_file_all_read(d) or d.is_aborted:  # 当前数据（文件）已全部提交推理
        if i < len(file_list) - 1:  # 还有其他数据文件
            i += 1
            info['i_pred'] = i
            d = file_list[i]
        else:
            return i, None
    return i, d


def get_current_file_rr() -> tuple[int, FileData]:
    """返回当前正在推理的一个数据文件（轮流调度策略）"""
    i = info['i_pred']
    f = None
    for j in range(len(file_list)):
        k = (i + j + 1) % len(file_list)
        d = file_list[k]
        if is_file_all_read(d) or d.is_aborted:  # 当前数据（文件）已全部提交推理
            continue
        else:
            i = info['i_pred'] = k
            f = d
            break
    return i, f


def get_preprocess_func(preprocess_file: str = None, output_type: str = None, kwargs: dict = None):
    if preprocess_file is None:
        preprocess_file = args.preprocess
    if preprocess_file not in preprocessors:
        if preprocess_file == 'none':
            f = lambda _s, *args, **kwargs: _s
        else:
            module = importlib.import_module(preprocess_file)
            f = getattr(module, 'convert_sample_to_input_ids')
        params = inspect.signature(f).parameters
        preprocessors[preprocess_file] = (f, params)
    f, params = preprocessors[preprocess_file]
    kwargs = kwargs or {}
    if args.chat_template_kwargs and 'chat_template_kwargs' not in kwargs:
        kwargs['chat_template_kwargs'] = args.chat_template_kwargs
    if output_type == 'reward':
        kwargs.update(
            remove_last_assistant=False,
            add_generation_prompt=False,
        )
    kwargs = {k: v for k, v in kwargs.items() if k in params}
    return f, kwargs


def get_postprocess_func(postprocess_file: str):
    key = postprocess_file
    if key not in postprocessors:
        f = getattr(importlib.import_module(postprocess_file), 'postprocess')
        if 'model_name' in inspect.signature(f).parameters:
            f = partial(f, model_name=args.model_name)
        postprocessors[key] = f
    return postprocessors[key]


def convert_output_to_text(output):
    """将推理输出的token ids转换为文本"""
    if not output:  # 推理无结果（输入超过模型最大长度？）
        return
    if isinstance(output, dict):  # /generate返回格式，包含text
        info['n_token'] += output.get('meta_info', {}).get('completion_tokens', 0)
        return output.get('text')
    if isinstance(output, str):  # output text
        return output
    if isinstance(output[0], int):  # output: list[int]
        info['n_token'] += len(output) + 1
        return remove_stop_str(decode(output), stop).strip()
    else:  # output: list[list[int]] | list[dict]
        return [convert_output_to_text(o) for o in output]


def decode(token_ids: list[int]) -> str:
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=not args.output_special_tokens
    )


def remove_stop_str(t, stop):
    """去除文本中包含的停止符"""
    if stop:
        if isinstance(stop, str):
            stop = [stop]
        while True:
            _t = t
            for s in stop:
                if _t.endswith(s):
                    _t = _t[:-len(s)]
            if _t == t:
                break
            else:
                t = _t
    return t


def set_result(i_data, i_sample, r):
    d = file_list[i_data]
    if d.postprocess is not None:
        r = get_postprocess_func(d.postprocess)(r)
    if d.results is None:
        d.results = []
        d.results_new = deque()
    rs = d.results
    while len(rs) < i_sample + 1:
        rs.append(None)
    rs[i_sample] = r
    if args.no_order:
        d.results_new.append(r)
    d.n_pred += 1
    d.write[1][i_sample] = asyncio.create_task(write_result_async(i_data, i_sample))


async def write_result_async(i_data, i_sample):
    """调用线程池异步写入结果"""
    d = file_list[i_data]
    sem, i2task = d.write
    if (d.input_path == d.output_path or args.reuse) and not d.read.done():
        await d.read
    async with sem:
        await asyncio.get_running_loop().run_in_executor(
            None, write_result, i_data
        )
    del i2task[i_sample]


def write_result(i_data):
    """将推理结果写入jsonl文件"""
    try:
        d = file_list[i_data]
        # print('save:', d.output_path)
        json_kwargs = dict(
            ensure_ascii=False,
            escape_forward_slashes=False,  # ujson需要设置，以保持与标准库（json）一致的行为
        )
        os.makedirs(os.path.split(d.output_path)[0], exist_ok=True)
        if d.n_write == -1:
            d.n_write = 0
            if d.overwrite or args.reuse:  # 覆盖旧文件
                f = open(d.output_path, 'w')
                f.close()
        with open(d.output_path, 'a') as f:
            if args.no_order:
                while d.results_new:
                    r = d.results_new.popleft()
                    f.write(json.dumps(r, **json_kwargs) + '\n')
                    d.n_write += 1
            else:
                while d.n_write < len(d.results):
                    i = d.n_write
                    r = d.results[i]
                    if r is None:
                        break
                    f.write(json.dumps(r, **json_kwargs) + '\n')
                    d.results[i] = None  # 释放内存
                    d.n_write += 1

    except:
        traceback.print_exc()
        info['error'] += 1
        raise


def load_file(i_data):
    """加载输入数据文件"""
    try:
        d = file_list[i_data]
        if d.file_type.startswith('json'):
            load_json_file(i_data)
        else:
            load_dataset(i_data)
        if args.subsample is not None:
            if args.subsample >= 1.:
                n_sample = int(args.subsample)
            else:
                n_sample = max(int(len(d.samples) * args.subsample), 1)
            if args.seed is not None:
                random.seed(args.seed)
            if n_sample < len(d.samples):
                d.subsample_ids = random.sample(range(len(d.samples)), n_sample)
            else:
                d.subsample_ids = list(range(len(d.samples)))
    except:
        traceback.print_exc()
        info['error'] += 1
        raise


def load_json_file(i_data):
    """加载json/jsonl格式的数据文件"""
    try:
        d = file_list[i_data]
        # print('read:', d.input_path)

        with open(d.input_path) as f:
            try:
                l = next(f)  # 读取一行，用来判断文件是json还是jsonl格式
            except StopIteration:  # 空文件
                d.samples = []
                d.is_done = True
                return
            else:
                f.seek(0)
                try:
                    _ = json.loads(l)
                except ValueError:
                    assert not args.reuse, 'reuse为True时，数据必须是jsonl格式！'
                    d.file_type = 'json'
                    d.samples = json.load(f)  # 整个文件是一个json对象
                else:
                    d.file_type = 'jsonl'  # 每一行是一个json对象
                    d.samples = []
                    if args.reuse:
                        if os.path.isfile(d.output_path):  # 先读取已有的输出文件
                            with open(d.output_path) as f1:
                                for l in f1:
                                    d.samples.append(json.loads(l))
                        for i, l in enumerate(f):  # 已输出行数可能小于输入行数，需要从输入文件读取剩余行数
                            if i >= len(d.samples):
                                d.samples.append(json.loads(l))
                    else:
                        # 已有jsonl格式的结果，跳过已预测的样本
                        n_samples, n_pred = 0, 0
                        if not d.overwrite and os.path.isfile(d.output_path):
                            n_samples = int(os.popen('wc -l ' + d.input_path).read().split()[0])
                            n_pred = int(os.popen('wc -l ' + d.output_path).read().split()[0])
                            d.j_pred = n_pred
                            d.n_pred = n_pred
                            d.n_write = n_pred
                        if n_samples == n_pred > 0:  # 已全部推理完成，无需加载样本
                            d.samples.extend(None for _ in range(n_samples))
                        else:
                            for i, l in enumerate(f):
                                d.samples.append(None if i < n_pred else json.loads(l))
                        if d.n_write == len(d.samples):  # 已有全部样本的预测结果
                            d.is_done = True

    except:
        traceback.print_exc()
        info['error'] += 1
        raise


def load_dataset(i_data):
    """加载datasets格式的数据文件"""
    try:
        d = file_list[i_data]
        if d.file_type == 'parquet':
            d.samples = datasets.load_dataset('parquet', data_files=d.input_path)['train']
        else:
            d.samples = datasets.load_from_disk(d.input_path)
        # 已有jsonl格式的结果，跳过已预测的样本
        if not d.overwrite and os.path.isfile(d.output_path):
            n_pred = int(os.popen('wc -l ' + d.output_path).read().split()[0])
            d.j_pred = n_pred
            d.n_pred = n_pred
            d.n_write = n_pred
        if d.n_write == len(d.samples):  # 已有全部样本的预测结果
            d.is_done = True
    except:
        traceback.print_exc()
        info['error'] += 1
        raise


def load_tokenizer(tokenizer_path, output_type='not_cosyvoice2', chat_template=None):
    """加载processor/tokenizer"""
    global tokenizer
    if output_type != 'cosyvoice2':
        tokenizer = AutoProcessor.from_pretrained(tokenizer_path, trust_remote_code=True)
    elif output_type in ['cosyvoice2']:
        from utils.cosyvoice import load_cosyvoice2_tokenizer
        tokenizer = load_cosyvoice2_tokenizer(tokenizer_path)
    if chat_template:
        if chat_template.endswith('.jinja'):
            with open(chat_template, encoding='utf-8') as f:
                chat_template = f.read()
        tokenizer.chat_template = chat_template


def add_file_for_prediction(
    data_path, output_dir=None, output_path=None, overwrite=None, generation_params=None, **kwargs
) -> List[FileData]:
    """
    添加文件到推理队列，支持下列输入格式：
        - json/jsonl（文件）
        - parquet（文件）
        - datasets（目录）

    Args:
        data_path (`str`):
            输入的文件路径，支持多个输入文件，以英文逗号分隔，或者输入目录，自动读取目录下所有支持的文件
        output_dir (`str | None`):
            输出的目录，会自动创建与源文件同名的结果文件，仅当未指定`output_path`时生效
        output_path (`str | None`):
            输出的文件路径，如果指定了`output_path`，则`output_dir`不生效
        overwrite (`bool | None`):
            如果输出文件已存在，是否覆盖
        generation_params (`str | dict | None`):
            模型生成答案的参数（如：采样温度等）
    """
    fs: list[FileData] = []
    if output_path:  # 如果指定了输出文件路径，输入路径必须是单个文件或dataset目录
        add_single_data(fs, data_path, output_path)
    else:  # 自动扫描数据目录并判断数据格式
        add_all_data(fs, data_path, output_dir)
    assert len(set(f.output_path for f in fs)) == len(fs), f'文件冲突，多个输入有相同的输出路径（尝试重命名或减少输入文件）：{get_file_data_info(fs)}'

    if overwrite is None:
        overwrite = args.overwrite
    if isinstance(generation_params, str):
        generation_params = json.loads(generation_params)

    added = []  # 新提交的文件
    existed = []  # 之前已提交过，避免重复推理
    for f in fs:
        f.overwrite = overwrite
        f.output_type = kwargs.get('output_type', args.output_type)
        for k, v in kwargs.items():
            assert hasattr(f, k)
            setattr(f, k, v)
            if k == 'preprocess':  # 提前加载相关函数，如有报错及时返回调用方
                get_preprocess_func(v, output_type=f.output_type)
            elif k == 'postprocess' and v:
                get_postprocess_func(v)

        k = (f.input_path, f.output_path)  # 根据“输入-输出”去重，因为有时即使输入路径相同，也会使用不同的preprocess，输出到不同路径
        if (
            k in file_path2idx
            and not is_file_all_done(d := file_list[file_path2idx[k]], written=True)
            and not d.is_aborted
        ):
            existed.append(f)
        else:
            file_path2idx[k] = len(file_list)
            f.generation_params = generation_params
            file_list.append(f)
            added.append(f)

    if added:
        print(f'已添加下列数据至推理队列：{get_file_data_info(added)}')
    if existed:
        print(f'下列数据已在推理队列中：{get_file_data_info(existed)}')
    return added, existed


def add_single_data(fs: list[FileData], data_path: str, output_path: str, raise_error: bool = True):
    file_type = None
    if os.path.isfile(data_path):
        if data_path.endswith('.jsonl') or data_path.endswith('.json'):
            file_type = 'json'
        elif data_path.endswith('.parquet'):
            file_type = 'parquet'
        output_path = output_path.rsplit('.', 1)[0] + '.jsonl'
    elif os.path.isdir(data_path) and os.path.isfile(os.path.join(data_path, 'dataset_info.json')):
        file_type = 'dataset'

    if file_type is not None:
        fs.append(FileData(
            input_path=data_path,
            output_path=output_path,
            file_type=file_type,
        ))
    elif raise_error:
        raise ValueError(f'输入文件不存在或不支持：{data_path}')


def add_all_data(fs: list[FileData], data_path: str, output_dir: str):
    if not os.path.isdir(data_path):  # 支持多个输入文件，以英文逗号分隔
        for f in data_path.split(','):
            output_path = os.path.join(output_dir, os.path.basename(f))
            add_single_data(fs, f, output_path)
    else:  # 尝试加载目录下的所有数据（包括子目录）
        for p, q, v in os.walk(data_path):
            if 'dataset_info.json' in v:  # datasets目录
                sub_path = os.path.basename(p) if p == data_path else p[len(data_path):].lstrip(os.path.sep)  # 保留子目录结构
                output_path = os.path.join(output_dir, sub_path) + '.jsonl'
                add_single_data(fs, p, output_path)
            for f in v:  # 所有目录中的文件
                f = os.path.join(p, f)
                output_path = os.path.join(output_dir, f[len(data_path):].lstrip(os.path.sep))
                add_single_data(fs, f, output_path, raise_error=False)  # 自动跳过不支持的文件类型


def get_file_data_info(fs: list[FileData]) -> str:
    return json.dumps(
        [{'input_path': f.input_path, 'output_path': f.output_path, 'file_type': f.file_type} for f in fs],
        ensure_ascii=False, escape_forward_slashes=False, indent=2
    )


def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, help="Data file to predict")
    parser.add_argument("--preprocess", type=str, default='preprocess', help="Module that provides preprocessing function")
    parser.add_argument("--postprocess", type=str, default=None, help="Module that provides postprocessing function")
    parser.add_argument("--prompt", type=str, default='Hithink', help="Prompt type")
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory")
    parser.add_argument("--output_path", type=str, default=None, help="Output file path")
    parser.add_argument("--output_key", type=str, default='output', help="Output key")
    parser.add_argument("--output_type", type=str, default='text', choices=OUTPUT_TYPES, help="Output type")
    parser.add_argument("--model_name", type=str, default=None, help="Used by some postprocess methods, e.g. postprocess_append_choices.py")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer path")
    parser.add_argument("--port", type=int, default=7888, help="Server port")
    parser.add_argument("--max_length", type=int, default=None, help="Max number of tokens (input and output)")
    parser.add_argument("--max_input_tokens", type=int, default=None,
                        help="Max number of input tokens, longer inputs will be truncated")
    parser.add_argument("--stop", type=str, default='', help="token/word at which generation will be stopped")
    parser.add_argument("--reuse", action='store_true',
                        help="If prediction file exists, reuse previous outputs and only predict samples with empty output")
    parser.add_argument("--overwrite", action='store_true',
                        help="Overwrite if output file exists. If not set, will skip finished samples")
    parser.add_argument("--no_order", action='store_true', help="Output will be written in different order than input")
    parser.add_argument("--subsample", type=float, help="proportion (0.0 - 1.0) or number (>= 1) of samples used")
    parser.add_argument("--seed", type=int, help="seed used for subsampling")
    parser.add_argument("--file_schedule", type=str, default='fifo', choices=['fifo', 'rr'], help="Schedule strategy for files")
    parser.add_argument("--num_outputs_per_model", type=int, help="limit the number of outputs by model_name (by counting the existing answers in 'choices')")
    parser.add_argument("--chat_template", type=str, help="chat template (text or .jinja file)")
    parser.add_argument("--chat_template_kwargs", type=str, help="kwargs for apply_chat_template, e.g. enable_thinking=False")
    parser.add_argument("--output_special_tokens", action='store_true', help="If set, use skip_special_tokens=False for tokernize.decode")
    parser.add_argument("--tool_call_parser", type=str, help="Used to parse the model-generated tool call into OpenAI API")
    parser.add_argument("--reasoning_parser", type=str, help="Used to parse the model-generated reasoning content into OpenAI API")

    global args
    args = parser.parse_args(data_args)

    if args.data:
        add_file_for_prediction(args.data, args.output_dir, args.output_path, postprocess=args.postprocess)
    if args.tokenizer:
        load_tokenizer(args.tokenizer, args.output_type, args.chat_template)
    if args.max_length:
        tokenizer.model_max_length = args.max_length  # preprocess 代码可能会从 tokenizer 读取最大长度
    if args.stop:
        global stop
        stop = args.stop.replace('\\n', '\n').split(',')
    if args.chat_template_kwargs:
        args.chat_template_kwargs = eval(f'dict({args.chat_template_kwargs})')
    if args.tool_call_parser:
        from vllm.entrypoints.openai.tool_parsers import ToolParserManager
        args.tool_call_parser = ToolParserManager.get_tool_parser(args.tool_call_parser)(tokenizer)
    if args.reasoning_parser:
        from vllm.reasoning import ReasoningParserManager
        args.reasoning_parser = ReasoningParserManager.get_reasoning_parser(args.reasoning_parser)(tokenizer)

    is_args_loaded.set()


if __name__ == "__main__":
    try:
        parse_args()
    except:
        traceback.print_exc()
        info['error'] += 1
    app.run(host='0.0.0.0', port=args.port, single_process=True, access_log=False)
