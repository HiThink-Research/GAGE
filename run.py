import yaml
import subprocess
import traceback
import select
import os
import argparse
import importlib
import asyncio
import aiohttp
import aiohttp
import time
import json
from tqdm import tqdm
from loguru import logger
from statistic import statistic


cur_path = os.path.dirname(os.path.abspath(__file__))

remote_model_port = os.getenv('REMOTE_MODEL_PORT')
judge_model_port = os.getenv('JUDGE_MODEL_PORT')
session: aiohttp.ClientSession = None
session: aiohttp.ClientSession = None
task_status: dict[str, dict] = {}
model_status: dict[str, dict] = {}
judge_status: dict[str, dict] = {}


async def start_inference_engine(model, port=None, **kwargs):
    """启动模型推理服务"""
    if remote_model_port is not None and port is None:  # 如果端口已被占用，端口号增加1，直至找到未被占用的端口
        port = int(remote_model_port) + 1
        while subprocess.run(f'netstat -tuln | grep -q ":{port} "', shell=True).returncode == 0:
            port += 1
    port = str(port)
    cmd = [
        "python", os.path.join(cur_path, "inference", "predict_multi_gpu.py"),
        "--model", model,
        "--server_port", port,
    ]
    kwargs.setdefault('prompt', 'chat_template')
    for k, v in kwargs.items():
        if v:
            cmd.extend([f'--{k}', str(v)])
    cmd.extend([
        "--run_forever",
        "&"
    ])
    cmd = ' '.join(cmd)
    logger.info(f'启动推理引擎：{cmd}')
    subprocess.run(cmd, shell=True)
    model_status.setdefault(port, {})['terminated'] = False
    return port


def check_inference_engine_running(port=None):
    if port is None:
        port = remote_model_port
    ps_res = os.popen(f'ps auxww | grep predict_multi_gpu.py | grep "\-\-server_port {port}" | grep -v grep').read()
    return bool(ps_res)


async def run_command(cmd):
    process = subprocess.Popen(
        cmd,
        shell=True,
    )
    while True:
        # 检查进程是否结束
        if process.poll() is not None:
            break
        await asyncio.sleep(1.)

    #获取命令的返回码
    rc = process.poll()

async def run_server(cmd):
    process = subprocess.Popen(
        cmd,
        shell=True,
    )
    #获取命令的返回码
    rc = process.poll()
    return rc


async def send_request(
    data, method, port, path='file', json=False, retry_interval=10, max_retries=10, ignore_errors=False
):
    """用于请求推理引擎，包括：提交文件、请求推理进度、发送终止信号"""
    assert method in ['get', 'post']
    retry_times = 0
    is_remote = isinstance(port, str) and port.startswith('http://')
    while True:
        if not is_remote and not check_inference_engine_running(port):
            raise RuntimeError(f'推理引擎已退出！({port=})')
        try:
            if method == 'post':
                f = session.post
                kwargs = {'json' if json else 'data': data}
            else:
                f = session.get
                kwargs = {'params': data}
            url = port if is_remote else f'http://localhost:{port}'
            url = f'{url}/{path}'
            async with f(url, **kwargs) as r:
                r = await r.json()
            assert r['status_code'] == 0, str(r)
            return r
        except OSError:  # 有时会出现 ConnectionResetError: [Errno 104] Connection reset by peer，超过系统连接数限制？尝试再次连接
            if not ignore_errors:
                traceback.print_exc()
            if retry_times > max_retries:
                if ignore_errors:
                    break
                else:
                    raise RuntimeError(f'请求失败：重试超过{max_retries}次！')
            await asyncio.sleep(retry_interval)
            retry_times += 1
            continue


async def run_judge_inference(config, task):
    """
    运行裁判员模型的推理流程，用于评估特定任务的结果。
    
    该函数负责准备裁判员模型的输入文件，提交推理请求，并等待推理完成。
    根据配置可以使用本地大模型或外部API作为裁判。
    
    参数:
        config (dict): 包含任务配置信息的字典，必须包含以下键:
            - save_dir: 保存输出结果的目录
            - tasks: 包含特定任务配置的字典
        task (str): 要评估的任务名称，此名称用于定位配置中的相关设置
            
    返回:
        list: 包含所有输出文件路径的列表
        
    流程:
        1. 准备裁判员模型的输入文件:
           - 如果没有预处理步骤，直接使用任务输出文件
           - 如果有预处理步骤，调用指定的预处理函数生成输入文件
        2. 提交裁判员推理:
           - 如果配置指定了外部API方法，调用相应的外部接口
           - 否则使用本地大模型作为裁判
        3. 等待推理完成并更新任务状态
        
    注意:
        - 预处理函数需要在配置中正确指定，格式为"module.submodule.function"
        - 外部API方法需要在ExternalApi枚举中定义
        - 函数执行是异步的，调用时需要使用await
    """
    output_path = []
    judge_data = config['tasks'][task]['judge']
    # 准备裁判员模型的输入文件
    if 'preprocess' not in judge_data:
        input_file = os.path.join(config['save_dir'], task + '.jsonl')
    else:
        preprocess_module, preprocess_func = judge_data['preprocess'].rsplit('.', 1)  # e.g. "utils.judge.data_preprocess"
        preprocess_module = importlib.import_module(preprocess_module)
        preprocess_func = getattr(preprocess_module, preprocess_func)
        judge_input_dir = 'judge/input'
        judge_prompt_path = config.get('tasks', {}).get(task, {}).get('judge_prompt')

        if judge_prompt_path:
            preprocess_func(input_path=config['save_dir'], input_file=task + '.jsonl', save_dir=judge_input_dir, prompt_path=judge_prompt_path)
        else:
            preprocess_func(input_path=config['save_dir'], input_file=task + '.jsonl', save_dir=judge_input_dir)
        #preprocess_func(input_path=config['save_dir'], input_file=task + '.jsonl', save_dir=judge_input_dir)
        input_file = os.path.join(config['save_dir'], judge_input_dir, task + '.jsonl')
    # 提交裁判员推理
    d = {
        'input_path': input_file,
        'output_path': os.path.join(config['save_dir'], 'judge/output', task + '.jsonl')
    }
    
    if 'method' in judge_data:
        for k, v in judge_data.get('inference', {}).items():
            if k == 'output_dir':
                d['output_path'] = os.path.join(config['save_dir'], v, task + '.jsonl')
            else:
                d[k] = v
        from envs.constants import ExternalApi
        external_api = ExternalApi[config['tasks'][task]['judge']['method']]
        os.makedirs(os.path.join(config['save_dir'], 'judge/output'),exist_ok=True)
        with open(input_file,'r',encoding='utf-8') as f:
            data = [json.loads(l) for l in f]
        out = []
        if output_path:
            output_path.extend(d['output_path'])
        else:
            output_path.append(d['output_path'])
        input_list = [sample['instruction'] if 'instruction' in sample else sample for sample in data]
        logger.info("开始调用外部接口：{} 评估".format(config['tasks'][task]['judge']['method']))
        await external_api.texts2texts(input_list,output_file=d['output_path'])
    else:
        logger.info("开始使用本地大模型作为裁判进行评估")
        jd = None  # judge info
        if judge_model_port:
            port = judge_model_port
        elif judge_data['judge_model_path'].startswith('http://'):  # 用predict_multi_gpu.py启动的远程服务
            port = judge_data['judge_model_path']
        else:  # 没有提前启动裁判员模型，根据子任务配置启动
            jd = judge_status.setdefault(judge_data['judge_model_path'], {})
            if tp := judge_data.get('judge_tensor_parallel'):
                jd['judge_tensor_parallel'] = max(tp, jd.get('judge_tensor_parallel', 1))
            jd['n_task'] = jd.get('n_task', 0) + 1
            while (port := jd.get('port')) is None:
                await asyncio.sleep(5)
            await asyncio.sleep(5)

        generation_params = {
            k: v for k in [
                'max_new_tokens', 'temperature', 'top_p'
            ] if (v := config.get(f'judge_{k}'))
        }
        if generation_params:
            d['generation_params'] = json.dumps(generation_params)
        r = await send_request(d, 'post', port)
        if r and r.get('output_path'):
            output_path.extend(r['output_path'])

            # 等待裁判员推理完成
            await wait_for_inference(task + ' judge', port, input_file, output_path[-1])

        if jd is not None:
            jd['n_done'] = jd.get('n_done', 0) + 1

    task_status[task]['judge'] = True  # 裁判员模型推理完成
    return output_path


async def run_task_inference(config, task):
    """
    为指定任务运行推理，使用外部API或本地模型。
    
    此函数通过以下步骤处理推理过程：
    1. 从配置中确定输入数据路径
    2. 检查是否应使用外部API
    3. 根据配置执行适当的推理方法
    
    参数:
        config (dict): 包含任务设置和路径的配置字典
        task (str): 要运行推理的任务名称
    
    返回:
        None: 此函数执行操作但不返回值
    
    可能引发:
        外部API调用或本地模型推理可能产生的异常
    """
    input_path = config['tasks'][task]['data_path']
    output_path = os.path.join(config['save_dir'], task + '.jsonl')
    #获取envs检测是否有external api
    external_api = os.environ.get('EXTERNAL_API')
    if external_api:
        logger.info("Using external api: {} to predict".format(external_api))
        from envs.constants import ExternalApi
        external_api = ExternalApi[external_api]
        await wait_for_external_api(config,task, external_api,input_path)
    else:
        await send_request({
            'input_path': input_path,
            'output_path': output_path,
            'output_type': config['tasks'][task]['type'],
            'generation_params': json.dumps({
                k: v for k, v in config['tasks'][task].items() if k in ['max_new_tokens']
            })
        }, 'post', remote_model_port)
        await wait_for_inference(task, remote_model_port, input_path)
    return output_path


async def wait_for_external_api(config,task, external_api,input_path):
    """
    调用外部API处理输入数据并保存结果。
    
    根据输入数据类型（图像或文本）自动选择相应的API调用方法，并将结果保存到指定目录下。
    
    Args:
        config (dict): 配置信息，必须包含'save_dir'键，指定结果保存目录
        task (str): 任务名称，用于生成输出文件名
        external_api (object): 外部API接口对象，需要实现images2texts和texts2texts方法
        input_path (str): 输入数据文件路径，应为jsonl格式
        
    Returns:
        list: API调用的结果列表
        
    Notes:
        - 输入文件必须是jsonl格式，每行为一个可解析为JSON的字符串
        - 当消息中包含"image_url"时会调用图像处理API，否则调用文本处理API
        - 输出文件将保存为{save_dir}/{task}.jsonl
    """
    data = [json.loads(i) for i in open(input_path, encoding='utf8')]
    subsample = config.get("subsample",100000000)
    data = data[:subsample]
    output_file = os.path.join(config['save_dir'], task + '.jsonl')
    os.makedirs(config['save_dir'], exist_ok=True)

    # NOTE 目前只做了图像和文本的调用支持
    if "image_url" in str(data[0]['messages']):
        results = await external_api.images2texts(data,output_file=output_file)
    else:
        results = await external_api.texts2texts(data,output_file=output_file)
        

async def wait_for_inference(task, port, input_path, output_path=None):
    params = {'input_path': input_path}
    if output_path is not None:
        params['output_path'] = output_path
    r_old = None
    while True:
        await asyncio.sleep(5)
        r = await send_request(params, 'get', port)

        if not r:
            continue

        if r != r_old and r.get('n_pred', 0) > 0:  # 只有状态发生变化时才打印
            logger.info("current task:{} status {}".format(task, r))  # {"status_code":0,"is_done":false,"n_samples":244128,"n_pred":74}
            r_old = r

        if r.get('is_done'):
            break


async def correct_answer_format(config, task):
    """
    检测“\\boxed”输出格式以及选择题，辅助模型以正确格式输出答案。

    参数:
        config (dict): 包含任务设置和路径的配置字典
        task (str): 要运行推理的任务名称

    返回:
        None: 此函数执行操作但不返回值
    """
    # 默认不开启，需要在config.yaml中添加“answer_format_assist: true”以启用
    if not config.get('answer_format_assist') or 'judge' in config['tasks'][task]:
        return

    # 仅针对compare_func中包含\boxed的任务做格式检查
    func_path = config['tasks'][task]['compare_func']['path']
    with open(func_path) as f:
        if '\\boxed' not in f.read():
            return

    # 读取模型生成结果
    output_path = os.path.join(config['save_dir'], task + '.jsonl')
    with open(output_path) as f:
        data = [json.loads(l) for l in f]

    import re
    batch = []
    batch_size = 128
    n_modified = 0
    for i, d in enumerate(data):
        try:
            assert d['messages']
            pred = d['predict_result']
            target = d['choices'][0]['message']['content'][0]['text']
            is_mc = re.match(r'[A-Z]$', target)  # 根据label判断是否为选择题
        except KeyError:
            continue

        # 预处理样本
        if pred and not (re.search(r'\\boxed{[A-Z]', pred) if is_mc else re.search(r'\\boxed{', pred)):
            # 生成答案缺少\boxed格式，拼接答案模板并再次生成
            n_modified += 1
            if len(pred.encode('utf-8')) != len(pred):  # 包含非ascii字符，默认为中文
                pred += '\n最终答案是\\boxed{'
            else:
                pred += '\nThe final answer is \\boxed{'

            s = d.copy()
            del s['predict_result']
            # 模型请求参数
            params = dict(prompt=s, preprocess_kwargs=dict(response_prefix=pred))
            if is_mc:  # 如果是选择题，则取下一token的top-n概率
                params['output_type'] = 'next_token_prob'
                params['generation_params'] = dict(top_logprobs_num=10)
            else:  # 不是选择题，则使用拼接后的答案模板生成答案
                params['output_type'] = 'text'
                params['generation_params'] = dict(max_new_tokens=20)  # 最终答案通常较短
            t = asyncio.create_task(send_request(params, 'post', remote_model_port, path='model', json=True))
            batch.append((i, is_mc, pred, t))

        # 提交batch推理，处理结果
        if len(batch) == batch_size or i == len(data) - 1:
            rs = await asyncio.gather(*[t for *_, t in batch])  # 并发请求，等待结果
            for r, (j, is_mc, pred, _) in zip(rs, batch):  # 处理结果
                if not r.get('output'):
                    continue
                if is_mc:  # output: [['A', -0.004694137256592512],['C', -6.528131484985352],...]
                    r = sorted(r['output'], key=lambda _i: _i[1], reverse=True)
                    print([[_i[0], round(_i[1], 2)] for _i in r])
                    choices = [s for t, _ in r if re.match(r'[A-Z]', s := t.strip())]
                    data[j]['predict_result'] = pred + (choices[0] if choices else r[0][0]) + '}'
                else:
                    print(r['output'])
                    data[j]['predict_result'] = pred + r['output']
            batch.clear()
            logger.info(f'{task=} total={len(data)} checked={i + 1} modified={n_modified}')

    if n_modified:
        with open(output_path, 'w') as f:
            for d in data:
                f.write(json.dumps(d, ensure_ascii=False) + '\n')


async def restart_inference_engine(port=remote_model_port, engine_args=None, **kwargs):
    """重启模型推理服务以更换模型"""
    args = [
        "--server_port", port,
        "--run_forever",
    ]
    kwargs.setdefault('prompt', 'chat_template')
    for k, v in kwargs.items():
        if v:
            args.extend([f'--{k}', str(v)])
    if engine_args is not None:
        args.extend(engine_args.split())
    await send_request({'args': args}, 'post', port, path='restart', json=True)
    return port


async def maybe_start_judge_model(config):
    """
    等待全部测试集推理完成后，根据需要启动裁判员模型。
    
    此协程监控推理过程并管理到裁判员模型的转换：
    1. 持续检查所有推理任务是否完成
    2. 如果任何任务需要裁判员模型，则终止推理引擎
    3. 在允许GPU资源释放的时间后启动裁判员模型进程

    参数:
        config (dict): 包含推理评估配置的字典

    返回：
        无
        
    异常：
        RuntimeError: 如果在监控过程中推理引擎意外退出
    
    注意：
        - 如果未设置judge_model_port，函数将立即退出
        - 当所有任务完成推理且裁判员模型（如需要）启动后，函数退出
        - 在终止推理后等待60秒再启动裁判员模型
    """
    while True:
        await asyncio.sleep(5)
        if not os.environ.get('EXTERNAL_API') and not check_inference_engine_running():
            raise RuntimeError(f'推理引擎已退出！')
        # import pdb;pdb.set_trace()
        if all(t.get('inference') for t in task_status.values()):  # 全部任务都已推理完成
            if any(t.get('judge') is False for t in task_status.values()):  # 任意任务需要裁判员模型
                logger.info('测试集推理完成，正在停止推理引擎，释放显卡，以启动裁判员模型')
                if judge_model_port:
                    await terminate_inference_engine(remote_model_port)  # 终止模型推理引擎，腾空显卡
                    logger.info('正在启动裁判员模型')
                    await send_request(None, 'post', judge_model_port, 'start')  # 启动裁判员模型推理进程
                elif os.environ.get('EXTERNAL_API'):
                    for k, v in judge_status.items():
                        logger.info('EXTERNAL_API推理结束 正在启动裁判员模型')
                        v['port'] = await start_inference_engine(
                            model=k,
                            tensor_parallel=v.get('judge_tensor_parallel') or config.get('judge_tensor_parallel'),
                            max_length=config.get('judge_max_length'),
                            max_new_tokens=config.get('judge_max_new_tokens'),
                            backend=config.get('judge_backend'),
                            engine_args=config.get('judge_engine_args'),
                        )
                        while v.get('n_done', 0) < v['n_task']:
                            await asyncio.sleep(5)
                else:
                    for k, v in judge_status.items():
                        logger.info('正在启动裁判员模型')
                        v['port'] = await restart_inference_engine(
                            model=k,
                            tensor_parallel=v.get('judge_tensor_parallel') or config.get('judge_tensor_parallel'),
                            max_length=config.get('judge_max_length'),
                            max_new_tokens=config.get('judge_max_new_tokens'),
                            backend=config.get('judge_backend'),
                            engine_args=config.get('judge_engine_args'),
                        )
                        while v.get('n_done', 0) < v['n_task']:
                            await asyncio.sleep(5)
            break


async def run_loss_eval(config, task, model_path):
    cmd = "python -u {} --model_name_or_path {} --template {} --task {} --split test --lang default --n_shot 0 --batch_size 8 --save_dir {} --task_path {} --eval_type {} && ".format(os.path.join(cur_path, "eval.py"), model_path,config['prompt_type'],task,config['save_dir'],config['tasks'][task]['data_path'],config['tasks'][task]['type'])
    cmd += "python {} --eval_func {} --input_path '{}' --output_path '{}'\n".format(os.path.join(cur_path, "post_eval.py"),os.path.join(cur_path, "utils/eval_loss.py"), os.path.join(config['save_dir'],task+'.jsonl'),os.path.join(config['save_dir'],task+'.log'))
    await run_command(cmd)
    task_status[task]['inference'] = True  # 模型推理完成


async def run_next_word_prob_eval(config, task, model_path):
    cmd = "python -u {} --model_name_or_path {} --template {} --task {} --split test --lang default --n_shot 0 --batch_size 8 --save_dir {} --task_path {} --eval_type {} && ".format(os.path.join(cur_path, "eval.py"), model_path,config['prompt_type'],task,config['save_dir'],config['tasks'][task]['data_path'],config['tasks'][task]['type'])
    cmd += "python {} --eval_func {} --input_path '{}' --output_path '{}'\n".format(os.path.join(cur_path, "post_eval.py"),os.path.join(cur_path, "utils/eval_next_word_probability.py"), os.path.join(config['save_dir'],task+'.jsonl'),os.path.join(config['save_dir'],task+'.log'))
    await run_command(cmd)
    task_status[task]['inference'] = True  # 模型推理完成


async def run_mt_bench_eval(config, task):
    async def get_response(msg):
        result = await send_request({"prompt":{"instruction":msg}}, 'post', remote_model_port, path='model', json=True)
        result = await send_request({"prompt":{"instruction":msg}}, 'post', remote_model_port, path='model', json=True)
        return result

    with open(config['tasks'][task]['data_path'],'r') as f:
        data = [json.loads(l) for l in f]

    out = []
    for d in tqdm(data,desc="predict mtbench"):
        prompt_1 = d['turns'][0]
        result_1 = await get_response(prompt_1)
        
        d['response'] = [result_1.json()['output']] if result_1 else []

        prompt_2 = "Question:{}\nAnswer:{}\nQuestion:{}".format(d['turns'][0],d['response'][0],d['turns'][1])
        result_2 = await get_response(prompt_2)
        if result_2:
            d['response'].append(result_2.json()['output'])
        out.append(d)
    with open("{}/mtbench.jsonl".format(config['save_dir']),'w') as f:
        for o in out:
            f.write(json.dumps(o,ensure_ascii=False)+'\n')
    task_status[task]['inference'] = True  # 模型推理完成
    
    await run_post_eval(config, task, os.path.join(config['save_dir'], task+'.jsonl'))


async def run_text_eval(config, task):
    await run_task_inference(config, task)
    await correct_answer_format(config, task)
    task_status[task]['inference'] = True  # 模型推理完成

    if 'judge' in config['tasks'][task]:
        task_status[task]['judge'] = False  # 等待裁判员模型推理
        output_path = await run_judge_inference(config, task)
        output_path = ','.join(output_path)
    else:
        output_path = os.path.join(config['save_dir'], task + '.jsonl')

    await run_post_eval(config, task, output_path)


async def run_cosyvoice2_eval(config, task):
    await run_task_inference(config, task)
    task_status[task]['inference'] = True  # 模型推理完成
    if 'judge' in config['tasks'][task]:
        task_status[task]['judge'] = False  # 等待裁判员模型推理
        output_path = await run_judge_inference(config, task)
        output_path = ','.join(output_path)
    else:
        output_path = os.path.join(config['save_dir'], task + '.jsonl')
    await run_post_eval(config, task, output_path)


async def run_retrieval_eval(config, task):
    """向量检索（embedding模型）评估"""
    # 输入数据（包含query和docs的json文件）
    input_path = config['tasks'][task]['data_path']
    input_dir = os.path.dirname(input_path)
    data_files = [f for f in os.listdir(input_dir) if f.endswith('json') or f.endswith('jsonl')]
    if not os.path.basename(input_path).startswith('test'):
        if fs := [f for f in data_files if f.startswith('test')]:
            input_path = os.path.join(input_dir, fs[0])
    logger.info(f'检索query文件路径：{input_path}')

    # docs数据
    if fs := [f for f in data_files if 'docs' in f]:
        docs_path = os.path.join(input_dir, fs[0])
    else:
        raise RuntimeError(f'{input_dir}路径下找不到检索docs文件！')
    logger.info(f'检索docs文件路径：{docs_path}')

    # 计算embedding
    config['tasks'] = config['tasks'].copy()  # 局部修改
    config['tasks'][task + '-query'] = {'data_path': input_path, 'type': 'embedding'}
    config['tasks'][task + '-docs'] = {'data_path': docs_path, 'type': 'embedding'}
    query_emb_file, docs_emb_file = await asyncio.gather(
        run_task_inference(config, task + '-query'),
        run_task_inference(config, task + '-docs'),
    )
    task_status[task]['inference'] = True  # 模型推理完成

    # retrieval
    while not all(t.get('inference') for t in task_status.values()):  # 等待全部任务推理完成
        await asyncio.sleep(5)
    await restart_inference_engine(
        output_type='retrieval',
        backend='faiss',
        documents=docs_emb_file,
        preprocess='none',
    )
    config['tasks'][task]['data_path'] = query_emb_file
    output_path = await run_task_inference(config, task)
    await run_post_eval(config, task, output_path)


async def run_mcp_eval(config, task):
    """MCP（工具调用）评估"""
    from tools.mcp_client import run_mcp_inference,run_wencai_plan_inference,run_wencai_summary_inference
    method = config['tasks'][task].get('method','')
    if method == 'wencai':
        output_path = await run_wencai_plan_inference(config,task)
        await restart_inference_engine(
            model=config['tasks'][task]['summary_model'].get('summary_model_path',''),
            tensor_parallel=config['tasks'][task]['summary_model'].get('summary_model_parallel',''),
        )
        summary_path = await run_wencai_summary_inference(config,task,output_path)
    else:
        output_path = await run_mcp_inference(config,task)
    await run_command("bash SandBoxEnv/stop_sandboxenv.sh")
    task_status[task]['inference'] = True  # 模型推理完成
    await run_post_eval(config, task, output_path)


async def run_post_eval(config, task, output_path):
    func_path = config['tasks'][task]['compare_func']['path']
    params = config['tasks'][task]['compare_func'].get('params')
    params = " --kwargs '{}'".format(json.dumps(params)) if params else ""
    cmd_eval = "python {} --eval_func {} --input_path '{}' --output_path '{}'{}\n".format(os.path.join(cur_path, "post_eval.py"),
        func_path, output_path, os.path.join(config['save_dir'], task + '.log'), params
    )
    print(cmd_eval)
    await run_command(cmd_eval)


def determine_task_type(task_config: dict):
    """根据评估函数，确定任务类型"""
    compare_func_path = task_config.get('compare_func', {}).get('path')
    if compare_func_path:
        compare_func_path = os.path.basename(compare_func_path)
        if compare_func_path.startswith('eval_turn_taking'):
            task_config['type'] = 'turn_taking'
        elif compare_func_path.startswith('eval_retrieval'):
            task_config['type'] = 'retrieval'
        elif compare_func_path.startswith('eval_mcp'):
            task_config['type'] = 'mcp'


async def llm_eval(file_path, model_name=""):
    """模型评估

    Args:
        file_path (str): config地址
        model_name (int): 可选参数，存储模型的sub path

    Returns:
        None

    Examples:
        >>> llm_eval(path, model_name)
        None
    """
    with open(file_path, 'r') as file:
        config = yaml.safe_load(file)
    config["save_dir"] = config["save_dir"] + f"/{model_name}"
    print(config)
    model_path = os.getenv('MODEL_PATH')
    os.makedirs(config["save_dir"], exist_ok=True)

    tasks_async = {}
    
    for task in config['tasks']:
        logger.info(f'开始评测任务：{task}')
        task_status[task] = {}

        determine_task_type(config['tasks'][task])
        if config['tasks'][task]['type'] in ['loss']:
            coro = run_loss_eval(config, task, model_path)
        elif config['tasks'][task]['type'] in ['next_word_probability']:
            coro = run_next_word_prob_eval(config, task, model_path)
        elif config['tasks'][task]['type'] in ['text', 'turn_taking']:
            if task == 'mtbench':
                coro = run_mt_bench_eval(config, task)
            else:
                coro = run_text_eval(config, task)
        elif config['tasks'][task]['type'] in ['cosyvoice2']:
            coro = run_cosyvoice2_eval(config, task)
        elif config['tasks'][task]['type'] in ['retrieval']:
            coro = run_retrieval_eval(config, task)
        elif config['tasks'][task]['type'] in ['mcp']:
            #start mcp server
            await run_server('java -jar SandBoxEnv/output/SandBoxEnv-0.0.4.jar --spring.profiles.active=dsw > SandBoxEnv.log 2>&1')
            #TODO tmp solution, wait for server ready, need a better way to check
            await asyncio.sleep(5)
            coro = run_mcp_eval(config, task)

        tasks_async[task] = asyncio.create_task(coro)
        await asyncio.sleep(1)

    await asyncio.gather(
        maybe_start_judge_model(config),
        *tasks_async.values()
    )

    statistic(config["save_dir"],config)


async def terminate_inference_engine(port=None, ignore_errors=False):
    if port:
        ports = [port]
    else:
        ports = [remote_model_port, judge_model_port] + [v['port'] for v in judge_status.values() if 'port' in v]
    for port in ports:
        if port and not model_status.get(port, {}).get('terminated'):
            await send_request(None, 'post', port, 'terminate', ignore_errors=ignore_errors)
            model_status.setdefault(port, {})['terminated'] = True
    await wait_for_inference_engine_exit(ports)


async def wait_for_inference_engine_exit(ports=None):
    ports = ports or [remote_model_port, judge_model_port]
    while any(check_inference_engine_running(p) for p in ports if p):
        await asyncio.sleep(5)


async def main(args):
    is_node_0 = os.getenv('WORLD_SIZE', '1') == '1' or os.getenv('RANK', '0') == '0'
    if is_node_0:
        global session
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout()) as session:  # 取消超时限制（默认5分钟）
            try:
                await llm_eval(args.config, args.model_name)
                await terminate_inference_engine()
            except:
                await terminate_inference_engine(ignore_errors=True)
                raise
    else:  # worker节点只需要等待推理进程退出
        await wait_for_inference_engine_exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="File path of config yaml")
    parser.add_argument("--model_name", type=str, required=False, default="", help="model name that is evaluated")
    args = parser.parse_args()
    ## 添加判断，知道推理服务启动再开始评估
    t0 = time.time()
    while True:
        logger.info("infer engine initializing {:.2f} s".format(time.time()-t0))
        if time.time()-t0 > 20:
            asyncio.run(main(args))
            break
        time.sleep(5)
